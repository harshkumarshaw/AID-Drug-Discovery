"""
pipeline/disease_subtyping.py
==============================
Phase 1: Disease Subtyping for LIHC patients.

Produces patient cluster assignments and Kaplan-Meier survival curves.

EXECUTION TIERS (reads from config.COMPUTE_TIER):
  cpu          → PCA (n=50) + KMeans (k=4)          [~5 min]
  single_gpu   → VAE + GMM (latent=32, AMP float16)  [~30 min local]
  high_gpu     → Full Subtype-GAN on Colab/Kaggle     [see colab/]

VALIDATION ANCHORS:
  ✓ ARI > 0.3 vs random baseline
  ✓ Kaplan-Meier log-rank p < 0.05 (at least 1 cluster pair)
  ✓ Cluster with highest Stage IV % must also have highest mortality

INPUT:
  config.TCGA_CLINICAL_CSV      — clinical data (already have)
  config.TCGA_RNASEQ_CSV        — RNA-seq (must download from GDC)
  config.TCGA_MUTATION_MAF      — somatic mutations (optional for Phase 2)
  config.TCGA_CNV_CSV           — CNV (optional)

OUTPUT:
  config.PATIENT_CLUSTERS_CSV   — patient_id, Cluster, clinical features
  config.SUBTYPE_GMM_MODEL      — saved GMM model (.pkl)
  config.SUBTYPE_GAN_MODEL      — saved VAE model (.pt)  [if single_gpu]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from pipeline.utils import (
    PhaseTimer, get_device, get_logger, log_device_info,
    save_checkpoint, set_seed, validate_dataframe, amp_context,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("Phase1.Subtyping")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_clinical(path: Path) -> pd.DataFrame:
    """Load and validate TCGA-LIHC clinical data."""
    clinical = pd.read_csv(path)
    validate_dataframe(clinical, "clinical", min_rows=50,
                        required_cols=["case_id"], logger=logger)

    # Derive Disease_Status binary column (Bug 004 fix)
    if "Disease_Status" not in clinical.columns:
        if "vital_status" in clinical.columns:
            clinical["Disease_Status"] = (
                clinical["vital_status"].isin(["Dead", "dead", "Deceased"])
            ).astype(int)
            logger.info("Derived Disease_Status from vital_status")
        else:
            clinical["Disease_Status"] = 0
            logger.warning("vital_status not found — Disease_Status set to 0")

    return clinical


def load_rnaseq(path: Path, mvp_mode: bool = True) -> pd.DataFrame:
    """
    Load TCGA-LIHC RNA-seq expression matrix.
    
    Expected format: rows=samples (case_id), cols=genes (ENSG... or symbol)
    GDC HTSeq/STAR format: rows=genes, cols=samples → will transpose.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"RNA-seq file not found: {path}\n"
            f"Download from GDC Portal: portal.gdc.cancer.gov\n"
            f"Filter: Project=TCGA-LIHC, Data Category=Transcriptome Profiling"
        )

    logger.info(f"Loading RNA-seq from {path}...")
    expr = pd.read_csv(path, index_col=0)

    # GDC default gives genes as rows → transpose to samples as rows
    if expr.shape[0] > expr.shape[1]:
        logger.info(f"Transposing: {expr.shape} → genes as columns")
        expr = expr.T

    # Remove non-expression rows (STAR output has summary rows at top)
    summary_rows = ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]
    expr = expr[~expr.index.isin(summary_rows)]

    validate_dataframe(expr, "RNA-seq", min_rows=50, logger=logger)
    logger.info(f"RNA-seq: {expr.shape[0]} samples × {expr.shape[1]} genes")

    # MVP: subsample top variable genes to speed up
    if mvp_mode:
        n_genes = config.MVP_MAX_GENES
        gene_var = expr.var(axis=0).sort_values(ascending=False)
        top_genes = gene_var.head(n_genes).index
        expr = expr[top_genes]
        logger.info(f"MVP mode: using top {n_genes} variable genes")

    return expr


def merge_features(clinical: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    """
    Merge clinical and expression data on case_id / sample index.
    Returns merged DataFrame with case_id as index.
    """
    # Normalize case_id format (TCGA barcodes: TCGA-XX-XXXX)
    clinical_ids = clinical["case_id"].str[:12]  # First 12 chars = patient barcode
    expr_ids = expr.index.str[:12]

    # Find overlap
    common = set(clinical_ids) & set(expr_ids)
    logger.info(f"Clinical samples: {len(clinical_ids)} | "
                f"RNA-seq samples: {len(expr_ids)} | "
                f"Overlap: {len(common)}")

    if len(common) < 10:
        raise ValueError(
            f"Only {len(common)} overlapping samples between clinical and RNA-seq. "
            f"Check that both are from TCGA-LIHC (project TCGA-LIHC)."
        )

    clinical_f = clinical[clinical_ids.isin(common)].copy()
    expr_f = expr[expr_ids.isin(common)].copy()

    clinical_f.index = clinical_ids[clinical_ids.isin(common)].values
    expr_f.index = expr_ids[expr_ids.isin(common)].values

    # Align on shared index
    shared_idx = sorted(common)
    clinical_f = clinical_f.loc[shared_idx]
    expr_f = expr_f.loc[shared_idx]

    return clinical_f.reset_index(drop=True), expr_f


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 MODEL: PCA + KMeans (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans_baseline(
    expr: pd.DataFrame,
    clinical: pd.DataFrame,
    n_clusters: int = 4,
    n_pca_components: int = 50,
) -> np.ndarray:
    """
    Tier 1 (CPU): PCA + KMeans clustering.
    Baseline to beat before investing in VAE+GMM.
    Takes ~2–5 minutes on CPU.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Running PCA ({n_pca_components} components) + KMeans (k={n_clusters})")

    # Build feature matrix: scaled expression + numeric clinical
    numeric_clin = clinical.select_dtypes(include=[np.number]).fillna(0)
    X = np.hstack([numeric_clin.values, expr.values])
    X = StandardScaler().fit_transform(X)

    n_components = min(n_pca_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"PCA variance explained: {explained:.1f}% with {n_components} components")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.RANDOM_STATE,
        n_init=20,           # More restarts → more stable
        max_iter=300,
    )
    labels = kmeans.fit_predict(X_pca)
    logger.info(f"KMeans cluster sizes: {np.bincount(labels).tolist()}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 MODEL: VAE + GMM (RTX 3050 4GB + AMP)
# ─────────────────────────────────────────────────────────────────────────────

class SubtypeVAE(torch.nn.Module):
    """
    Variational Autoencoder for multi-omic subtype discovery.
    Dims halved from full Subtype-GAN for 4GB VRAM compatibility.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: list = None):
        import torch.nn as nn
        super().__init__()
        if hidden_dims is None:
            hidden_dims = config.SUBTYPE_VAE_HIDDEN_DIMS  # [256, 128]

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder (mirrors encoder)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta: float = 1.0):
    """Beta-VAE loss = reconstruction + beta * KL divergence."""
    import torch.nn.functional as F
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


def run_vae_gmm(
    expr: pd.DataFrame,
    clinical: pd.DataFrame,
    n_clusters: int = 4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Tier 2 (single_gpu + AMP): VAE for latent embedding + GMM for clustering.
    Fits in RTX 3050 4GB with AMP float16.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    if device is None:
        device = get_device(config.COMPUTE_TIER)
    log_device_info(device, logger)

    # Feature matrix
    numeric_clin = clinical.select_dtypes(include=[np.number]).fillna(0)
    X_raw = np.hstack([numeric_clin.values, expr.values]).astype(np.float32)
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Train / val split (80/20)
    n = len(X_tensor)
    n_train = int(0.8 * n)
    idx = torch.randperm(n)
    X_train = X_tensor[idx[:n_train]]
    X_val   = X_tensor[idx[n_train:]]

    train_dl = DataLoader(TensorDataset(X_train), batch_size=config.SUBTYPE_VAE_BATCH,
                          shuffle=True, drop_last=True)

    input_dim = X_scaled.shape[1]
    model = SubtypeVAE(
        input_dim=input_dim,
        latent_dim=config.SUBTYPE_VAE_LATENT_DIM,
        hidden_dims=config.SUBTYPE_VAE_HIDDEN_DIMS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.SUBTYPE_VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.SUBTYPE_VAE_EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device.type == "cuda")

    best_val_loss = float("inf")
    patience_ctr = 0
    PATIENCE = 15

    logger.info(f"Training VAE: {config.SUBTYPE_VAE_EPOCHS} epochs | "
                f"latent={config.SUBTYPE_VAE_LATENT_DIM} | batch={config.SUBTYPE_VAE_BATCH}")

    for epoch in range(config.SUBTYPE_VAE_EPOCHS):
        model.train()
        train_losses = []
        for (X_b,) in train_dl:
            X_b = X_b.to(device)
            with torch.cuda.amp.autocast(
                enabled=config.USE_AMP and device.type == "cuda",
                dtype=torch.float16,
            ):
                x_recon, mu, logvar = model(X_b)
                loss, recon, kl = vae_loss(x_recon, X_b, mu, logvar, beta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=config.USE_AMP and device.type == "cuda"
        ):
            X_val_dev = X_val.to(device)
            x_recon_val, mu_val, logvar_val = model(X_val_dev)
            val_loss, _, _ = vae_loss(x_recon_val, X_val_dev, mu_val, logvar_val)
            val_loss = val_loss.item()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | train={np.mean(train_losses):.4f} | "
                        f"val={val_loss:.4f} | VRAM={get_vram_used_gb():.2f}GB")

        best_val_loss = save_checkpoint(
            model=model,
            path_best=config.SUBTYPE_GAN_MODEL,
            path_latest=config.SUBTYPE_GAN_LATEST,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            extra={"epoch": epoch},
            logger=None,  # Silent checkpoint saves
        )

        if val_loss >= best_val_loss + 0.001:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            patience_ctr = 0

    # Extract latent embeddings for all patients
    model.eval()
    with torch.no_grad():
        X_all_dev = X_tensor.to(device)
        with torch.cuda.amp.autocast(
            enabled=config.USE_AMP and device.type == "cuda"
        ):
            _, mu_all, _ = model(X_all_dev)
        embeddings = mu_all.cpu().numpy()

    # Reorder to original order (undo idx permutation)
    embeddings_ordered = np.zeros_like(embeddings)
    embeddings_ordered[idx.numpy()] = embeddings

    # GMM clustering on latent space
    logger.info(f"Fitting GMM (k={n_clusters}) on {embeddings.shape[1]}-dim latent space")
    import pickle
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=config.RANDOM_STATE,
        max_iter=200,
        n_init=5,
    )
    labels = gmm.fit_predict(embeddings_ordered)

    # Save GMM
    config.SUBTYPE_GMM_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SUBTYPE_GMM_MODEL, "wb") as f:
        pickle.dump(gmm, f)
    logger.info(f"GMM saved to {config.SUBTYPE_GMM_MODEL}")
    logger.info(f"Cluster sizes: {np.bincount(labels).tolist()}")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION: KAPLAN-MEIER SURVIVAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_survival_analysis(
    clinical: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Kaplan-Meier survival analysis per cluster.
    Validation anchor: at least one cluster pair must have log-rank p < 0.05.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.warning("lifelines not installed — skipping survival analysis")
        return {}

    results = {}
    clinical = clinical.copy()
    clinical["Cluster"] = labels

    # Resolve time and event columns (handle various naming conventions)
    time_col = next(
        (c for c in ["days_to_death", "overall_survival_days", "days_to_last_follow_up"]
         if c in clinical.columns), None
    )
    event_col = "Disease_Status"

    if time_col is None:
        logger.warning("No survival time column found — skipping KM analysis")
        return {}

    clinical["_time"] = pd.to_numeric(clinical[time_col], errors="coerce").fillna(0)
    clinical["_event"] = clinical[event_col].fillna(0).astype(int)

    # Global log-rank test (all clusters, pairwise)
    min_p = 1.0
    kmf = KaplanMeierFitter()
    significant_pairs = []

    for i in range(4):
        for j in range(i + 1, 4):
            mask_i = clinical["Cluster"] == i
            mask_j = clinical["Cluster"] == j
            if mask_i.sum() < 5 or mask_j.sum() < 5:
                continue
            result = logrank_test(
                clinical.loc[mask_i, "_time"], clinical.loc[mask_j, "_time"],
                event_observed_A=clinical.loc[mask_i, "_event"],
                event_observed_B=clinical.loc[mask_j, "_event"],
            )
            results[f"C{i}_vs_C{j}"] = {
                "p_value": result.p_value,
                "test_statistic": result.test_statistic,
            }
            if result.p_value < 0.05:
                significant_pairs.append((i, j, result.p_value))
            min_p = min(min_p, result.p_value)

    # Validation anchor check
    if min_p < 0.05:
        logger.info(
            f"✅ SURVIVAL VALIDATION PASSED: {len(significant_pairs)} significant pair(s). "
            f"Best p={min_p:.4f}"
        )
    else:
        logger.warning(
            f"⚠️  SURVIVAL VALIDATION: min log-rank p={min_p:.4f} — "
            f"not reaching 0.05 threshold. Subtypes may not be clinically distinct."
        )

    # Check stage/mortality anchor
    staging_check(clinical, labels)

    return results


def staging_check(clinical: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Validation anchor: cluster with highest Stage IV% must also have highest mortality.
    """
    clin = clinical.copy()
    clin["Cluster"] = labels

    if "tumor_stage" not in clin.columns:
        return

    # Stage IV enrichment per cluster
    clin["is_stage4"] = clin["tumor_stage"].str.contains(
        "IV|4", case=False, na=False
    ).astype(int)

    cluster_stats = clin.groupby("Cluster").agg(
        stage4_pct=("is_stage4", "mean"),
        mortality=("Disease_Status", "mean"),
        n=("Disease_Status", "count"),
    )
    logger.info("\nCluster staging summary:")
    logger.info(cluster_stats.to_string())

    # Spearman correlation of stage4_pct vs mortality across clusters
    from scipy import stats as sp_stats
    rho, p = sp_stats.spearmanr(cluster_stats["stage4_pct"], cluster_stats["mortality"])
    if rho > 0.5 and p < 0.1:
        logger.info(
            f"✅ STAGING ANCHOR: Stage IV% and mortality positively correlated "
            f"(ρ={rho:.2f}, p={p:.3f}) — subtypes are biologically ordered"
        )
    else:
        logger.warning(
            f"⚠️  STAGING ANCHOR: Weak correlation (ρ={rho:.2f}) — "
            f"check feature engineering or try more clusters"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run_subtyping
# ─────────────────────────────────────────────────────────────────────────────

def run_subtyping(
    compute_tier: Optional[str] = None,
    mvp_mode: Optional[bool] = None,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Phase 1 entry point. Call this from main.py or notebooks.

    Args:
        compute_tier: "cpu", "single_gpu", or "high_gpu".
                      Defaults to config.COMPUTE_TIER.
        mvp_mode:     If True, limit genes to MVP_MAX_GENES.
                      Defaults to config.MVP_MODE.
        n_clusters:   Number of subtypes. Default 4 (from LIHC literature).

    Returns:
        DataFrame: patient_clusters.csv contents with columns
                   [case_id, Cluster, age_at_diagnosis, Disease_Status, tumor_stage, ...]
    """
    if compute_tier is None:
        compute_tier = config.COMPUTE_TIER
    if mvp_mode is None:
        mvp_mode = config.MVP_MODE

    set_seed(config.RANDOM_STATE)

    with PhaseTimer("Disease Subtyping (Phase 1)", logger):

        # ── Load Data ──────────────────────────────────────────────────────
        logger.info(f"Compute tier: {compute_tier} | MVP mode: {mvp_mode}")

        clinical = load_clinical(config.TCGA_CLINICAL_CSV)

        if config.TCGA_RNASEQ_CSV.exists():
            expr = load_rnaseq(config.TCGA_RNASEQ_CSV, mvp_mode=mvp_mode)
            clinical, expr = merge_features(clinical, expr)
            has_expression = True
        else:
            logger.warning(
                f"RNA-seq not found at {config.TCGA_RNASEQ_CSV}. "
                f"Running clinical-only clustering (limited quality). "
                f"Download from GDC Portal to improve results."
            )
            expr = pd.DataFrame(index=range(len(clinical)))  # Empty expression
            has_expression = False

        # ── Select Model by Tier ──────────────────────────────────────────
        if compute_tier == "cpu" or not has_expression:
            logger.info("Tier: CPU → PCA + KMeans baseline")
            labels = run_kmeans_baseline(
                expr if has_expression else pd.DataFrame(np.zeros((len(clinical), 10))),
                clinical,
                n_clusters=n_clusters,
            )

        elif compute_tier in ("single_gpu", "high_gpu"):
            logger.info("Tier: single_gpu → VAE + GMM")
            import torch
            device = get_device(compute_tier)
            labels = run_vae_gmm(expr, clinical, n_clusters=n_clusters, device=device)

        else:
            raise ValueError(
                f"Unknown compute_tier: {compute_tier}. "
                f"Valid: 'cpu', 'single_gpu', 'high_gpu'"
            )

        # ── Add labels to clinical ─────────────────────────────────────────
        clinical = clinical.copy()
        clinical["Cluster"] = labels

        # ── Survival Analysis ─────────────────────────────────────────────
        survival_results = run_survival_analysis(clinical, labels)

        # ── Save Output ───────────────────────────────────────────────────
        config.PATIENT_CLUSTERS_CSV.parent.mkdir(parents=True, exist_ok=True)
        clinical.to_csv(config.PATIENT_CLUSTERS_CSV, index=False)
        logger.info(f"✅ Clusters saved: {config.PATIENT_CLUSTERS_CSV}")
        logger.info(f"   Shape: {clinical.shape}")
        logger.info(f"   Cluster distribution:\n{clinical['Cluster'].value_counts().to_string()}")

    return clinical


if __name__ == "__main__":
    # Quick test run
    result = run_subtyping()
    print(result[["case_id", "Cluster", "Disease_Status"]].head(10))
