"""
scripts/create_notebooks.py
============================
Creates all 6 phase notebooks (Phase 01-06) as proper .ipynb files.
Run this script once to generate the notebooks directory.

Usage: python scripts/create_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

def cell_md(text: str) -> dict:
    lines = [l + "\n" for l in text.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def cell_code(text: str, tags: list = None) -> dict:
    lines = [l + "\n" for l in text.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    meta = {}
    if tags:
        meta["tags"] = tags
    return {"cell_type": "code", "execution_count": None,
            "metadata": meta, "outputs": [], "source": lines}

def make_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def save_nb(nb: dict, name: str):
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Disease Subtyping
# ─────────────────────────────────────────────────────────────────────────────
p1 = make_notebook([
    cell_md("""# Phase 1 — Disease Subtyping
**Goal:** Cluster LIHC patients into molecular subtypes using multi-omic data.

| Tier | Compute | Model | Expected Time |
|---|---|---|---|
| MVP | CPU | PCA + KMeans | ~5 min |
| Tier 2 | RTX 3050 (AMP float16) | VAE + GMM | ~30 min |
| Paper | Kaggle P100 | Subtype-GAN | ~3 hr |

**Validation Anchor:** Kaplan-Meier log-rank p < 0.05 between at least one cluster pair.
"""),

    cell_code("""# ── Environment check ─────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # so imports find config.py

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  |  VRAM: {props.total_memory/1e9:.1f} GB")
else:
    print("Running on CPU — set COMPUTE_TIER = 'cpu' below")
"""),

    cell_code("""# ── Config ────────────────────────────────────────────────────────────
import config

# ✏️  Edit these to match your setup:
COMPUTE_TIER = config.COMPUTE_TIER          # 'cpu' | 'single_gpu' | 'high_gpu'
MVP_MODE     = config.MVP_MODE              # True = top 5000 genes only
N_CLUSTERS   = config.N_CLUSTERS           # 4 for LIHC (literature consensus)
RANDOM_STATE = config.RANDOM_STATE

print(f"Compute tier : {COMPUTE_TIER}")
print(f"MVP mode     : {MVP_MODE}")
print(f"N clusters   : {N_CLUSTERS}")
print(f"Clinical CSV : {config.TCGA_CLINICAL_CSV}")
print(f"RNA-seq CSV  : {config.TCGA_RNASEQ_CSV}  (exists={config.TCGA_RNASEQ_CSV.exists()})")
"""),

    cell_md("""## 1. Load TCGA-LIHC Data

**Required files:**
- `data/processed/cleaned_clinical_data.csv` ✅ already have
- `data/processed/tcga_lihc_rnaseq.csv` ⬇️ download from GDC Portal

> If RNA-seq is missing, the notebook will run clinical-only clustering (limited quality).
"""),

    cell_code("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Clinical data ────────────────────────────────────────────────────
clinical = pd.read_csv(config.TCGA_CLINICAL_CSV)

# Derive survival outcome column
if 'Disease_Status' not in clinical.columns:
    if 'vital_status' in clinical.columns:
        clinical['Disease_Status'] = (
            clinical['vital_status'].isin(['Dead', 'dead', 'Deceased'])
        ).astype(int)
        print("Derived Disease_Status from vital_status")
    else:
        clinical['Disease_Status'] = 0

print(f"Clinical:  {clinical.shape[0]} patients × {clinical.shape[1]} features")
print(f"Columns:   {list(clinical.columns[:10])}")
print(f"Mortality: {clinical['Disease_Status'].mean()*100:.1f}%")
clinical.head(3)
"""),

    cell_code("""# ── RNA-seq expression data ───────────────────────────────────────────
HAS_EXPRESSION = config.TCGA_RNASEQ_CSV.exists()

if HAS_EXPRESSION:
    expr_raw = pd.read_csv(config.TCGA_RNASEQ_CSV, index_col=0)
    
    # GDC default format: genes as rows, samples as cols → transpose
    if expr_raw.shape[0] > expr_raw.shape[1]:
        expr_raw = expr_raw.T

    # Remove HTSeq summary rows
    summary_rows = ['N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous']
    expr_raw = expr_raw[~expr_raw.index.isin(summary_rows)]
    
    print(f"RNA-seq raw: {expr_raw.shape[0]} samples × {expr_raw.shape[1]} genes")
    
    # MVP: keep top N most variable genes
    if MVP_MODE:
        N_GENES = config.MVP_MAX_GENES
        top_genes = expr_raw.var(axis=0).nlargest(N_GENES).index
        expr_raw  = expr_raw[top_genes]
        print(f"MVP mode: trimmed to {N_GENES} most variable genes")
    
    expr_raw.head(2)
else:
    print("⚠️  RNA-seq not found. Download from GDC portal:")
    print("   https://portal.gdc.cancer.gov/ → Cases → TCGA-LIHC")
    print("   Files → Transcriptome Profiling → STAR Counts → Add to Cart")
    expr_raw = pd.DataFrame()
    print("Continuing with clinical features only...")
"""),

    cell_md("## 2. Exploratory Data Analysis"),

    cell_code("""import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Mortality distribution
clinical['Disease_Status'].value_counts().plot(
    kind='bar', ax=axes[0], color=['#52b788', '#e63946'], edgecolor='white')
axes[0].set_title('Vital Status', fontweight='bold')
axes[0].set_xticklabels(['Alive', 'Dead'], rotation=0)

# Age distribution
if 'age_at_diagnosis' in clinical.columns:
    clinical['age_at_diagnosis'].hist(bins=20, ax=axes[1], color='#4361ee', edgecolor='white')
    axes[1].set_title('Age at Diagnosis', fontweight='bold')
    axes[1].set_xlabel('Age (years)')

# Tumor stage (if available)
stage_col = next((c for c in ['tumor_stage','ajcc_pathologic_stage'] if c in clinical.columns), None)
if stage_col:
    stage_counts = clinical[stage_col].value_counts().head(6)
    stage_counts.plot(kind='barh', ax=axes[2], color='#7209b7', edgecolor='white')
    axes[2].set_title('Tumor Stage Distribution', fontweight='bold')

plt.suptitle('TCGA-LIHC Clinical Overview', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../docs/eda_clinical.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Total patients: {len(clinical)}")
"""),

    cell_code("""# Gene expression overview (if available)
if HAS_EXPRESSION and len(expr_raw) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distribution of mean expression per sample
    sample_means = expr_raw.mean(axis=1)
    sample_means.hist(bins=30, ax=axes[0], color='#4cc9f0', edgecolor='white')
    axes[0].set_title('Mean Expression per Sample', fontweight='bold')
    axes[0].set_xlabel('Mean log-TPM')

    # Top 20 most variable genes
    gene_vars = expr_raw.var(axis=0).nlargest(20)
    gene_vars.plot(kind='bar', ax=axes[1], color='#f72585', edgecolor='white')
    axes[1].set_title('Top 20 Most Variable Genes', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    print(f"NaN in expression matrix: {expr_raw.isna().sum().sum()}")
else:
    print("No expression data to visualize")
"""),

    cell_md("""## 3. Feature Engineering & Sample Alignment

Merge clinical and expression data on TCGA patient barcodes.
TCGA barcodes: `TCGA-XX-XXXX-XXX` — we match on first 12 characters (patient level).
"""),

    cell_code("""from sklearn.preprocessing import StandardScaler

# ── Align clinical + expression on patient barcodes ──────────────────
if HAS_EXPRESSION and len(expr_raw) > 0:
    # Normalize barcode to 12 chars (patient level)
    clin_ids  = clinical['case_id'].str[:12]
    expr_ids  = expr_raw.index.str[:12]
    
    common = sorted(set(clin_ids) & set(expr_ids))
    print(f"Clinical patients : {len(clin_ids)}")
    print(f"Expression samples: {len(expr_ids)}")
    print(f"Overlapping       : {len(common)}")
    
    if len(common) < 10:
        raise ValueError(
            f"Only {len(common)} overlapping samples! "
            "Check that clinical and RNA-seq are both from TCGA-LIHC."
        )
    
    clin_f = clinical[clin_ids.isin(common)].copy()
    expr_f = expr_raw[expr_ids.isin(common)].copy()
    clin_f.index  = clin_ids[clin_ids.isin(common)].values
    expr_f.index  = expr_ids[expr_ids.isin(common)].values
    clin_f = clin_f.loc[sorted(common)]
    expr_f = expr_f.loc[sorted(common)]
    
    # Combined feature matrix (numeric clinical + expression)
    numeric_clin = clin_f.select_dtypes(include=[np.number]).fillna(0)
    X_raw = np.hstack([numeric_clin.values, expr_f.values]).astype(np.float32)
    print(f"Feature matrix: {X_raw.shape[0]} samples × {X_raw.shape[1]} features")

else:
    # Clinical-only fallback
    clin_f = clinical.copy()
    numeric_clin = clin_f.select_dtypes(include=[np.number]).fillna(0)
    X_raw = numeric_clin.values.astype(np.float32)
    print(f"Clinical-only feature matrix: {X_raw.shape}")

X_scaled = StandardScaler().fit_transform(X_raw)
print(f"Scaled: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")
"""),

    cell_md("""## 4. Model — Tier Selection

| `COMPUTE_TIER` | Model | Details |
|---|---|---|
| `'cpu'` | PCA (50 components) + KMeans | Fast, interpretable baseline |
| `'single_gpu'` | VAE (latent=32) + GMM | 4GB VRAM safe with AMP |
| `'high_gpu'` | Subtype-GAN | See `colab/colab_phase1_subtype_gan.py` |
"""),

    cell_code("""import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── TIER 1: PCA + KMeans (always run as baseline) ─────────────────────
np.random.seed(RANDOM_STATE)

n_components = min(50, X_scaled.shape[0]-1, X_scaled.shape[1])
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {n_components} components explain "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_pca)
print(f"KMeans cluster sizes: {np.bincount(kmeans_labels).tolist()}")
"""),

    cell_code("""# ── TIER 2: VAE + GMM (skip if COMPUTE_TIER == 'cpu') ─────────────────
vae_labels = None

if COMPUTE_TIER in ('single_gpu', 'high_gpu') and torch.cuda.is_available():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.mixture import GaussianMixture
    import pickle

    device = torch.device('cuda')
    print(f"Training VAE on {torch.cuda.get_device_name(0)}")

    # ── Model Definition ────────────────────────────────────────────────
    class SubtypeVAE(nn.Module):
        def __init__(self, in_dim, latent_dim=32, hidden=[256, 128]):
            super().__init__()
            enc, dec = [], []
            prev = in_dim
            for h in hidden:
                enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
                prev = h
            self.encoder = nn.Sequential(*enc)
            self.fc_mu     = nn.Linear(prev, latent_dim)
            self.fc_logvar = nn.Linear(prev, latent_dim)
            prev = latent_dim
            for h in reversed(hidden):
                dec += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            dec.append(nn.Linear(prev, in_dim))
            self.decoder = nn.Sequential(*dec)

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, lv):
            std = torch.exp(0.5 * lv)
            return mu + std * torch.randn_like(std) if self.training else mu

        def forward(self, x):
            mu, lv = self.encode(x)
            return self.decoder(self.reparameterize(mu, lv)), mu, lv

    def vae_loss(xr, x, mu, lv, beta=1.0):
        import torch.nn.functional as F
        return F.mse_loss(xr, x) + beta * (-0.5 * torch.mean(1 + lv - mu**2 - lv.exp()))

    # ── Training ─────────────────────────────────────────────────────────
    X_t   = torch.tensor(X_scaled, dtype=torch.float32)
    n     = len(X_t)
    idx   = torch.randperm(n)
    n_tr  = int(0.8 * n)
    dl    = DataLoader(TensorDataset(X_t[idx[:n_tr]]),
                       batch_size=config.SUBTYPE_VAE_BATCH, shuffle=True, drop_last=True)

    model = SubtypeVAE(X_scaled.shape[1], latent_dim=config.SUBTYPE_VAE_LATENT_DIM,
                       hidden=config.SUBTYPE_VAE_HIDDEN_DIMS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=config.SUBTYPE_VAE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.SUBTYPE_VAE_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_val, best_ep = float('inf'), 0
    config.SUBTYPE_GAN_MODEL.parent.mkdir(parents=True, exist_ok=True)

    losses = []
    for epoch in range(config.SUBTYPE_VAE_EPOCHS):
        model.train()
        ep_loss = []
        for (xb,) in dl:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                xr, mu, lv = model(xb)
                loss = vae_loss(xr, xb, mu, lv)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            ep_loss.append(loss.item())
        sched.step()

        # Validation loss on held-out set
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            xv = X_t[idx[n_tr:]].to(device)
            xvr, muv, lvv = model(xv)
            val_loss = vae_loss(xvr, xv, muv, lvv).item()

        # Save latest every epoch (Colab recovery)
        torch.save(model.state_dict(), config.SUBTYPE_GAN_LATEST)
        if val_loss < best_val:
            best_val, best_ep = val_loss, epoch
            torch.save(model.state_dict(), config.SUBTYPE_GAN_MODEL)

        losses.append(val_loss)
        if epoch % 20 == 0:
            vram = torch.cuda.memory_allocated()/1e9
            print(f"  ep {epoch:3d} | train={np.mean(ep_loss):.4f} | "
                  f"val={val_loss:.4f} | best_ep={best_ep} | VRAM={vram:.2f}GB")

    print(f"Training done. Best val_loss={best_val:.4f} at epoch {best_ep}")

    # ── Extract embeddings + GMM ────────────────────────────────────────────
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, mu_all, _ = model(X_t.to(device))
    emb = mu_all.cpu().numpy()

    # reorder to original sample order
    emb_ord = np.zeros_like(emb)
    emb_ord[idx.numpy()] = emb

    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full',
                           random_state=RANDOM_STATE, max_iter=200, n_init=5)
    vae_labels = gmm.fit_predict(emb_ord)
    print(f"GMM cluster sizes: {np.bincount(vae_labels).tolist()}")

    with open(config.SUBTYPE_GMM_MODEL, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"GMM saved → {config.SUBTYPE_GMM_MODEL}")

    # Plot loss curve
    plt.figure(figsize=(8, 3))
    plt.plot(losses, color='#4361ee')
    plt.axvline(best_ep, color='#e63946', ls='--', label=f'best={best_ep}')
    plt.xlabel('Epoch'); plt.ylabel('Val Loss')
    plt.title('VAE Training Loss')
    plt.legend(); plt.tight_layout(); plt.show()

else:
    print(f"Skipping VAE (COMPUTE_TIER='{COMPUTE_TIER}' or no CUDA)")
    print("Using KMeans baseline labels instead.")
"""),

    cell_md("## 5. Cluster Analysis & Visualization"),

    cell_code("""# Use best available labels
labels = vae_labels if vae_labels is not None else kmeans_labels

# ── 2D UMAP visualization ──────────────────────────────────────────────
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    emb_2d = reducer.fit_transform(X_pca)   # use PCA embedding as UMAP input (faster)
    use_umap = True
except ImportError:
    # Fallback: show first 2 PCA components
    emb_2d = X_pca[:, :2]
    use_umap = False

palette = ['#e63946', '#4361ee', '#52b788', '#f4a261', '#7209b7']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cluster assignments
scatter = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                           c=labels, cmap='Set2', s=30, alpha=0.8, edgecolors='white', lw=0.3)
axes[0].set_title(f"{'UMAP' if use_umap else 'PCA'} — Cluster Assignments", fontweight='bold')
axes[0].set_xlabel('Dim 1'); axes[0].set_ylabel('Dim 2')
plt.colorbar(scatter, ax=axes[0])

# Mortality overlay
mortality = clin_f['Disease_Status'].values if 'Disease_Status' in clin_f.columns else np.zeros(len(emb_2d))
axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                c=mortality, cmap='RdYlGn_r', s=30, alpha=0.8, edgecolors='white', lw=0.3)
axes[1].set_title(f"{'UMAP' if use_umap else 'PCA'} — Mortality Overlay", fontweight='bold')

plt.tight_layout()
plt.savefig('../docs/phase1_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

    cell_code("""# ── Per-cluster clinical summary ──────────────────────────────────────
clin_f_labeled = clin_f.copy()
clin_f_labeled['Cluster'] = labels

summary_cols = ['Disease_Status']
if 'age_at_diagnosis' in clin_f_labeled.columns:
    summary_cols.append('age_at_diagnosis')
stage_col = next((c for c in ['tumor_stage','ajcc_pathologic_stage']
                  if c in clin_f_labeled.columns), None)
if stage_col:
    clin_f_labeled['Stage_IV'] = clin_f_labeled[stage_col].str.contains('IV|4',
                                                                          case=False, na=False)
    summary_cols.append('Stage_IV')

summary = clin_f_labeled.groupby('Cluster')[summary_cols].agg(['mean', 'count'])
print("Cluster Clinical Summary:")
display(summary.round(3))
"""),

    cell_md("## 6. Survival Analysis — Validation Anchor"),

    cell_code("""from scipy import stats as sp_stats

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    # Resolve time column
    time_col = next(
        (c for c in ['days_to_death', 'overall_survival_days', 'days_to_last_follow_up']
         if c in clin_f_labeled.columns), None
    )

    if time_col:
        clin_f_labeled['_time']  = pd.to_numeric(clin_f_labeled[time_col], errors='coerce').fillna(0)
        clin_f_labeled['_event'] = clin_f_labeled['Disease_Status'].fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e63946', '#4361ee', '#52b788', '#f4a261']
        kmf = KaplanMeierFitter()
        min_p = 1.0

        for cluster_id in sorted(clin_f_labeled['Cluster'].unique()):
            mask = clin_f_labeled['Cluster'] == cluster_id
            n = mask.sum()
            kmf.fit(
                clin_f_labeled.loc[mask, '_time'],
                event_observed=clin_f_labeled.loc[mask, '_event'],
                label=f'Cluster {cluster_id} (n={n})'
            )
            kmf.plot_survival_function(ax=ax, color=colors[cluster_id % len(colors)], ci_show=True)

        # Pairwise log-rank
        results = {}
        for i in range(N_CLUSTERS):
            for j in range(i+1, N_CLUSTERS):
                mi = clin_f_labeled['Cluster'] == i
                mj = clin_f_labeled['Cluster'] == j
                if mi.sum() < 5 or mj.sum() < 5:
                    continue
                lr = logrank_test(
                    clin_f_labeled.loc[mi, '_time'], clin_f_labeled.loc[mj, '_time'],
                    event_observed_A=clin_f_labeled.loc[mi, '_event'],
                    event_observed_B=clin_f_labeled.loc[mj, '_event'],
                )
                results[f'C{i}_vs_C{j}'] = lr.p_value
                min_p = min(min_p, lr.p_value)

        ax.set_title('Kaplan-Meier Survival by Cluster — LIHC', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days Since Diagnosis')
        ax.set_ylabel('Survival Probability')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('../docs/phase1_survival.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\\nPairwise Log-rank Test p-values:")
        for k, p in sorted(results.items()):
            flag = '✅ SIGNIFICANT' if p < 0.05 else '⚠️  NOT sig'
            print(f"  {k}: p={p:.4f}  {flag}")

        # ── VALIDATION ANCHOR CHECK ──────────────────────────────────────
        print(f"\\nBest log-rank p: {min_p:.4f}")
        if min_p < 0.05:
            print("✅ SURVIVAL VALIDATION PASSED — subtypes are clinically distinct")
        else:
            print("⚠️  SURVIVAL VALIDATION: p >= 0.05 — subtypes may not be clinically distinct")
            print("   Try: more clusters, include CNV/mutation features, longer survival follow-up")
    else:
        print("No survival time column found in clinical data")
except ImportError:
    print("lifelines not installed. Run: pip install lifelines")
"""),

    cell_md("## 7. Save Results"),

    cell_code("""# ── Save patient clusters ──────────────────────────────────────────────
config.PATIENT_CLUSTERS_CSV.parent.mkdir(parents=True, exist_ok=True)
output = clin_f_labeled.copy()
output.to_csv(config.PATIENT_CLUSTERS_CSV, index=False)
print(f"✅ Clusters saved → {config.PATIENT_CLUSTERS_CSV}")
print(f"   Shape: {output.shape}")
print(f"   Cluster distribution:\\n{output['Cluster'].value_counts().sort_index().to_string()}")
"""),

    cell_code("""# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 1 COMPLETE — NEXT STEPS")
print("=" * 55)
print(f"  Output: {config.PATIENT_CLUSTERS_CSV}")
print(f"  Model tier used: {'VAE+GMM' if vae_labels is not None else 'KMeans'}")
print()
print("Next:")
print("  1. Review survival plot — are subtypes clinically distinct?")
print("  2. Open Phase_02_Biomarker_Discovery.ipynb")
print("  3. Pass patient clusters as subtype labels for differential expression")
if vae_labels is None:
    print()
    print("Upgrade to VAE+GMM:")
    print("  Set COMPUTE_TIER = 'single_gpu' and re-run (requires GPU)")
"""),
])
save_nb(p1, "Phase_01_Disease_Subtyping.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Biomarker Discovery
# ─────────────────────────────────────────────────────────────────────────────
def stub_nb(phase_num, title, goal, tier_table, anchor, inputs, outputs, next_nb):
    return make_notebook([
        cell_md(f"""# Phase {phase_num:02d} — {title}
**Goal:** {goal}

{tier_table}

**Validation Anchor:** {anchor}
"""),
        cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import config, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')

print(f"CUDA: {torch.cuda.is_available()}")
print(f"COMPUTE_TIER: {config.COMPUTE_TIER}")
"""),
        cell_md(f"## Inputs\n{inputs}"),
        cell_code(f"# TODO: implement Phase {phase_num} logic here\npass"),
        cell_md(f"## Outputs\n{outputs}"),
        cell_md(f"## Next\nOpen `{next_nb}`"),
    ])

p2 = stub_nb(
    2, "Biomarker Discovery",
    "Identify top differentially expressed genes and mutations that distinguish LIHC subtypes.",
    """| Tier | Model | Details |
|---|---|---|
| MVP | Random Forest + SHAP | Top 50 genes by SHAP importance |
| Tier 2 | Multi-modal AE + SHAP | Expression + mutation autoencoder (LMF fusion) |""",
    "Top-10 biomarker list must contain ≥3 known LIHC oncogenes (TP53, CTNNB1, AFP)",
    "- `data/processed/patient_clusters.csv` (Phase 1 output)\n- `data/processed/tcga_lihc_rnaseq.csv`\n- `data/processed/tcga_lihc_mutations.maf`",
    "- `data/processed/biomarkers.csv` — ranked gene list with SHAP scores",
    "Phase_03_Drug_Response_Prediction.ipynb"
)
save_nb(p2, "Phase_02_Biomarker_Discovery.ipynb")

p3 = stub_nb(
    3, "Drug Response Prediction",
    "Predict IC50 drug sensitivity for LIHC cell lines using GDSC2 + CCLE + LINCS data.",
    """| Tier | Model | Details |
|---|---|---|
| MVP | Simple MLP | CCLE expression + GDSC2 IC50 |
| Tier 2 | MMDRP-lite | 3-branch AE (expr+mut+cnv) + LMF fusion |
| Paper | Full MMDRP + LINCS gctx | Run `colab/Colab_Phase_03_Full_MMDRP.ipynb` |""",
    "Sorafenib predicted IC50 must be in bottom 25th percentile for HCC cell lines",
    "- `data/external/gdsc2_drug_sensitivity.csv`\n- `data/external/ccle_expression.csv`\n- `LINCS/consensi TSVs`",
    "- `data/processed/drug_response_predictions.csv`",
    "Phase_04_Drug_Repurposing.ipynb"
)
save_nb(p3, "Phase_03_Drug_Response_Prediction.ipynb")

p4 = stub_nb(
    4, "Drug Repurposing",
    "Identify candidate drugs for LIHC repurposing using Knowledge Graph embeddings + BioGPT path reasoning.",
    """| Tier | Model | Details |
|---|---|---|
| MVP | PyKEEN TransE (64-dim) | Filtered DRKG ~500K triples |
| Tier 2 | PyKEEN RotatE + BioGPT base | Hybrid KG+LLM scoring |
| Paper | RotatE + BioGPT-Large | Run on Kaggle P100 |""",
    "Sorafenib AND Lenvatinib must appear in top-50 repurposing candidates (hard gate)",
    "- `data/external/drkg/drkg_lihc_filtered.tsv` (run `scripts/filter_drkg.py` first)\n- `data/external/gdsc2_drug_sensitivity.csv`",
    "- `data/processed/repurposing_candidates.csv` — top-N drugs with scores + LLM explanations",
    "Phase_05_Molecular_Simulation.ipynb"
)
save_nb(p4, "Phase_04_Drug_Repurposing.ipynb")

p5 = stub_nb(
    5, "Molecular Simulation",
    "Predict binding affinity of candidate drugs against LIHC target proteins. Filter by ADMET safety.",
    """| Tier | Model | Details |
|---|---|---|
| Local | AutoDock Vina | Requires MGLTools + `.pdbqt` structure prep |
| Paper | AlphaFold3 co-folding | Run `colab/Colab_Phase_05_AF3.ipynb` on Kaggle |""",
    "≥5 of top-10 candidates pass Lipinski Rule of 5 + ADMET filters",
    "- `data/targets/*.pdb` (7 LIHC targets — auto-downloaded)\n- `data/processed/repurposing_candidates.csv` SMILES column",
    "- `data/processed/interaction_scores.csv` — binding scores + ADMET flags per drug-target pair",
    "Phase_06_Personalized_Medicine.ipynb"
)
save_nb(p5, "Phase_05_Molecular_Simulation.ipynb")

p6 = stub_nb(
    6, "Personalized Medicine / Digital Twin",
    "Generate personalized drug recommendations for a patient profile integrating all pipeline outputs.",
    """| Tier | Model | Details |
|---|---|---|
| MVP | Rule-based matching | Subtype → known drug mapping |
| Tier 2 | Digital Twin scoring | Weighted combination of Phases 1-5 scores |""",
    "Known approved LIHC drug must be top recommendation for a stage III/IV patient profile",
    "- All outputs from Phases 1-5\n- Patient profile (subtype, mutation profile, stage)",
    "- `data/processed/recommendations.json` — explainable JSON recommendation",
    "See `README.md` for next steps"
)
save_nb(p6, "Phase_06_Personalized_Medicine.ipynb")

print("\nAll notebooks created:")
for f in sorted(NB_DIR.glob("Phase_*.ipynb")):
    print(f"  {f.name}")
