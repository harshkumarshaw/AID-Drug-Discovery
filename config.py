"""
config.py — Central configuration for the AI Drug Discovery Pipeline.
Updated: April 2026 — reflects 2026 SOTA tools (AlphaFold3, DRKG, BioGPT)

All paths, hyperparameters, and constants are defined here.
Import this module in every pipeline file instead of hardcoding values.
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────
# COMPUTE TIER — SET THIS FIRST
# ─────────────────────────────────────────────
# Options:
#   "cpu"        — No GPU. Lightweight models, small datasets, ~16GB RAM
#                  E.g.: laptop, no CUDA. Disables AF3, BioGPT, full gctx.
#   "single_gpu" — 1 GPU with 6–12 GB VRAM (e.g., RTX 3060/4060, T4 on Colab)
#                  Enables DiffDock (AF3 fallback), MMDRP, PyKEEN on GPU.
#   "high_gpu"   — 1+ GPU with ≥24 GB VRAM (e.g., A100, V100, RTX 4090)
#                  Enables full AlphaFold3 co-folding, BioGPT-Large, full gctx.
COMPUTE_TIER: str = "single_gpu"  # RTX 3050 4GB VRAM

# Easy override: auto-detect GPU if available
_has_gpu = torch.cuda.is_available()
_gpu_vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if _has_gpu else 0)
# Auto-detect only overrides if user left default "cpu" but has a GPU
if COMPUTE_TIER == "cpu" and _has_gpu:
    COMPUTE_TIER = "high_gpu" if _gpu_vram_gb >= 20 else "single_gpu"

# ─────────────────────────────────────────────
# COLAB OFFLOAD PLAN — RTX 3050 4GB setup
# ─────────────────────────────────────────────
# Phases listed here should be run in notebooks/colab/ on Google Colab (T4 15GB)
# or Kaggle (P100 16GB) — they exceed 4GB VRAM budget.
# Everything NOT listed here runs locally on the RTX 3050.
COLAB_OFFLOAD = {
    # Phase 1 Tier-2: Full Subtype-GAN (>4GB at full dims) → Colab T4
    "subtype_gan_full":    True,
    # Phase 3 Tier-2: Full MMDRP with all 3 omic AEs + 42GB gctx → Colab T4
    "mmdrp_full":          True,
    # Phase 3 gctx: 42GB LINCS file — mount on Google Drive in Colab
    "gctx_loading":        True,
    # Phase 4 Tier-2: BioGPT path reasoning (BioGPT-Large ~6GB float16) → Colab
    "biogpt_large":        True,
    # Phase 5 Tier-2: AlphaFold3 (needs ≥10GB VRAM) → Colab A100 or T4
    "alphafold3":          True,
    # Phase 7 stretch: DiffMol generative model → Colab
    "diffmol":             True,

    # These run locally on RTX 3050 4GB:
    # Phase 1 Tier-1/2: VAE+GMM, small Subtype-GAN (reduced dims)
    # Phase 2: Autoencoder + SHAP (fits in 4GB with batch=32)
    # Phase 3 Tier-1: Simple MLP + MMDRP-lite (top 2000 genes, batch=32)
    # Phase 4 Tier-1/2: PyKEEN TransE (64-dim, fits easily)
    # Phase 4 LLM: BioGPT base (NOT Large) for embedding extraction
    # Phase 5 Tier-1: AutoDock Vina (CPU-based, no GPU needed)
    # Phase 6: Patient recommendation (CPU)
}

# ─── Mixed precision (AMP) — crucial for 4GB VRAM ──────────────────────────
USE_AMP = True        # torch.cuda.amp.autocast — halves VRAM for all phases
AMP_DTYPE = "float16" # float16 for RTX 3050 (bfloat16 only on Ampere A100+)

# ─── Gradient checkpointing — trades compute for memory ────────────────────
USE_GRAD_CHECKPOINT = True  # Enable in autoencoder/GAN training to save VRAM

# MVP mode: run simplified end-to-end first before expanding any phase
# Set to False only after ALL 6 phases run successfully end-to-end.
MVP_MODE: bool = True

# ─── Per-phase model selection based on COMPUTE_TIER ───────────────────────
# Each phase reads this to choose the right model complexity.
PHASE_MODELS = {
    "subtyping": {
        "cpu":        "kmeans_pca",          # Fast, interpretable baseline
        "single_gpu": "vae_gmm",             # VAE + GMM (lighter than Subtype-GAN)
        "high_gpu":   "subtype_gan",         # Full GAN adversarial (SOTA)
    },
    "biomarker": {
        "cpu":        "random_forest",       # RF + SelectFromModel
        "single_gpu": "autoencoder_shap",    # Single AE + SHAP (no LMF)
        "high_gpu":   "lmf_shap",            # Full LMF fusion + SHAP
    },
    "drug_response": {
        "cpu":        "mlp_gdsc",            # Simple MLP on GDSC2 + fingerprints
        "single_gpu": "mmdrp_lite",          # MMDRP without CNV AE branch
        "high_gpu":   "mmdrp_full",          # Full MMDRP (all 3 omic AEs)
    },
    "repurposing": {
        "cpu":        "fingerprint_cosine",  # Morgan + cosine similarity baseline
        "single_gpu": "pykeen_transe",       # KG embedding only (no LLM)
        "high_gpu":   "pykeen_biogpt",       # Full KG + BioGPT path reasoning
    },
    "molecular_sim": {
        "cpu":        "rdkit_lipinski",      # Lipinski filter only (no docking)
        "single_gpu": "vina_docking",        # AutoDock Vina (lightweight)
        "high_gpu":   "alphafold3",          # Full AF3 protein-ligand co-folding
    },
    "personalized": {
        "cpu":        "knn_cluster",         # Simple KNN on cluster + drug scores
        "single_gpu": "ranked_simulation",   # Multi-filter ranking
        "high_gpu":   "digital_twin",        # Full Digital Twin simulation + LLM
    },
}

# Convenience accessor
def get_model(phase: str) -> str:
    """Return the appropriate model name for the current COMPUTE_TIER."""
    return PHASE_MODELS[phase][COMPUTE_TIER]

# MVP dataset limits (only active when MVP_MODE=True)
MVP_MAX_GENES = 2000          # Use top 2000 variable genes instead of all ~60K
MVP_MAX_CELL_LINES = 200      # Use 200 cell lines instead of ~1800 CCLE lines
MVP_MAX_DRUGS = 100           # Use 100 drugs from GDSC2 instead of ~300
MVP_MAX_KG_TRIPLES = 100_000  # Use 100K DRKG triples instead of 5.87M
MVP_MAX_EPOCHS = 20           # Quick training to validate pipeline runs

# ─────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
# ─────────────────────────────────────────────
# GAN TRAINING GUARD (Gap #1 Fix) — NaN Loss Detection
# ─────────────────────────────────────────────
# GAN training is unstable with AMP (float16 underflows in discriminator).
# These guards MUST be applied in subtype_gan training loops.
#
# GAN NaN guard pattern:
#   scaler.unscale_(optimizer)          # unscale before clip
#   torch.nn.utils.clip_grad_norm_(model.parameters(), GAN_GRAD_CLIP)
#   if any(p.grad is not None and torch.isnan(p.grad).any()
#           for p in model.parameters()):
#       optimizer.zero_grad()            # skip this step entirely
#       print(f"[GAN] NaN gradient at epoch {epoch} — skipping step")
#       continue
#   scaler.step(optimizer)
#   scaler.update()
#
GAN_GRAD_CLIP: float = 1.0           # Gradient norm clip for GAN stability
GAN_LABEL_SMOOTH: float = 0.1        # Label smoothing: real=0.9, fake=0.1
GAN_NAN_SKIP: bool = True            # Skip steps with NaN gradients
GAN_MIN_DISC_LOSS: float = 0.01      # Divergence guard: if disc_loss<0.01,
                                     # generator has collapsed — restart
# If disc_loss < GAN_MIN_DISC_LOSS for 10 consecutive epochs:
#   raise ValueError("GAN mode collapse detected. Reduce LR or add noise.")
GAN_COLLAPSE_PATIENCE: int = 10

# ─────────────────────────────────────────────
# GOOGLE DRIVE FOLDER STRUCTURE (for Colab/Kaggle sessions)
# ─────────────────────────────────────────────
# Standard layout on Google Drive (set DRIVE_ROOT in Colab cells):
# MyDrive/AID_Research/
#   checkpoints/              ← model .pt files saved from Colab
#     phase1_subtype/
#     phase3_drp/
#     phase4_repurposing/
#     phase5_af3/
#   data/
#     gdsc2_drug_sensitivity.csv
#     ccle_expression.csv
#     drkg.tsv
#     modzs.gctx            ← 42GB LINCS (upload once)
#   outputs/
#     patient_clusters.csv
#     repurposing_candidates.csv
#     interaction_scores_af3.csv
#     recommendations.json
#
DRIVE_ROOT_COLAB: str = "/content/drive/MyDrive/AID_Research"
DRIVE_CHECKPOINTS: str = f"{DRIVE_ROOT_COLAB}/checkpoints"
DRIVE_DATA: str = f"{DRIVE_ROOT_COLAB}/data"
DRIVE_OUTPUTS: str = f"{DRIVE_ROOT_COLAB}/outputs"
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "plots"

# ─────────────────────────────────────────────
# DATASET DIRECTORIES
# ─────────────────────────────────────────────
TCGA_DIR = ROOT_DIR / "TCGA"
EMTAB_DIR = ROOT_DIR / "E-MTAB-5902"
LINCS_DIR = ROOT_DIR / "LINCS"
DRUG_REPURPOSING_DIR = ROOT_DIR / "Drug Repurposing"

# ─────────────────────────────────────────────
# TCGA FILES
# ─────────────────────────────────────────────
TCGA_CLINICAL_CSV = PROCESSED_DIR / "cleaned_clinical_data.csv"
TCGA_BIOSPECIMEN_CSV = PROCESSED_DIR / "cleaned_biospecimen_data.csv"
TCGA_RNASEQ_CSV = PROCESSED_DIR / "tcga_lihc_rnaseq.csv"       # To be downloaded
TCGA_CNV_CSV = PROCESSED_DIR / "tcga_lihc_cnv.csv"             # To be downloaded
TCGA_MUTATION_MAF = PROCESSED_DIR / "tcga_lihc_mutations.maf"  # To be downloaded
TCGA_METHYL_CSV = PROCESSED_DIR / "tcga_lihc_methylation.csv"  # Optional; ~2 GB

# ─────────────────────────────────────────────
# E-MTAB-5902 FILES
# ─────────────────────────────────────────────
EMTAB_VARIANT_TSV = EMTAB_DIR / "var-GS000001153-ASM.tsv"
EMTAB_METADATA_TSV = EMTAB_DIR / "E-MTAB-5902.sdrf.txt"

# ─────────────────────────────────────────────
# LINCS FILES
# ─────────────────────────────────────────────
LINCS_GCTX = LINCS_DIR / "modzs.gctx"                          # 42 GB — load via cmapPy
LINCS_DRUGBANK_TSV = LINCS_DIR / "consensi-drugbank.tsv"        # Use individually (NOT concat all)
LINCS_KNOCKDOWN_TSV = LINCS_DIR / "consensi-knockdown.tsv"
LINCS_OVEREXPRESSION_TSV = LINCS_DIR / "consensi-overexpression.tsv"
LINCS_PERT_TSV = LINCS_DIR / "consensi-pert_id.tsv"

# ─────────────────────────────────────────────
# ChEMBL-35 FILES
# ─────────────────────────────────────────────
CHEMBL_SDF = DRUG_REPURPOSING_DIR / "chembl_35.sdf"
CHEMBL_SQLITE_ARCHIVE = DRUG_REPURPOSING_DIR / "chembl_35_sqlite.tar.gz"
CHEMBL_H5 = DRUG_REPURPOSING_DIR / "chembl_35.h5"
CHEMBL_DB_DIR = DRUG_REPURPOSING_DIR / "chembl_db"
CHEMBL_DB_PATH = CHEMBL_DB_DIR / "chembl_35.db"                 # After extraction
DEEPDRA_ZIP = DRUG_REPURPOSING_DIR / "DeepDRA-main.zip"
DEEPDRA_DIR = DRUG_REPURPOSING_DIR / "DeepDRA-main"

# ─────────────────────────────────────────────
# EXTERNAL DATA (GDSC2, CCLE, DRKG, validation DBs) — 2026
# ─────────────────────────────────────────────
GDSC2_CSV = EXTERNAL_DIR / "gdsc2_drug_sensitivity.csv"
CCLE_EXPRESSION_CSV = EXTERNAL_DIR / "ccle_expression.csv"
CCLE_MUTATION_CSV = EXTERNAL_DIR / "ccle_mutations.csv"
CCLE_CNV_CSV = EXTERNAL_DIR / "ccle_cnv.csv"
CCLE_MODEL_INFO_CSV = EXTERNAL_DIR / "ccle_model_info.csv"      # name harmonization
PRISM_CSV = EXTERNAL_DIR / "prism_drug_sensitivity.csv"         # Optional extra IC50

# Drug Repurposing Knowledge Graph
DRKG_DIR = EXTERNAL_DIR / "drkg"
DRKG_TRIPLES_TSV = DRKG_DIR / "drkg.tsv"

# STRING Protein-Protein Interactions (for KG augmentation)
STRING_PPI_TSV = EXTERNAL_DIR / "string_ppi.txt"

# DisGeNET gene-disease associations (for KG augmentation)
DISGENET_TSV = EXTERNAL_DIR / "disgenet_gene_disease.tsv"

# MSigDB KEGG gene sets (for biomarker enrichment)
MSIGDB_KEGG_GMT = EXTERNAL_DIR / "msigdb_kegg.gmt"

# Human Protein Atlas (liver cancer expression — for biomarker validation)
HPA_PATHOLOGY_TSV = EXTERNAL_DIR / "hpa_pathology.tsv"

# COSMIC Cancer Gene Census (for biomarker validation)
COSMIC_CENSUS_CSV = EXTERNAL_DIR / "cosmic_census.csv"

# BindingDB (for Phase 5 Vina prediction validation)
BINDINGDB_TSV = EXTERNAL_DIR / "bindingdb_lihc_targets.tsv"

# AlphaFold3 (open-source, January 2026)
AF3_MODEL_DIR = EXTERNAL_DIR / "alphafold3_weights"

# Drug target PDB structures (7 LIHC targets)
TARGETS_DIR = ROOT_DIR / "data" / "targets"
LIHC_TARGET_PDBS = {
    "VEGFR2":    TARGETS_DIR / "4ASD.pdb",   # Primary sorafenib/lenvatinib target
    "VEGFR2_alt": TARGETS_DIR / "3VHE.pdb",  # Better resolved structure
    "EGFR":     TARGETS_DIR / "1IVO.pdb",   # Overexpressed in 60% LIHC
    "BRAF":     TARGETS_DIR / "3PPJ.pdb",   # MAPK pathway — sorafenib target
    "mTOR":     TARGETS_DIR / "4JT6.pdb",   # PI3K/mTOR pathway
    "TP53":     TARGETS_DIR / "2OCJ.pdb",   # Mutated in 25% LIHC
    "FGFR4":    TARGETS_DIR / "5NUD.pdb",   # Amplified ~30% LIHC
}

# ─────────────────────────────────────────────
# PROCESSED OUTPUTS
# ─────────────────────────────────────────────
PATIENT_CLUSTERS_CSV = PROCESSED_DIR / "patient_clusters.csv"
BIOMARKERS_CSV = PROCESSED_DIR / "biomarkers.csv"
DRUG_RESPONSE_CSV = PROCESSED_DIR / "drug_response_predictions.csv"
REPURPOSING_CSV = PROCESSED_DIR / "repurposing_candidates.csv"
INTERACTION_SCORES_CSV = PROCESSED_DIR / "interaction_scores.csv"
RECOMMENDATIONS_JSON = PROCESSED_DIR / "recommendations.json"

# ─────────────────────────────────────────────
# MODEL OUTPUTS — Best + Latest checkpoints per phase
# latest.pt: overwritten every epoch (Colab session recovery)
# best.pt:   only overwritten when val_loss improves
# ─────────────────────────────────────────────
_SUBTYPE_DIR  = MODELS_DIR / "disease_subtypes"
_DRP_DIR      = MODELS_DIR / "drug_response"
_REPURP_DIR   = MODELS_DIR / "repurposing"
_MOLSIM_DIR   = MODELS_DIR / "molecular_sim"

SUBTYPE_GAN_MODEL   = _SUBTYPE_DIR / "subtype_gan_best.pt"
SUBTYPE_GAN_LATEST  = _SUBTYPE_DIR / "subtype_gan_latest.pt"   # overwrite every epoch
SUBTYPE_GMM_MODEL   = _SUBTYPE_DIR / "gmm_clustering.pkl"
BIOMARKER_AE_MODEL  = _SUBTYPE_DIR / "biomarker_ae_best.pt"
BIOMARKER_AE_LATEST = _SUBTYPE_DIR / "biomarker_ae_latest.pt"

DRUG_RESPONSE_MODEL   = _DRP_DIR / "mmdrp_best.pt"
DRUG_RESPONSE_LATEST  = _DRP_DIR / "mmdrp_latest.pt"

KG_EMBEDDING_MODEL   = _REPURP_DIR / "pykeen_transe_best.pt"
KG_EMBEDDING_LATEST  = _REPURP_DIR / "pykeen_transe_latest.pt"
DEEPDRA_MODEL        = _REPURP_DIR / "deepdra_finetuned.pt"

# Canonical checkpoint-saving helper (use in all training loops):
# def save_checkpoint(model, path_best, path_latest, val_loss, best_val_loss):
#     torch.save(model.state_dict(), path_latest)       # always — for Colab recovery
#     if val_loss < best_val_loss:
#         torch.save(model.state_dict(), path_best)     # only on improvement
#     return min(val_loss, best_val_loss)

# ─────────────────────────────────────────────
# PHASE 1 HYPERPARAMS — DISEASE SUBTYPING
# Tuned for RTX 3050 4GB VRAM + AMP
# ─────────────────────────────────────────────
RANDOM_STATE = 42
N_CLUSTERS = 4                                # GMM number of components

# Local (RTX 3050): VAE+GMM with reduced dims — fits in 4GB with AMP
SUBTYPE_VAE_LATENT_DIM = 32                  # Was 64 — halved for 4GB
SUBTYPE_VAE_HIDDEN_DIMS = [256, 128]         # Was [512,256] — halved for 4GB
SUBTYPE_VAE_EPOCHS = 100                     # Was 200 — AMP makes this faster
SUBTYPE_VAE_LR = 1e-4
SUBTYPE_VAE_BATCH = 32                       # Was 64 — smaller batch for 4GB

# Colab (T4 15GB): Full Subtype-GAN — run via colab/colab_phase1_subtype_gan.py
SUBTYPE_GAN_LATENT_DIM = 64
SUBTYPE_GAN_HIDDEN_DIMS = [512, 256]
SUBTYPE_GAN_DISC_HIDDEN = [128, 64]
SUBTYPE_GAN_EPOCHS = 200
SUBTYPE_GAN_LR = 1e-4
SUBTYPE_GAN_BATCH = 64

# ─────────────────────────────────────────────
# PHASE 2 HYPERPARAMS — BIOMARKER DISCOVERY
# Tuned for RTX 3050 4GB VRAM + AMP
# ─────────────────────────────────────────────
EXPR_AE_DIMS = [1024, 256, 32]               # Was [2048,512,64] — halved for 4GB
VARIANT_AE_DIMS = [256, 64, 16]             # Was [500,128,32]
LMF_RANK = 8                                 # Was 16 — lower rank for local
RF_N_ESTIMATORS = 100                        # Was 200 — faster for MVP
CHUNK_SIZE = 100_000                         # For reading large variant files
TOP_N_BIOMARKERS = 50

# ─────────────────────────────────────────────
# PHASE 3 HYPERPARAMS — DRUG RESPONSE
# Local: MMDRP-lite (2 omic AEs, top 2000 genes, batch=32)
# Colab: Full MMDRP (3 omic AEs + gctx, batch=128)
# ─────────────────────────────────────────────
# Local dims (RTX 3050 4GB + AMP)
CELL_EXPR_AE_DIM = 128                       # Was 256 — halved for 4GB
CELL_MUT_AE_DIM = 64                         # Was 128
CELL_CNV_AE_DIM = 32                         # Was 64 (skip in lite mode)
DRUG_AE_DIM = 64                             # Was 128
DRUG_LINCS_DIM = 32                          # Was 64
DRP_LMF_RANK = 16                            # Was 32
DRP_MLP_HIDDEN = [128, 64, 32]              # Was [256,128,64]
DRP_DROPOUT = 0.3
DRP_LR = 1e-4
DRP_EPOCHS = 50                              # Was 100; use early stopping
DRP_BATCH = 32                               # Was 128; critical for 4GB VRAM
DRP_PATIENCE = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─────────────────────────────────────────────
# PHASE 4 HYPERPARAMS — DRUG REPURPOSING
# Local: PyKEEN TransE (64-dim) + BioGPT base embeddings
# Colab/Kaggle (paper results): PyKEEN RotatE (128-dim) + BioGPT-Large
# ─────────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
KG_MODEL = "TransE"                          # Local: TransE (faster, less memory)
KG_EMBEDDING_DIM = 64                        # Was 128; 64 fits easily in 4GB
KG_EPOCHS = 50                               # Was 100; use early stopping
KG_LR = 0.01

# ─── DRKG Filtering (Gap #3 Fix) ──────────────────────────────────
# Full DRKG has 5.87M triples across many unrelated entity types.
# Filter to ONLY drug↔gene, gene↔disease, drug↔disease triples for LIHC.
# This reduces 5.87M → ~500K triples — PyKEEN-manageable even locally.
#
# DRKG relation prefixes to KEEP (all others discarded):
DRKG_KEEP_PREFIXES: list = [
    "GNBR::",          # Gene-disease from text mining (most relevant)
    "DRUGBANK::",      # Drug-target from DrugBank (high quality)
    "DGIDB::",         # Drug-gene interaction DB
    "DISEASES::",      # Gene-disease from text + knowledge channels
    "STRING::",        # PPI edges (protein-protein, keep high conf only)
    "HETIONET::",      # Multi-type bio network
]

# Filter STRING PPI to only high-confidence interactions:
STRING_MIN_SCORE: int = 700          # STRING combined score ≥ 700 (out of 1000)

# DRKG entity prefixes to KEEP (filter to biology relevant to LIHC):
# Entities named "Compound::", "Gene::", "Disease::" are kept.
# Entities named "Anatomy::", "Symptom::", "Pathway::" are also kept.
# Side effects ("SideEffect::") and demographics ("Species::") are dropped.
DRKG_KEEP_ENTITY_TYPES: list = [
    "Compound", "Gene", "Disease", "Pathway", "Biological Process",
]

# Cap for local PyKEEN (RTX 3050 4GB): after filtering apply this limit
DRKG_MAX_TRIPLES_LOCAL: int = 500_000   # Full filtered set is ~1-2M; cap here

# On Kaggle/Colab for paper runs: use the full filtered set (no cap needed)
DRKG_MAX_TRIPLES_PAPER: int = 0          # 0 = no cap

# DRKG filter function (import and call in pipeline/drug_repurposing.py):
# See scripts/filter_drkg.py for implementation

# LLM for path reasoning:
# LOCAL (RTX 3050 4GB): use BioGPT base — paper note below
# PAPER RESULTS (Kaggle P100): MUST use BioGPT-Large — see note
LLM_MODEL_NAME = "microsoft/BioGPT"         # Base model, local dev
LLM_MODEL_NAME_COLAB = "microsoft/BioGPT-Large"  # Kaggle P100 for paper
LLM_FALLBACK = "pritamdeka/S-PubMedBert-MS-MARCO"

# PAPER NOTE (Gap #2): For publication, ALL repurposing results MUST be produced
# with BioGPT-Large on Kaggle. Local BioGPT base results are development-only.
# Similarly, all Phase 5 docking scores must use AF3 (not Vina) for the paper.
# Mark any result produced with base model or Vina as "dev baseline" in your tables.

# Score weights — PLACEHOLDER until ablation justifies these
KG_SCORE_WEIGHT = 0.40
LLM_PATH_SCORE_WEIGHT = 0.30
DEEPDRA_SCORE_WEIGHT = 0.30
TOP_N_CANDIDATES = 50

# ─────────────────────────────────────────────
# PHASE 5 HYPERPARAMS — MOLECULAR SIMULATION
# LOCAL (RTX 3050 4GB): AutoDock Vina (CPU-based, no GPU needed)
# COLAB (T4/A100): AlphaFold3 protein-ligand co-folding
# ─────────────────────────────────────────────

# Local: AutoDock Vina settings
VINA_EXHAUSTIVENESS = 8                      # Lower = faster, less thorough (CPU)
VINA_NUM_MODES = 5                           # Top binding poses to return
VINA_ENERGY_RANGE = 3                        # kcal/mol range for poses

# Colab: AlphaFold3 settings
AF3_MODEL_PRESET = "model_1"
AF3_NUM_PREDICTIONS = 3
AF3_BINDING_THRESHOLD = 0.5                  # ipTM confidence threshold

# ADMET filters (applies to both local and Colab)
ADMET_PASS_THRESHOLDS = {
    "Lipinski": True,                        # Relaxed Ro5: MW≤550, LogP≤6
    "BBB": None,                             # Blood-brain barrier: not required
    "hERG": False,                           # hERG toxicity: must NOT flag
    "AMES": False,                           # AMES mutagenicity: must NOT flag
}

# ─────────────────────────────────────────────
# PHASE 6 HYPERPARAMS — DIGITAL TWIN / PERSONALIZED MEDICINE
# ─────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 10
SHAP_N_SAMPLES = 100                         # SHAP background samples
LLM_EXPLANATION_MODEL = "microsoft/BioGPT"  # For clinical narrative generation

# ─────────────────────────────────────────────
# API CONFIG
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True


def validate_paths():
    """Check that critical data directories and files exist. Create output dirs."""
    print("=" * 60)
    print("AI Drug Discovery Pipeline — Path Validation (April 2026)")
    print("=" * 60)

    # Check input directories
    input_dirs = [TCGA_DIR, EMTAB_DIR, LINCS_DIR, DRUG_REPURPOSING_DIR]
    for p in input_dirs:
        status = "✓ OK" if p.exists() else "✗ MISSING"
        print(f"  {status}  {p}")

    # Check critical input files
    critical_files = [
        LINCS_GCTX, CHEMBL_SQLITE_ARCHIVE, EMTAB_VARIANT_TSV,
        LINCS_DRUGBANK_TSV, DEEPDRA_ZIP
    ]
    for f in critical_files:
        if f.exists():
            size_gb = f.stat().st_size / 1e9
            print(f"  ✓ OK  {f.name}  ({size_gb:.1f} GB)")
        else:
            print(f"  ✗ MISSING  {f}")

    # Check external data (downloads required)
    ext_files = [GDSC2_CSV, CCLE_EXPRESSION_CSV, DRKG_TRIPLES_TSV]
    print("\n  External downloads (required before Phase 3/4):")
    for f in ext_files:
        status = "✓ Downloaded" if f.exists() else "⚠ Not yet downloaded"
        print(f"  {status}  {f.name}")

    # Create all output directories
    output_dirs = [
        PROCESSED_DIR, EXTERNAL_DIR, PLOTS_DIR, DRKG_DIR,
        MODELS_DIR / "disease_subtypes",
        MODELS_DIR / "drug_response",
        MODELS_DIR / "repurposing",
        CHEMBL_DB_DIR,
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("\n  ✓ All output directories created.")
    print("=" * 60)


if __name__ == "__main__":
    validate_paths()
