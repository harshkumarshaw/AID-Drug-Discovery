# Compute Strategy: RTX 3050 4GB + Colab/Kaggle
## Hardware-Aware Execution Plan — April 2026

### Hardware Summary
| Resource | Spec | Notes |
|---|---|---|
| Local GPU | RTX 3050 4GB VRAM | CUDA capable; AMP (float16) halves effective VRAM usage |
| Local RAM | 16 GB | Limits in-memory dataset size; 42 GB gctx impossible |
| Colab | Free T4 (15 GB VRAM, ~12 GB RAM) | 12-hour session limit; use for 1 heavy phase at a time |
| Kaggle | Free P100 (16 GB VRAM, 13 GB RAM) | 30 hr/week; persistent kernel; better for long runs |

---

## Phase-by-Phase Execution Map

| Phase | Task | Where | Model | Est. Time |
|---|---|---|---|---|
| 1 | Disease Subtyping (MVP) | **Local** | PCA + KMeans | 5 min |
| 1 | Disease Subtyping (Tier 2) | **Local** | VAE + GMM (dims halved) + AMP | 20–40 min |
| 1 | Disease Subtyping (Tier 2 full) | **Colab** | Full Subtype-GAN | 1–2 hr |
| 2 | Biomarker Discovery (MVP) | **Local** | RandomForest + SHAP | 10 min |
| 2 | Biomarker Discovery (Tier 2) | **Local** | Single AE + SHAP, batch=32, AMP | 30–60 min |
| 3 | Drug Response (MVP) | **Local** | Simple MLP + GDSC2 + consensi TSV | 15 min |
| 3 | Drug Response (Tier 2) | **Local** | MMDRP-lite (2 AEs, batch=32, AMP) | 1–2 hr |
| 3 | Drug Response (full gctx) | **Colab** | Full MMDRP + modzs.gctx | 3–4 hr |
| 4 | Drug Repurposing (MVP) | **Local** | Cosine similarity (Morgan FPs) | 5 min |
| 4 | Drug Repurposing (Tier 2) | **Local** | PyKEEN TransE (64-dim, 50K triples) | 30–60 min |
| 4 | BioGPT base embeddings | **Local** | BioGPT base, embed-only, batch=8 | 1–2 hr |
| 4 | BioGPT-Large path reasoning | **Colab** | BioGPT-Large, float16 | 2–4 hr |
| 5 | Mol Sim (MVP) | **Local** | Lipinski filter (CPU, RDKit) | 2 min |
| 5 | Mol Sim (Tier 2) | **Local** | AutoDock Vina (CPU, ~30 min/drug) | Hours |
| 5 | AF3 protein-ligand co-folding | **Colab** | AlphaFold3, T4 (~30 min/complex) | Per complex |
| 6 | Recommendation (MVP/Tier 2) | **Local** | KNN → ranked simulation | 5 min |

---

## Local Execution Constraints (RTX 3050 4GB)

### AMP is Mandatory
Every PyTorch training loop MUST use Automatic Mixed Precision:
```python
from torch.cuda.amp import autocast, GradScaler
from config import USE_AMP

scaler = GradScaler()

for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    with autocast(dtype=torch.float16, enabled=USE_AMP):
        pred = model(X)
        loss = criterion(pred, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
AMP halves VRAM from ~8GB to ~4GB for most models. **Non-negotiable for RTX 3050.**

### Gradient Checkpointing for Deep AEs
For autoencoders with >3 layers, enable gradient checkpointing:
```python
from torch.utils.checkpoint import checkpoint
from config import USE_GRAD_CHECKPOINT

class EncoderBlock(nn.Module):
    def forward(self, x):
        if USE_GRAD_CHECKPOINT and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
```
Trades ~2× slower backward pass for ~40% VRAM reduction.

### Dataset Caps for Local Runs
From `config.py` MVP limits:
- Top `MVP_MAX_GENES = 2000` variable genes (not all 60K)
- `MVP_MAX_CELL_LINES = 200` (not 1800 CCLE)
- `MVP_MAX_DRUGS = 100` (not 300 GDSC2)
- `MVP_MAX_KG_TRIPLES = 100_000` DRKG (not 5.87M)

Apply in code:
```python
from config import MVP_MODE, MVP_MAX_GENES

if MVP_MODE:
    # Select top genes by variance
    gene_var = expr_df.var(axis=0)
    top_genes = gene_var.nlargest(MVP_MAX_GENES).index
    expr_df = expr_df[top_genes]
```

### VRAM Budget Per Phase (with AMP)
| Phase | Model | Approx VRAM |
|---|---|---|
| 1 (VAE+GMM) | Encoder [256,128,32], batch=32 | ~800 MB ✅ |
| 2 (AE+SHAP) | AE [1024,256,32], batch=32 | ~1.2 GB ✅ |
| 3 (MMDRP-lite) | 2 AEs + LMF + MLP, batch=32 | ~2.5 GB ✅ |
| 4 (PyKEEN TransE) | 64-dim embeddings | ~500 MB ✅ |
| 4 (BioGPT base) | ~350 MB model, batch=8 | ~2 GB ✅ |
| 4 (BioGPT-Large) | ~1.5 GB model | ~6 GB ❌ → Colab |
| 5 (Vina) | CPU-only | 0 MB GPU ✅ |
| 5 (AF3) | Co-folding | ~12 GB ❌ → Colab |

---

## Colab Execution Strategy

### Session Management (Critical)
Free Colab disconnects after 12 hours of runtime inactivity.

**Mitigation rules**:
1. Save model checkpoint every epoch (not just best)
2. Always output to Google Drive (`/content/drive/MyDrive/AID_Pipeline/`)
3. Never start a Colab job that takes >10 hours in one session
4. Use Kaggle for longer jobs (30 hr/week budget)

### Google Drive Setup
```python
# Cell 1 of every Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

# Adjust to your Drive folder structure:
DRIVE_ROOT = '/content/drive/MyDrive/AID_Pipeline'
MODELS_DIR = f'{DRIVE_ROOT}/models'
DATA_DIR = f'{DRIVE_ROOT}/data'
```

### Data Transfer Strategy
- **Small files** (<100 MB): upload directly to Colab session storage
- **Medium files** (100 MB – 2 GB): upload to Google Drive, mount
- **Large files** (>2 GB, e.g., modzs.gctx 42 GB): 
  - Upload to Google Drive once
  - Mount in all sessions — DO NOT re-upload each time
  - Alternatively: use Kaggle (persistent datasets up to 100 GB free)

### Which Platform for What
| Task | Use |
|---|---|
| Full Subtype-GAN (2–4 hr) | Colab (start before sleeping) |
| Full MMDRP + gctx (3–5 hr) | Kaggle (longer session, persistent data) |
| BioGPT-Large (3–4 hr) | Colab (fits in T4 15 GB) |
| AlphaFold3 (30–60 min/complex) | Colab T4 (sufficient for small proteins) |

---

## Phase 4: BioGPT Local Strategy

BioGPT base (`microsoft/BioGPT`) fits in 4 GB with batch=8 and AMP.

**Use for embedding extraction only** — not generation:
```python
from transformers import AutoTokenizer, AutoModel
import torch
from config import LLM_MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModel.from_pretrained(LLM_MODEL_NAME).cuda()
model.eval()

# Path string from KG traversal:
path = "sorafenib → inhibits → VEGFR2 → drives → angiogenesis → promotes → LIHC"

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    inputs = tokenizer(path, return_tensors="pt", max_length=128,
                       truncation=True).to("cuda")
    out = model(**inputs)
    embedding = out.last_hidden_state.mean(dim=1)  # [1, 1024]

# Use this embedding as a feature, NOT the raw text output
```

If BioGPT base still OOMs:
```python
# Fallback: PubMedBERT sentence embeddings (CPU-viable, 420 MB)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
embedding = model.encode(path)  # [768] vector, no GPU needed
```

---

## Phase 5: AutoDock Vina Local + AF3 Colab

### AutoDock Vina (Local, CPU)
```bash
pip install vina
```
```python
from vina import Vina
from config import VINA_EXHAUSTIVENESS

v = Vina(sf_name='vina')
v.set_receptor('protein.pdbqt')
v.set_ligand_from_file('ligand.pdbqt')
v.compute_vina_maps(center=[x, y, z], box_size=[20, 20, 20])
v.dock(exhaustiveness=VINA_EXHAUSTIVENESS, n_poses=5)
# Returns binding energy in kcal/mol — more negative = better binding
```

**Speed on CPU**: ~5–30 min per drug-target pair (depending on protein size).
For 50 candidates × 3 LIHC targets = 150 docking runs ≈ 12–75 hours on CPU.
**Strategy**: Run overnight locally, or push to Kaggle for parallelism.

### AlphaFold3 (Colab) — see `colab/colab_phase5_af3.py`

---

## Data Download Priority (Before Starting Phase 1)

Execute in this order (all free):

```bash
# 1. GDSC2 (~50 MB) — needed for Phase 3 MVP
# Download from: https://www.cancerrxgene.org/downloads/bulk_download
# File: GDSC2_fitted_dose_response_27Oct23.xlsx (convert to CSV)

# 2. TCGA-LIHC RNA-seq (~200 MB) — needed for Phase 1
# Download from: https://portal.gdc.cancer.gov
# Filter: Project=TCGA-LIHC, Data Category=Transcriptome Profiling, Workflow=HTSeq-Counts

# 3. CCLE expression (~500 MB) — needed for Phase 3
# Download from: https://depmap.org/portal/download → OmicsExpressionProteinCodingGenesTPMLogp1.csv

# 4. DRKG (~100 MB compressed) — needed for Phase 4
# wget https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tsv.gz
```
