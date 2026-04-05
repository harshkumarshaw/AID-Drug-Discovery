# 🧬 AI-Powered Drug Discovery & Personalized Medicine Pipeline

<div align="center">

**An end-to-end AI pipeline for drug discovery and personalized cancer treatment**  
Targeting Liver Hepatocellular Carcinoma (LIHC) · 2026 SOTA · MVP-First · Hardware-Aware

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-yellow)]()
[![Tests](https://img.shields.io/badge/Tests-17%20passing-brightgreen)]()

</div>

---

## The Problem

Liver hepatocellular carcinoma (LIHC) is the **6th most common cancer** worldwide with a **5-year survival rate of only 18%**. Fewer than 8 FDA-approved systemic therapies exist — and drug development takes 12–15 years and costs over $2.6 billion per drug.

Current challenges:
- **Tumour heterogeneity**: LIHC has at least 4 molecular subtypes, each requiring different treatments
- **No personalization**: Treatment protocols are one-size-fits-all, ignoring individual genomic profiles  
- **Slow repurposing**: Approved drugs for other cancers are rarely re-evaluated for LIHC systematically
- **Drug-target mismatch**: Binding predictions lack experimental reproducibility at scale

---

## The Solution

A six-phase AI pipeline that ingests multi-omic patient data and produces personalized, explainable drug recommendations:

```
┌──────────────────────────────────────────────────────────────────────────┐
│              AI Drug Discovery & Personalized Medicine Pipeline           │
│                          LIHC Focus · April 2026                         │
└──────────────────────────────────────────────────────────────────────────┘

  TCGA-LIHC        ┌─────────────────┐   Subtypes +
  RNA-seq     ───▶ │  Phase 1        │──▶ Survival Groups
  Clinical         │  Disease        │
  CNV / Mutation   │  Subtyping      │   Model: VAE+GMM (local)
                   │  (Subtype-GAN)  │          Subtype-GAN (Colab)
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐   Top 50 Biomarkers
                   │  Phase 2        │──▶ SHAP Rankings
                   │  Biomarker      │    KEGG Pathways
                   │  Discovery      │
                   │  (AE + SHAP)    │   Validation: COSMIC Gene Census
                   └────────┬────────┘
                            │
  GDSC2 + CCLE  ──▶ ┌───────▼────────┐   IC50 Predictions
  LINCS Sigs         │  Phase 3       │──▶ Cell-line Sensitivity
                     │  Drug Response │
                     │  (MMDRP-style) │   Model: MMDRP-lite (local)
                     └───────┬────────┘          Full MMDRP (Kaggle)
                             │
  ChEMBL + DRKG ──▶ ┌────────▼────────┐   Top-50 Candidates
  BioGPT paths       │  Phase 4        │──▶ Mechanism Explanations
                     │  Drug           │
                     │  Repurposing    │   Anchor: Sorafenib/Lenvatinib
                     │  (KG + LLM)    │      must appear in Top-20
                     └────────┬────────┘
                              │
  PDB structures  ──▶ ┌───────▼────────┐   Binding Scores
  ADMET filters        │  Phase 5       │──▶ ADMET Safety Flags
                       │  Molecular     │
                       │  Simulation    │   Local: AutoDock Vina
                       │  (AF3/Vina)   │   Paper: AlphaFold3 (Kaggle)
                       └───────┬────────┘
                               │
  Patient Profile ──▶ ┌────────▼────────┐   Personalized
                      │  Phase 6        │──▶ Drug Recommendations
                      │  Digital Twin   │    Clinical Trial Matches
                      │  Recommendation │    Explainable JSON Output
                      └─────────────────┘
```

---

## Key Research Contributions

| # | Contribution | Novelty |
|---|---|---|
| 1 | **Subtype-GAN for LIHC** | First adversarial subtyping model applied specifically to TCGA-LIHC multi-omics |
| 2 | **MMDRP-style drug response** | LINCS L1000 + GDSC2 + CCLE fusion with LMF for LIHC-specific IC50 prediction |
| 3 | **DRKG + BioGPT repurposing** | KG embeddings + LLM path reasoning in a hybrid repurposing architecture |
| 4 | **AlphaFold3 co-folding** | Protein-ligand co-folding replacing AF2+DiffDock for LIHC targets |
| 5 | **Digital Twin framing** | Patient-level simulation combining subtypes + drug response + ADMET for personalized medicine |
| 6 | **Hardware-aware MVP framework** | Systematic local/cloud split enabling publication-grade results on consumer hardware |

---

## 2026 SOTA Alignment

| Component | Previous (2024) | This Pipeline (2026) |
|---|---|---|
| Protein-Ligand Docking | AlphaFold2 + DiffDock | **AlphaFold3 co-folding** (Jan 2026 open source) |
| Drug Repurposing | Cosine similarity | **KG (PyKEEN) + LLM (BioGPT-Large) hybrid** |
| Patient Stratification | K-Means | **Variational Autoencoder + GMM → Subtype-GAN** |
| Personalized Medicine | KNN treatment matching | **Digital Twin patient simulation** |
| Explainability | Feature importance | **SHAP + Integrated Gradients + LLM path narratives** |

---

## Hardware Strategy

| Environment | Hardware | Use Case |
|---|---|---|
| **Local** | RTX 3050 4GB + 16GB RAM | Development: all 6 phases (reduced dims, AMP float16) |
| **Kaggle** | P100 16GB (free 30hr/week) | Paper results: BioGPT-Large, PyKEEN RotatE, full MMDRP |
| **Colab** | T4 15GB (free 12hr) | AlphaFold3 docking, full LINCS gctx loading |

> **AMP (float16) is mandatory** in all local PyTorch training loops. This halves VRAM usage.  
> See `docs/compute_strategy.md` for phase-by-phase execution decisions.

---

## Project Structure

```
AID_Code/
├── README.md                    ← This file
├── CLAUDE.md                    ← AI assistant context and rules
├── config.py                    ← Single source of truth for all paths/hyperparams
├── requirements.txt             ← Pinned dependencies (no version drift)
├── .gitignore                   ← Excludes large data/model files
│
├── pipeline/                    ← Core pipeline modules
│   ├── disease_subtyping.py     ← Phase 1: VAE+GMM → Subtype-GAN
│   ├── biomarker_discovery.py   ← Phase 2: Autoencoder + SHAP
│   ├── drug_response.py         ← Phase 3: MMDRP-style MLP
│   ├── drug_repurposing.py      ← Phase 4: PyKEEN + BioGPT
│   ├── molecular_simulation.py  ← Phase 5: Vina (local) / AF3 (Kaggle)
│   └── personalized_medicine.py ← Phase 6: Digital Twin
│
├── colab/                       ← Cloud execution scripts
│   ├── colab_phase3_full_mmdrp.py   ← Full MMDRP + gctx on Kaggle
│   └── colab_phase5_af3.py          ← AlphaFold3 on Colab/Kaggle
│
├── scripts/                     ← Utilities
│   ├── download_datasets.py     ← One-command data download
│   └── filter_drkg.py           ← Filter DRKG 5.87M → 500K triples
│
├── tests/                       ← Integration tests
│   └── test_pipeline_e2e.py     ← 17 tests on synthetic data (all passing ✅)
│
├── docs/                        ← Documentation
│   ├── architecture.md          ← System design + bug catalogue
│   ├── pipeline_flow.md         ← Phase-by-phase technical detail
│   ├── datasets.md              ← Dataset schemas and loading notes
│   ├── data_inventory.md        ← Complete download checklist
│   ├── compute_strategy.md      ← Local vs Kaggle execution map
│   └── research_analysis.md     ← 2026 SOTA literature review (55+ papers)
│
├── notebooks/                   ← Exploratory Jupyter notebooks
│   └── 01_clinical_biospecimen.ipynb
│
├── data/                        ← Data (gitignored; see data_inventory.md)
│   ├── processed/               ← Cleaned CSVs
│   ├── external/                ← GDSC2, CCLE, DRKG, STRING
│   └── targets/                 ← PDB protein structures
│
└── .claude/                     ← AI assistant rules and commands
    ├── rules/
    │   ├── data_loading.md
    │   └── ml_training.md
    └── commands/
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/AID-Drug-Discovery.git
cd AID-Drug-Discovery

# Install PyTorch with CUDA first (for RTX 3050 with CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install remaining requirements:
pip install -r requirements.txt
```

### 2. Verify Installation (Run Integration Tests)

```bash
pytest tests/test_pipeline_e2e.py -v
# Expected: 17 passed, 1 skipped ✅
```

This confirms all 6 pipeline phases work on synthetic data **before downloading any real data**.

### 3. Download Required Data

```bash
# Auto-downloads: DRKG, CCLE, PDB targets, pip packages (~1.2 GB)
python scripts/download_datasets.py --tier mvp

# See what needs manual download (GDSC2, TCGA, COSMIC):
python scripts/download_datasets.py --list
```

For GDSC2 and TCGA (require browser/registration):
→ See `docs/data_inventory.md` for step-by-step instructions

### 4. Configure for Your Hardware

Edit `config.py`:
```python
COMPUTE_TIER = "single_gpu"  # or "cpu" or "high_gpu"
MVP_MODE = True              # Keep True until full pipeline validated
```

### 5. Filter DRKG to LIHC-Relevant Triples (Required Before Phase 4)

```bash
# Reduces 5.87M triples → ~500K relevant drug-gene-disease triples
python scripts/filter_drkg.py
```

### 6. Run Pipeline (After Real Data Downloaded)

```python
from pipeline.disease_subtyping import run_subtyping
from config import COMPUTE_TIER, MVP_MODE

# Phase 1: LIHC patient subtyping
clusters = run_subtyping(compute_tier=COMPUTE_TIER, mvp_mode=MVP_MODE)
```

---

## Datasets Used

| Dataset | Source | Size | Used In |
|---|---|---|---|
| TCGA-LIHC | GDC Portal | ~250 MB | Phase 1, 2 |
| GDSC2 | cancerrxgene.org | ~20 MB | Phase 3 |
| CCLE | DepMap | ~600 MB | Phase 3 |
| LINCS L1000 (gctx) | Broad Institute | 42 GB | Phase 3 (Kaggle) |
| DRKG | DGL/DRKG GitHub | ~380 MB | Phase 4 |
| ChEMBL-35 | EBI | ~3 GB | Phase 4 |
| STRING PPI | string-db.org | ~1.2 GB | Phase 4 |
| RCSB PDB (7 targets) | RCSB | ~50 MB | Phase 5 |
| AlphaFold3 weights | DeepMind | ~1.5 GB | Phase 5 (Kaggle) |

All datasets are **publicly available and free**. No proprietary data is used.

---

## Validation Anchors

> "Any drug repurposing method must recover sorafenib and lenvatinib (both FDA-approved for LIHC) in the top-20 candidates. If they don't appear, the method is broken."

| Phase | Validation Anchor | Pass Threshold |
|---|---|---|
| 1 — Subtyping | Cluster enriched for Stage IV + Dead patients | KM log-rank p < 0.05 |
| 2 — Biomarkers | Top-10 include ≥3 known LIHC oncogenes (TP53, CTNNB1, AFP) | ≥3/10 |
| 3 — Drug Response | Sorafenib IC50 < 25th percentile for HCC cell lines | Lower quartile |
| 4 — Repurposing | Sorafenib + Lenvatinib in top-50 candidates | **Hard gate** |
| 5 — Mol Sim | ≥5 candidates pass Lipinski + ADMET filters | ≥5 of top-10 |
| 6 — Recommend | Known drug in top recommendation for HCC patient profile | PASS |

---

## Results (TBD — In Progress)

| Metric | Dev Baseline (Local) | Paper Target (Kaggle) |
|---|---|---|
| Subtyping KM log-rank p | — | < 0.05 |
| Drug Response RMSE (GDSC2) | — | < 1.0 |
| Drug Response Pearson r | — | > 0.5 |
| Repurposing Hits@20 | — | Sorafenib + Lenvatinib ✓ |
| AF3 ipTM (binding confidence) | — | > 0.5 for ≥3 targets |

*Results section will be updated as pipeline phases complete.*

---

## Architecture Principles

1. **MVP-First**: All 6 phases run end-to-end on synthetic data before any real data work
2. **Hardware-Aware**: `COMPUTE_TIER` in `config.py` drives model selection automatically  
3. **Reproducible**: All hyperparameters in `config.py`; no magic numbers in code
4. **Explainable**: SHAP values, Integrated Gradients, and BioGPT path narratives at every phase
5. **Validated**: Every phase has a scientific baseline to beat and a validation anchor to recover

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{aid_pipeline_2026,
  title  = {AI-Powered Drug Discovery and Personalized Medicine Pipeline for LIHC},
  author = {Shaw, Harsh Kumar},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/AID-Drug-Discovery}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> This project is for academic research only. It does not constitute medical advice.  
> All drug recommendations are computational predictions and have not been clinically validated.
