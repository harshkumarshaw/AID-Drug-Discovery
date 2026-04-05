# AI Drug Discovery & Personalized Medicine Pipeline
## Project Context for Claude — Updated April 2026

> **Mission**: Build a complete end-to-end AI pipeline that simulates molecular interactions,
> identifies biomarkers, predicts drug response, repurposes existing drugs, and recommends
> personalized treatments — focused on cancer and other life-threatening diseases where
> approved drugs do not yet exist. Primary focus: **Liver Hepatocellular Carcinoma (LIHC)**.
>
> **Hardware**: RTX 3050 4GB VRAM + 16GB RAM. COMPUTE_TIER = "single_gpu" in config.py.
> AMP (float16) is mandatory in ALL PyTorch training loops. See @docs/compute_strategy.md.
> Heavy phases (AF3, full MMDRP, BioGPT-Large) run on Colab/Kaggle — see colab/ directory.

---

## 📁 Project Structure

```
f:\Research_Study\AID\AID Code\
├── CLAUDE.md                        ← You are here (updated April 2026)
├── config.py                        ← Central config (paths, hyperparams)
├── requirements.txt                 ← All Python dependencies
│
├── pipeline/                        ← Core ML pipeline modules
│   ├── __init__.py
│   ├── data_loader.py               ← Unified data loading + validation
│   ├── disease_subtyping.py         ← Phase 1: Subtype-GAN + Transformer
│   ├── biomarker_discovery.py       ← Phase 2: Multi-omics AE + LMF + SHAP
│   ├── drug_response.py             ← Phase 3: MMDRP-style multimodal
│   ├── drug_repurposing.py          ← Phase 4: LLM + KG (PyKEEN + BioGPT)
│   ├── molecular_simulation.py      ← Phase 5: AlphaFold3 + ADMET-AI
│   └── personalized_medicine.py    ← Phase 6: Digital Twin simulation
│
├── api/                             ← Flask REST API
│   └── app.py                       ← 6 pipeline endpoints
│
├── notebooks/                       ← Canonical, fixed Jupyter notebooks
│   ├── 01_clinical_biospecimen.ipynb
│   ├── 02_biomarker_discovery.ipynb
│   ├── 03_drug_response.ipynb
│   ├── 04_drug_repurposing.ipynb
│   ├── 05_molecular_simulation.ipynb
│   └── 06_personalized_medicine.ipynb
│
├── tests/                           ← Unit tests (pytest)
├── models/                          ← Saved trained models
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/                    ← GDSC2, CCLE downloads
│
└── docs/
    ├── architecture.md              ← @docs/architecture.md (bug catalogue, decisions)
    ├── datasets.md                  ← @docs/datasets.md (schemas, locations)
    ├── pipeline_flow.md             ← @docs/pipeline_flow.md (I/O diagrams)
    └── research_analysis.md        ← @docs/research_analysis.md (SOTA, gaps)
```

---

## 🧬 Tech Stack (April 2026)

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Subtyping | PyTorch (Subtype-GAN) | Custom GAN architecture |
| ML/DL | scikit-learn, TensorFlow 2.x | |
| Drug response | PyTorch (MMDRP-style) | Multi-modal fusion |
| KG Repurposing | PyKEEN + transformers (BioGPT) | LLM + KG hybrid |
| Molecular sim | **AlphaFold3** (open-source Jan 2026) | Replaces AF2+DiffDock |
| ADMET | **ADMET-AI** | 42 endpoints from SMILES |
| Cheminformatics | RDKit, cmapPy, deepchem | |
| Data | pandas, numpy, h5py | |
| Visualization | matplotlib, seaborn, plotly | |
| API | Flask 3.x, flask-cors | |
| XAI | shap, captum (Integrated Gradients) | At every phase |
| Testing | pytest | |

---

## 📊 Datasets (Canonical Locations)

| Dataset | Purpose | Location |
|---|---|---|
| TCGA-LIHC | Clinical + biospecimen + RNA-seq + CNV | `TCGA/` |
| E-MTAB-5902 | Genomic variants (WGS) | `E-MTAB-5902/` |
| LINCS L1000 | Drug perturbation profiles | `LINCS/` |
| ChEMBL-35 | Drug structures + targets | `Drug Repurposing/` |
| GDSC2 + CCLE | Cell viability IC50 labels | `data/external/` (download) |
| modzs.gctx | Full LINCS L1000 (42 GB) | `LINCS/modzs.gctx` |
| DRKG | Drug Repurposing Knowledge Graph | `data/external/drkg/` (download) |

> **RULE**: Always import paths from `config.py`. NEVER hardcode absolute paths.

---

## ⚙️ Coding Conventions

- **4-space indentation** (PEP 8)
- **Type hints** on all function signatures
- **Docstrings**: Google style on all public functions
- **No magic numbers**: extract to `config.py` constants
- **Data validation**: always check `df.shape[0] > 0` before fitting
- **Chunked loading**: files > 500 MB must use `chunksize` or `h5py`
- **Error messages**: `f"Step [{step_name}] failed: {e}"` format
- **Save outputs**: every pipeline step saves to `data/processed/`
- **SHAP**: every model exposes `.explain(X)` method returning SHAP values
- **Logging**: use `logging.getLogger(__name__)` not `print()`

---

## 🔑 Key Commands

```bash
# Validate environment & paths
python config.py

# Run full pipeline
python -m pipeline

# Run specific phase
python -m pipeline.drug_response

# Start Flask API
python api/app.py

# Run tests
pytest tests/ -v

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab notebooks/
```

---

## ⚠️ Known Bugs (See @docs/architecture.md for full catalogue)

1. **Bug 001** — LINCS concat → 0 rows: do NOT `pd.concat` all 4 TSVs with `axis=1`
2. **Bug 002** — ChEMBL SQLite: must extract `.tar.gz` before `sqlite3.connect()`
3. **Bug 003** — Disease subtyping groupby: `select_dtypes(include=[np.number])` first
4. **Bug 004** — `Disease_Status` missing: derive from `vital_status` in clinical data
5. **Bug 006** — All `D:\` paths: replace with `config.py` imports

---

## 🎯 Pipeline Phases & 2026 Models

### Phase 1 — Disease Subtyping
- **Model**: Subtype-GAN (adversarial multi-omics autoencoder + GMM)
- **Novel**: First Subtype-GAN application to TCGA-LIHC
- **Output**: `data/processed/patient_clusters.csv`

### Phase 2 — Biomarker Discovery
- **Model**: Per-omics autoencoders → LMF fusion → SHAP + Integrated Gradients
- **Novel**: Multi-modal LIHC biomarker (variant + expression + clinical)
- **Output**: `data/processed/biomarkers.csv`

### Phase 3 — Drug Response Prediction
- **Model**: MMDRP-style (omic AEs + drug AE + LMF + MLP) ← confirmed SOTA 2026
- **Novel**: First LINCS perturbation signatures → GDSC IC50 multimodal model
- **Output**: `models/drug_response/model.pt`

### Phase 4 — Drug Repurposing ⭐ 2026 UPGRADED
- **Model**: PyKEEN TransE (KG base) + **BioGPT** (path reasoning) + DeepDRA
- **Novel**: LLM+KG repurposing for LIHC (no existing curative drug)
- **Output**: `data/processed/repurposing_candidates.csv` with LLM explanations

### Phase 5 — Molecular Simulation ⭐ 2026 UPGRADED
- **Model**: **AlphaFold3** (open-source Jan 2026, protein-ligand co-folding) + ADMET-AI
- **Novel**: AF3-based simulation for LIHC-specific drug-target pairs
- **Output**: `data/processed/interaction_scores.csv`

### Phase 6 — Digital Twin Patient Simulation ⭐ 2026 UPGRADED
- **Model**: Subtype-GAN inference → MMDRP inference → SHAP + LLM clinical narrative
- **Novel**: First explainable digital twin recommender for LIHC
- **Output**: JSON treatment recommendation with explanation

---

## 📝 Development Rules

- Before modifying any notebook, create the equivalent `.py` module in `pipeline/`
- Every new dataset must be documented in `docs/datasets.md`
- Every architectural decision must be logged in `docs/architecture.md`
- Every model must expose `.explain(X)` using SHAP or Integrated Gradients
- Run `pytest tests/ -v` before marking any phase complete
- Use `tasks.md` in the conversation artifacts to track sprint progress

---

## 🔗 Additional Context

- @docs/architecture.md — Full bug catalogue + design decisions log (April 2026)
- @docs/datasets.md — Dataset schemas, locations, download instructions
- @docs/pipeline_flow.md — I/O flow diagrams for all 6 phases
- @docs/research_analysis.md — SOTA literature review + 2026 verification
- @.claude/rules/data_loading.md — Rules for loading large biomedical files
- @.claude/rules/ml_training.md — Rules for model training and validation
- @.claude/skills/fix_notebook.md — Repair and refactor broken notebooks
- @.claude/skills/add_pipeline_phase.md — Add new pipeline stages cleanly
