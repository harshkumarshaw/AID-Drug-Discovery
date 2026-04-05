# System Architecture
## Updated: April 2026 — Reflects 2026 SOTA Methods

## End-to-End Pipeline Flow (April 2026)

```
Raw Data Sources
     │
     ├─ TCGA-LIHC (clinical + biospecimen + RNA-seq + CNV)
     ├─ E-MTAB-5902 (WGS variants — chunked loading)
     ├─ LINCS L1000 (modzs.gctx via cmapPy + consensi TSVs)
     ├─ ChEMBL-35 (SDF + SQLite extracted + h5)
     ├─ GDSC2 / CCLE (IC50 cell viability — downloaded to data/external/)
     └─ DRKG (Drug Repurposing Knowledge Graph — pre-built)
     │
     ▼
[Phase 1] Disease Subtyping                            [2026: Subtype-GAN + Transformer]
     Multi-omics (mRNA + CNV + clinical) → Subtype-GAN adversarial AE → GMM → patient_clusters.csv
     Optional: Transformer fusion encoder for cross-omics attention
     │
     ▼
[Phase 2] Biomarker Discovery                          [2026: Multi-modal AE + SHAP]
     Variant counts + RNA-seq + clusters → Per-omics autoencoders → LMF fusion
     → SHAP DeepExplainer + Integrated Gradients → biomarkers.csv
     │
     ▼
[Phase 3] Drug Response Prediction                     [2026: MMDRP-style confirmed SOTA]
     LINCS modzs.gctx + GDSC2 IC50 + CCLE → Omic AEs + Drug AE + LMF + MLP
     → drug_response_model.pt + drug_response_predictions.csv
     │
     ▼
[Phase 4] Drug Repurposing                             [2026: LLM + Knowledge Graph]
     DRKG + ChEMBL + KEGG + DisGeNET → PyKEEN TransE/RotatE
     + BioGPT multi-hop path reasoning → LLM-generated explanation
     + DeepDRA scoring → repurposing_candidates.csv
     │
     ▼
[Phase 5] Molecular Simulation                         [2026: AlphaFold3 open-source]
     Top-50 candidates + LIHC targets → AlphaFold3 protein-ligand co-folding
     + ADMET-AI (42 endpoints) → interaction_scores.csv
     │
     ▼
[Phase 6] Digital Twin Patient Simulation              [2026: Digital Twin framing]
     Patient profile → Subtype-GAN inference → Simulated IC50 per drug
     → AF3 binding filter → ADMET filter → SHAP + LLM clinical narrative
     → ranked_recommendations.json
     │
     ▼
[Stretch] Phase 7 — De Novo Drug Generation           [2026: Active clinical translation]
     LIHC biomarker genes → DiffMol/MolGAN → Lipinski filter
     → ADMET-AI filter → AF3 binding score → novel_candidates.csv
     │
     ▼
Flask REST API (api/app.py)
     ├─ POST /api/v1/subtype          → patient cluster assignment
     ├─ POST /api/v1/drug_response    → predicted IC50
     ├─ POST /api/v1/repurpose        → LLM-explained candidates
     ├─ POST /api/v1/simulate         → AF3 binding + ADMET
     ├─ POST /api/v1/recommend        → digital twin treatment plan
     └─ GET  /api/v1/status           → pipeline health
```

---

## 2026 Architecture Upgrades Applied

### AU-001: AlphaFold3 Replaces AF2 + DiffDock
- **Date**: April 2026
- **Change**: Use `alphafold3` (open-source Jan 2026) for protein-ligand co-folding
- **Replaces**: `colabfold` (AlphaFold2) + DiffDock (separate docking step)
- **Why**: AF3 handles protein + small molecule simultaneously in one model; more accurate
- **Source**: `github.com/google-deepmind/alphafold3`
- **Impact**: Phase 5 — simpler workflow, better binding pose accuracy

### AU-002: LLM + Knowledge Graph for Drug Repurposing
- **Date**: April 2026
- **Change**: Add BioGPT/BioBERT path reasoning on top of PyKEEN KG embeddings
- **Why**: 2025–2026 SOTA (CLEAR, LLaDR, DrugCORpath) shows LLM on KG paths
  gives mechanism-aware scores + human-readable explanations
- **Architecture**: PyKEEN score → BioGPT reads drug→gene→disease→pathway paths
  as "biological sentences" → path-aware repurposing score + LLM explanation text
- **Tools**: `pykeen`, `transformers` (BioGPT or PubMedBERT)
- **Impact**: Phase 4 — candidates come with clinical justification built-in

### AU-003: Digital Twin Patient Simulation
- **Date**: April 2026
- **Change**: Reframe Phase 6 from "KNN recommendation" to "Digital Twin simulation"
- **Why**: 2026 clinical AI standard language; same computation, much stronger research framing
- **Impact**: Phase 6 — patient gets a virtual simulation of their response before treatment

### AU-004: Transformer Fusion for Disease Subtyping
- **Date**: April 2026
- **Change**: Add optional Transformer encoder as comparison model alongside Subtype-GAN
- **Why**: 2025–2026 Transformers outperform GAN approaches on some benchmarks
- **Impact**: Phase 1 — two models trained, best selected or ensembled

---

## Bug Catalogue

### Bug 001 — LINCS Concat Produces 0 Rows
- **File**: `LINCS/drug_response_prediction.ipynb`
- **Root cause**: `pd.concat([drugbank, knockdown, overexpression, pert_id], axis=1).dropna()` — the four files have completely different perturbagen sets, inner intersection is empty.
- **Fix**: Use each LINCS file separately. For drug response: use `consensi-drugbank.tsv` only. Load full signatures from `modzs.gctx` via cmapPy.
- **Status**: OPEN

### Bug 002 — ChEMBL SQLite Not Extracted
- **File**: `Drug Repurposing/Drug_Repurposing_Pipeline.ipynb`
- **Root cause**: `sqlite3.connect("chembl_35_sqlite.tar.gz")` — must extract first.
- **Fix**: `import tarfile; tarfile.open(...).extractall("chembl_db/")` then connect to `.db`
- **Status**: OPEN

### Bug 003 — Disease Subtyping GroupBy TypeError
- **File**: `ai_drug_discovery_pipeline.ipynb`
- **Root cause**: `data.groupby('Cluster').mean()` on DataFrame with UUID string columns.
- **Fix**: `data.select_dtypes(include=[np.number]).groupby(data['Cluster']).mean()`
- **Status**: OPEN

### Bug 004 — Biomarker Discovery Missing Column
- **File**: `ai_drug_discovery_pipeline.ipynb`
- **Root cause**: `biomarker_discovery()` expects `Disease_Status` column — doesn't exist.
- **Fix**: `df['Disease_Status'] = df['vital_status'].map({'Alive': 0, 'Dead': 1})`
- **Status**: OPEN

### Bug 005 — StandardScaler with 0 Samples
- **File**: `LINCS/drug_response_prediction.ipynb`
- **Root cause**: Downstream of Bug 001 — empty DataFrame to `StandardScaler().fit_transform()`
- **Fix**: Fix Bug 001 first; add `if df.shape[0] == 0: raise ValueError(...)`
- **Status**: OPEN (blocked by Bug 001)

### Bug 006 — Hardcoded Absolute Paths (D:\)
- **File**: All notebooks
- **Root cause**: Paths like `D:\Research Study\AID\Code\...` — wrong drive.
- **Fix**: Import all paths from `config.py`
- **Status**: OPEN

### Bug 007 — E-MTAB-5902 Single Sample
- **File**: `E-MTAB-5902/biomarker_discovery_pipeline.ipynb`
- **Root cause**: Only 1 patient sample. ML requires multiple samples.
- **Fix**: Download additional EBI samples + use TCGA-LIHC RNA-seq
- **Status**: OPEN

### Bug 008 — modzs.gctx Not Integrated
- **File**: Root directory
- **Root cause**: 42 GB LINCS dataset not loaded anywhere.
- **Fix**: `cmapPy.pandasGEXpr.parse()` with chunk-based loading
- **Status**: PLANNED

---

## Design Decisions Log

### DD-001: GDSC/CCLE for Drug Response Labels
- **Date**: 2026-04-06
- **Decision**: Integrate GDSC2 + CCLE IC50 data as true drug response labels.
- **Rationale**: User needs actual cell viability measurements, not just transcriptomic perturbation.
- **Impact**: Download GDSC2 (~100 MB) + CCLE (~500 MB) to `data/external/`

### DD-002: EBI Download + TCGA RNA-seq for Biomarkers
- **Date**: 2026-04-06
- **Decision**: Download additional E-MTAB samples AND use TCGA-LIHC RNA-seq.
- **Rationale**: Dual strategy for best biomarker coverage.

### DD-003: Full LINCS via modzs.gctx
- **Date**: 2026-04-06
- **Decision**: Integrate modzs.gctx via cmapPy (full 42 GB dataset).
- **Rationale**: Full dataset gives richer perturbation signatures.

### DD-004: Full ChEMBL-35 SQLite Extraction
- **Date**: 2026-04-06
- **Decision**: Extract `chembl_35_sqlite.tar.gz` to `chembl_db/`.
- **Rationale**: Full SQLite gives bioactivity, target types, mechanism of action.

### DD-005: AlphaFold3 Over AF2+DiffDock
- **Date**: April 2026
- **Decision**: Use AF3 (open-source Jan 2026) for protein-ligand co-folding.
- **Rationale**: AF3 handles protein+ligand simultaneously; more accurate; single workflow step.
- **Impact**: Phase 5 simplified and upgraded.

### DD-006: LLM + KG Hybrid for Drug Repurposing
- **Date**: April 2026
- **Decision**: Use PyKEEN (KG embedding) + BioGPT (multi-hop path reasoning) for repurposing.
- **Rationale**: 2025–2026 SOTA (CLEAR, LLaDR, DrugCORpath) — mechanism-aware + explainable.
- **Impact**: Phase 4 produces LLM-generated clinical justification per candidate.

### DD-007: Digital Twin Framing for Personalized Medicine
- **Date**: April 2026
- **Decision**: Reframe Phase 6 as "Digital Twin patient simulation".
- **Rationale**: 2026 clinical AI standard — same computation, stronger clinical positioning.
- **Impact**: Phase 6 is now framed as patient-specific drug response simulation.

### DD-008: DRKG as Knowledge Graph Base
- **Date**: April 2026
- **Decision**: Use Drug Repurposing Knowledge Graph (DRKG) as base graph + ChEMBL/KEGG augmentation.
- **Rationale**: DRKG is pre-built (Hetionet + DRKG edges) — saves 2–3 days of KG construction.
- **Impact**: Phase 4 can start on pre-built graph and augment with project-specific data.
