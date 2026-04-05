# Complete Dataset & Resource Inventory
## AI Drug Discovery Pipeline — LIHC Focus
### Updated: April 2026

---

## How to Read This Document

Each resource is tagged:
- 🔴 **REQUIRED** — Pipeline cannot run this phase without it
- 🟡 **IMPORTANT** — Significantly improves model quality
- 🟢 **OPTIONAL** — For publication-grade polish or stretch phases
- ✅ Already have | ⬇️ Must download | 📝 Register required | 💰 Paid

---

## PHASE 1 — Disease Subtyping

| Resource | Tag | Size | Source + URL | Command | Config Key |
|---|---|---|---|---|---|
| TCGA-LIHC Clinical (have) | ✅ 🔴 | ~5 MB | Already in `TCGA/` | — | `TCGA_CLINICAL_CSV` |
| TCGA-LIHC Biospecimen (have) | ✅ 🔴 | ~5 MB | Already in `TCGA/` | — | `TCGA_BIOSPECIMEN_CSV` |
| **TCGA-LIHC RNA-seq (HTSeq Counts)** | ⬇️ 🔴 | ~200 MB | [GDC Portal](https://portal.gdc.cancer.gov) | See below | `TCGA_RNASEQ_CSV` |
| **TCGA-LIHC Somatic Mutation (MAF)** | ⬇️ 🔴 | ~50 MB | GDC Portal | See below | `TCGA_MUTATION_MAF` |
| TCGA-LIHC CNV | ⬇️ 🟡 | ~30 MB | GDC Portal | See below | `TCGA_CNV_CSV` |
| TCGA-LIHC Methylation (450K array) | ⬇️ 🟢 | ~2 GB | GDC Portal | See below | `TCGA_METHYL_CSV` |

### Download Instructions — TCGA (All Free, Requires Browser)

```
1. Go to: https://portal.gdc.cancer.gov/exploration
2. Filter: Cases → Project = TCGA-LIHC
3. Files tab → Data Category:
   - "Transcriptome Profiling" + Workflow = "STAR - Counts"    → RNA-seq
   - "Simple Nucleotide Variation" + Data Type = "Masked Somatic Mutation" → MAF
   - "Copy Number Variation" + Data Type = "Gene Level Copy Number" → CNV

4. Add all to Cart → Download Manifest
5. Install GDC client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
6. Run: gdc-client download -m manifest.txt -d TCGA/raw/
```

**Alternative (faster, no client needed):** Use TCGAbiolinks in Python/R:
```python
# pip install pyreadr requests
# OR use TCGAbiolinks in R (faster for TCGA downloads)
```

---

## PHASE 2 — Biomarker Discovery

| Resource | Tag | Size | Source + URL | Notes |
|---|---|---|---|---|
| E-MTAB-5902 variant TSV (have) | ✅ 🟡 | 2.5 GB | Already in `E-MTAB-5902/` | Single sample — supplementary only |
| **TCGA-LIHC RNA-seq** | ⬇️ 🔴 | ~200 MB | GDC Portal (same as Phase 1) | PRIMARY biomarker source |
| **TCGA-LIHC Somatic Mutation MAF** | ⬇️ 🔴 | ~50 MB | GDC Portal | Mutation-level biomarkers |
| **COSMIC Cancer Gene Census** | 📝 🟡 | ~1 MB CSV | [cosmic-gene-census](https://cancer.sanger.ac.uk/census) | Free registration. Used to VALIDATE top biomarkers against known LIHC oncogenes |
| **MSigDB Gene Sets** | ⬇️ 🟡 | ~50 MB | [gsea-msigdb.org](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/c2.cp.kegg.v2023.2.Hs.symbols.gmt) | Free. KEGG pathway gene sets for enrichment analysis of top biomarkers |
| **Human Protein Atlas — LIHC** | ⬇️ 🟢 | ~5 MB | [proteinatlas.org](https://www.proteinatlas.org/download/pathology.tsv.zip) | Free. Cross-reference biomarkers against protein expression in liver cancer |
| Additional E-MTAB samples | ⬇️ 🟢 | Varies | [EBI ArrayExpress](https://www.ebi.ac.uk/biostudies/arrayexpress/studies?query=hepatocellular+carcinoma) | Studies: E-MTAB-5202, E-MTAB-4428 |

---

## PHASE 3 — Drug Response Prediction

| Resource | Tag | Size | Source + URL | Command | Config Key |
|---|---|---|---|---|---|
| LINCS consensi TSVs (have) | ✅ 🔴 | ~200 MB | Already in `LINCS/` | — | `LINCS_DRUGBANK_TSV` etc |
| modzs.gctx (have, Colab only) | ✅ 🟡 | 42 GB | Already in `LINCS/` | — | `LINCS_GCTX` |
| **GDSC2 Drug Sensitivity** | ⬇️ 🔴 | ~20 MB | [cancerrxgene.org](https://www.cancerrxgene.org/downloads/bulk_download) | See below | `GDSC2_CSV` |
| **CCLE Expression** | ⬇️ 🔴 | ~600 MB | [depmap.org](https://depmap.org/portal/download/all/) | See below | `CCLE_EXPRESSION_CSV` |
| **CCLE Somatic Mutations** | ⬇️ 🟡 | ~250 MB | depmap.org (same page) | See below | `CCLE_MUTATION_CSV` |
| CCLE CNV | ⬇️ 🟡 | ~80 MB | depmap.org (same page) | See below | `CCLE_CNV_CSV` |
| **PRISM Drug Repurposing Screen** | ⬇️ 🟡 | ~100 MB | [depmap.org](https://depmap.org/portal/download/all/) | Search "PRISM" | alternative IC50 labels |
| Drug SMILES (from ChEMBL API) | ⬇️ 🔴 | API | [ChEMBL REST API](https://www.ebi.ac.uk/chembl/api/data/molecule?format=json) | `requests` in Python | derived from ChEMBL-35 |

### Download Commands

```bash
# GDSC2 — from cancerrxgene.org (free, no login)
# Go to: https://www.cancerrxgene.org/downloads/bulk_download
# Download: "GDSC2 fitted dose-response parameters" (.xlsx) → convert to CSV
# File: GDSC2_fitted_dose_response_27Oct23.xlsx

# CCLE / DepMap — use DepMap Public release (24Q4 recommended)
# All files from: https://depmap.org/portal/download/all/

# Expression (TPM, log1p normalized — use this one)
wget "https://depmap.org/portal/api/downloads/files?fileName=OmicsExpressionProteinCodingGenesTPMLogp1.csv" \
     -O "data/external/ccle_expression.csv"

# Somatic mutations (binary matrix: gene × cell_line, 1=damaging mutation)
wget "https://depmap.org/portal/api/downloads/files?fileName=OmicsSomaticMutationsMatrixDamaging.csv" \
     -O "data/external/ccle_mutations.csv"

# CNV (segments)
wget "https://depmap.org/portal/api/downloads/files?fileName=OmicsCNSegmentsProfile.csv" \
     -O "data/external/ccle_cnv.csv"

# Cell line mapping (CCLE name → depmap_id, used for harmonization)
wget "https://depmap.org/portal/api/downloads/files?fileName=Model.csv" \
     -O "data/external/ccle_model_info.csv"
```

> **Cell line name harmonization is hard.** GDSC2 uses `HCC-LM3`, CCLE uses `HCCLM3_LIVER`.
> Use `data/external/ccle_model_info.csv` (DepMap Model metadata) to map between name conventions.

---

## PHASE 4 — Drug Repurposing

| Resource | Tag | Size | Source + URL | Command | Config Key |
|---|---|---|---|---|---|
| ChEMBL-35 SQLite (have) | ✅ 🔴 | ~3 GB compressed | Already in `Drug Repurposing/` | Extract first | `CHEMBL_SQLITE_ARCHIVE` |
| **DRKG (Drug Repurposing KG)** | ⬇️ 🔴 | ~100 MB compressed | [GitHub gnn4dr/DRKG](https://github.com/gnn4dr/DRKG) | See below | `DRKG_TRIPLES_TSV` |
| **DeepDRA model** (have) | ✅ 🔴 | — | Already in `Drug Repurposing/DeepDRA-main.zip` | — | `DEEPDRA_DIR` |
| **DisGeNET** | ⬇️ 🟡 | ~500 MB | [disgenet.com/downloads](https://www.disgenet.com/downloads) | Free registration | Gene-disease associations for KG augmentation |
| **STRING PPI** | ⬇️ 🟡 | ~1 GB | [string-db.org](https://string-db.org/cgi/download?sessionId=&species_text=Homo+sapiens) | See below | Protein-protein interactions for KG edges |
| KEGG PATHWAY | 🟢 | API | [KEGG REST API](https://rest.kegg.jp) | `import requests` | Free API (no bulk download). Pathway info per gene |
| **BioGPT model** | ⬇️ 🔴 | ~350 MB | [HuggingFace](https://huggingface.co/microsoft/BioGPT) | `auto` via transformers | `LLM_MODEL_NAME` |
| BioGPT-Large (Colab only) | ⬇️ 🟡 | ~1.5 GB | [HuggingFace](https://huggingface.co/microsoft/BioGPT-Large) | auto | `LLM_MODEL_NAME_COLAB` |
| PubMedBERT (CPU fallback) | ⬇️ 🟡 | ~420 MB | [HuggingFace](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) | auto | `LLM_FALLBACK` |
| **OncoKB** | 📝 🟢 | API | [oncokb.org](https://www.oncokb.org/apiAccess) | Free academic license | Drug-gene clinical actionability for LIHC |

### Download Commands

```bash
# DRKG (~100 MB, free, no login)
wget https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tsv.gz \
     -O data/external/drkg/drkg.tsv.gz
cd data/external/drkg && gunzip drkg.tsv.gz

# STRING PPI (Homo sapiens, high-confidence links)
wget "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz" \
     -O data/external/string_ppi.txt.gz
gunzip data/external/string_ppi.txt.gz

# DisGeNET (gene-disease associations)
# Requires free registration at: https://www.disgenet.com/
# Download: "curated gene-disease associations" → all_gene_disease_associations.tsv.gz

# BioGPT (downloaded automatically on first use via transformers)
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BioGPT')"
```

---

## PHASE 5 — Molecular Simulation

| Resource | Tag | Size | Source + URL | Notes | Config Key |
|---|---|---|---|---|---|
| **AlphaFold3 weights** | 📝 🟡 | ~1.5 GB | [DeepMind form](https://forms.gle/svvpY4u2jsHEwWYS6) | 1–7 day wait for email. Free academic use only | `AF3_MODEL_DIR` |
| **AlphaFold3 source** | ⬇️ 🟡 | ~50 MB | `git clone https://github.com/google-deepmind/alphafold3` | Colab only (10 GB VRAM) | — |
| **AutoDock Vina** | ⬇️ 🔴 | pip | `pip install vina` | Local Phase 5 MVP | — |
| **RCSB PDB (target structures)** | ⬇️ 🟡 | ~2–50 MB/protein | [rcsb.org](https://www.rcsb.org) | For Vina: need .pdbqt format. See below | — |
| **ADMET-AI** | ⬇️ 🟡 | pip | `pip install admet-ai` | 42 ADMET endpoints from SMILES | — |
| **MGLTools (for Vina prep)** | ⬇️ 🔴 | ~300 MB | [ccsb.scripps.edu](https://ccsb.scripps.edu/mgltools/downloads/) | Prepares .pdb → .pdbqt for Vina docking | — |
| **Open Babel** | ⬇️ 🔴 | pip | `pip install openbabel-wheel` | SMILES ↔️ 3D mol conversion for Vina | — |

### LIHC Drug Target Structures to Download from PDB

These are the key targets to dock against. Download now and save to `data/targets/`:

| Target | Gene | PDB ID | Disease Role |
|---|---|---|---|
| VEGFR2 | KDR | **4ASD** | Angiogenesis — sorafenib/lenvatinib target |
| VEGFR2 (alt) | KDR | **3VHE** | Better resolved ligand-bound structure |
| EGFR | EGFR | **1IVO** | Overexpressed in 60% LIHC |
| c-RAF (BRAF) | RAF1 | **3PPJ** | MAPK pathway — sorafenib target |
| mTOR | MTOR | **4JT6** | PI3K-Akt-mTOR pathway in LIHC |
| TP53 | TP53 | **2OCJ** | Mutated in 25% LIHC — hotspot region |
| FGFR4 | FGFR4 | **5NUD** | Amplified in ~30% LIHC, lenvatinib target |

```bash
# Download PDB files (free, no login)
mkdir -p data/targets

for pdb_id in 4ASD 3VHE 1IVO 3PPJ 4JT6 2OCJ 5NUD; do
  wget "https://files.rcsb.org/download/${pdb_id}.pdb" \
       -O "data/targets/${pdb_id}.pdb"
done
```

---

## PHASE 6 — Personalized Medicine / Digital Twin

| Resource | Tag | Source | Notes |
|---|---|---|---|
| **ClinicalTrials.gov API** | 🟢 | [clinicaltrials.gov/api](https://clinicaltrials.gov/api/gui) | Free, no auth. Match patient to open LIHC trials |
| **LIHC Treatment Guidelines** | 🟢 | [NCCN](https://www.nccn.org/guidelines/category_1) / [ESMO](https://www.esmo.org/guidelines) | Free PDF. For LLM explanation context |
| **OncoTree** | ⬇️ 🟢 | [oncotree.mskcc.org](https://oncotree.mskcc.org/api/tumorTypes/tree) | Free JSON API. Tumor type hierarchy for patient profiling |

---

## MODEL WEIGHTS & PRE-TRAINED MODELS

| Model | Tag | Size | Source | Notes |
|---|---|---|---|---|
| **BioGPT** (local) | ⬇️ 🔴 | 350 MB | HuggingFace auto | `transformers` fetches automatically |
| **BioGPT-Large** (Colab) | ⬇️ 🟡 | 1.5 GB | HuggingFace auto | Colab only |
| **PubMedBERT** (CPU fallback) | ⬇️ 🟡 | 420 MB | HuggingFace auto | Works on CPU |
| **AlphaFold3 weights** | 📝 🟡 | 1.5 GB | DeepMind form | Registration required |
| **DeepDRA weights** | ✅ 🔴 | — | In `DeepDRA-main.zip` | Already have |

---

## VALIDATION DATASETS (Critical for Research Credibility)

These are NEEDED to prove your pipeline works — reviewers will ask about them.

| Resource | Tag | Source | Purpose |
|---|---|---|---|
| **COSMIC Cancer Gene Census** | 📝 🟡 | [cancer.sanger.ac.uk/census](https://cancer.sanger.ac.uk/census) | Validate Phase 2 biomarkers against known LIHC oncogenes |
| **Known LIHC drugs** | ✅ 🔴 | Literature | Sorafenib, Lenvatinib, Regorafenib, Cabozantinib MUST appear in repurposing top-20 |
| **TCGA survival data** | ✅ 🔴 | GDC (in clinical CSV) | Validate Phase 1 subtypes via Kaplan-Meier |
| **Human Protein Atlas** | ⬇️ 🟡 | [proteinatlas.org/download](https://www.proteinatlas.org/download/pathology.tsv.zip) | Cross-reference Phase 2 biomarkers against HPA liver cancer protein expression |
| **ImmPort** | 🟢 | [immport.org](https://www.immport.org) | Immune gene panels for LIHC immune subtype analysis |
| **BindingDB** | ⬇️ 🟡 | [bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp](https://www.bindingdb.org) | Experimental binding affinity data — validate Phase 5 Vina predictions |

---

## COMPLETE DOWNLOAD CHECKLIST (Ordered by Priority)

### 🚀 Start TODAY (Required for MVP)

- [ ] **TCGA-LIHC RNA-seq** — GDC portal (~200 MB)  
  `→ data/processed/tcga_lihc_rnaseq.csv`

- [ ] **TCGA-LIHC Somatic Mutation MAF** — GDC portal (~50 MB)  
  `→ data/processed/tcga_lihc_mutations.maf`

- [ ] **GDSC2 Drug Sensitivity** — cancerrxgene.org (~20 MB)  
  `→ data/external/gdsc2_drug_sensitivity.csv`

- [ ] **CCLE Expression** — depmap.org (~600 MB)  
  `→ data/external/ccle_expression.csv`

- [ ] **CCLE Model Info** — depmap.org (~5 MB, for name harmonization)  
  `→ data/external/ccle_model_info.csv`

- [ ] **DRKG** — GitHub/S3 (~100 MB compressed)  
  `→ data/external/drkg/drkg.tsv`

- [ ] **PDB target files** — RCSB (7 files, ~2–50 MB each)  
  `→ data/targets/4ASD.pdb, 3VHE.pdb, 1IVO.pdb, 3PPJ.pdb, 4JT6.pdb, 2OCJ.pdb, 5NUD.pdb`

### 📅 This Week (Required for Tier 2)

- [ ] **CCLE Somatic Mutations** — depmap.org (~250 MB)  
  `→ data/external/ccle_mutations.csv`

- [ ] **CCLE CNV** — depmap.org (~80 MB)  
  `→ data/external/ccle_cnv.csv`

- [ ] **COSMIC Cancer Gene Census** — cancer.sanger.ac.uk (register)  
  `→ data/external/cosmic_census.csv`

- [ ] **MSigDB KEGG gene sets** — gsea-msigdb.org (~2 MB)  
  `→ data/external/msigdb_kegg.gmt`

- [ ] **STRING PPI** — string-db.org (~1 GB)  
  `→ data/external/string_ppi.txt`

- [ ] **Human Protein Atlas** — proteinatlas.org (~5 MB)  
  `→ data/external/hpa_pathology.tsv`

- [ ] **AutoDock Vina** — `pip install vina`

- [ ] **MGLTools** — ccsb.scripps.edu (~300 MB installer)

- [ ] **Register for AlphaFold3 weights** — forms.gle/svvpY4u2jsHEwWYS6  
  (Wait 1–7 days for email with download link)

### 🔭 Before Publication (Validation + Stretch)

- [ ] **DisGeNET** — disgenet.com (register, free academic)  
  `→ data/external/disgenet_gene_disease.tsv`

- [ ] **BindingDB** — bindingdb.org (~2 GB, use subset)  
  `→ data/external/bindingdb_lihc_targets.tsv`

- [ ] **TCGA-LIHC Methylation** — GDC portal (~2 GB)  
  `→ data/processed/tcga_lihc_methylation.csv`

- [ ] **OncoKB API access** — oncokb.org/apiAccess (free academic)

---

## Total Download Size Estimate

| Priority | Size | Location |
|---|---|---|
| MVP TODAY | ~900 MB | local machine |
| Tier 2 THIS WEEK | ~1.8 GB | local machine |
| Colab (gctx, AF3) | ~44 GB | Google Drive (one-time upload) |
| Registration wait (AF3 weights) | ~1.5 GB | after DeepMind email |
| **Total local storage needed** | **~50 GB** | across machine + Drive |

> **Note**: LINCS modzs.gctx (42 GB) stays on Google Drive and is only accessed through Colab.
> All other data fits on a standard laptop with any SSD >50 GB free.
