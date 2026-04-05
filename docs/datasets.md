# Dataset Schemas & Locations
## Updated: April 2026

All paths are relative to `config.py` constants. NEVER hardcode absolute paths.

---

## 1. TCGA-LIHC (Primary Disease Dataset)

**Location**: `TCGA_DIR = ROOT_DIR / "TCGA"`

| File | Config Key | Description |
|---|---|---|
| `cleaned_clinical_data.csv` | `TCGA_CLINICAL_CSV` | Patient age, gender, vital_status, tumor_stage |
| `cleaned_biospecimen_data.csv` | `TCGA_BIOSPECIMEN_CSV` | Sample IDs, tissue type, preservation method |
| `tcga_lihc_rnaseq.csv` | `TCGA_RNASEQ_CSV` | RNA-seq counts matrix (genes × samples) — **download needed** |
| `tcga_lihc_cnv.csv` | `TCGA_CNV_CSV` | Copy number variation segments — **download needed** |

**Schema — cleaned_clinical_data.csv**:
```
case_id (UUID), age_at_diagnosis, gender, vital_status, tumor_stage,
days_to_death, days_to_last_follow_up, primary_diagnosis
```

**Derived column** (Bug 004 fix):
```python
df['Disease_Status'] = df['vital_status'].map({'Alive': 0, 'Dead': 1})
```

**Download instructions** (TCGA-LIHC RNA-seq + CNV):
```
1. Go to https://portal.gdc.cancer.gov/
2. Filter: Project = TCGA-LIHC, Data Category = Transcriptome Profiling
3. Download HTSeq-Counts or STAR counts (RNA-seq) and Gene Level Copy Number (CNV)
4. Place at: data/processed/tcga_lihc_rnaseq.csv, data/processed/tcga_lihc_cnv.csv
```

---

## 2. E-MTAB-5902 (Genomic Variants — WGS)

**Location**: `EMTAB_DIR = ROOT_DIR / "E-MTAB-5902"`

| File | Config Key | Description |
|---|---|---|
| `var-GS000001153-ASM.tsv` | `EMTAB_VARIANT_TSV` | Whole-genome variant calls (2.5 GB) |
| `E-MTAB-5902.sdrf.txt` | `EMTAB_METADATA_TSV` | Sample metadata |

**Schema — variant TSV**:
```
chromosome, position, ref_allele, alt_allele, FILTER, INFO, FORMAT, SAMPLE
```

**⚠️ Warning**: Single patient sample (EM0231_T1_primary). For biomarker discovery,
supplement with additional EBI samples OR use TCGA-LIHC RNA-seq (preferred).

**Loading**: MUST use chunked loading (`chunksize=100_000`) — file is 2.5 GB.
Filter `FILTER == 'PASS'` to keep high-quality variants only.

**Derived features** (input to Phase 2):
```python
variant_counts_per_chrom = df.groupby('chromosome')['alt_allele'].count()
# Use SNV + indel + CNV counts per chromosome as feature vector
```

---

## 3. LINCS L1000

**Location**: `LINCS_DIR = ROOT_DIR / "LINCS"`

| File | Config Key | Description | Size |
|---|---|---|---|
| `modzs.gctx` | `LINCS_GCTX` | Full L1000 moderated z-scores (978 genes × all perts) | 42 GB |
| `consensi-drugbank.tsv` | `LINCS_DRUGBANK_TSV` | DrugBank compound signatures | ~50 MB |
| `consensi-knockdown.tsv` | `LINCS_KNOCKDOWN_TSV` | Gene knockdown signatures | ~50 MB |
| `consensi-overexpression.tsv` | `LINCS_OVEREXPRESSION_TSV` | Gene overexpression signatures | ~50 MB |
| `consensi-pert_id.tsv` | `LINCS_PERT_TSV` | Perturbation ID-level signatures | ~50 MB |

**⚠️ Bug 001 warning**: DO NOT `pd.concat(..., axis=1).dropna()` — produces 0 rows.
Use each file individually for its specific purpose.

**Usage in Phase 3**:
- Drug features: LINCS drugbank z-scores (978-gene perturbation signature per drug)
- Labels: GDSC2 LN_IC50 (matched by drug name → ChEMBL ID)
- modzs.gctx: for full-gene-space signatures, load via `cmapPy`

---

## 4. ChEMBL-35

**Location**: `DRUG_REPURPOSING_DIR = ROOT_DIR / "Drug Repurposing"`

| File | Config Key | Description | Size |
|---|---|---|---|
| `chembl_35.sdf` | `CHEMBL_SDF` | 2.4M drug structures in SDF format | Large |
| `chembl_35_sqlite.tar.gz` | `CHEMBL_SQLITE_ARCHIVE` | SQLite database (compressed) | ~3 GB |
| `chembl_35.h5` | `CHEMBL_H5` | HDF5 format of compound data | Large |
| `chembl_db/chembl_35.db` | `CHEMBL_DB_PATH` | Extracted SQLite DB | After extraction |
| `DeepDRA-main.zip` | `DEEPDRA_ZIP` | DeepDRA model source code | — |

**⚠️ Bug 002 warning**: Must extract archive before connecting:
```python
# See .claude/rules/data_loading.md Rule 3
```

**Key ChEMBL tables** (after SQLite extraction):
```sql
-- Drug target interactions:
SELECT cs.canonical_smiles, td.pref_name AS target, act.standard_value AS ic50
FROM compound_structures cs
JOIN activities act ON cs.molregno = act.molregno
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
WHERE act.standard_type = 'IC50'
```

---

## 5. GDSC2 + CCLE (Drug Response Labels) — April 2026

**Location**: `EXTERNAL_DIR = ROOT_DIR / "data" / "external"`

| File | Config Key | Description | Download |
|---|---|---|---|
| `gdsc2_drug_sensitivity.csv` | `GDSC2_CSV` | Cell line IC50 data (LN_IC50, AUC) | cancerrxgene.org |
| `ccle_expression.csv` | `CCLE_EXPRESSION_CSV` | DepMap RNA-seq (TPM log) | depmap.org |
| `ccle_mutations.csv` | `CCLE_MUTATION_CSV` | DepMap mutation matrix | depmap.org |
| `ccle_cnv.csv` | `CCLE_CNV_CSV` | DepMap CNV segments | depmap.org |

**GDSC2 schema**:
```
CELL_LINE_NAME, TCGA_DESC, DRUG_NAME, DRUG_ID, PUTATIVE_TARGET,
LN_IC50, AUC, RMSE, DATASET (GDSC2)
```

**Download commands**:
```bash
# GDSC2 (from cancerrxgene.org/downloads/bulk_download):
# Download: GDSC2_fitted_dose_response_27Oct23.xlsx → convert to CSV

# CCLE (from depmap.org/portal/download/all/ — 24Q4 release):
wget https://depmap.org/portal/api/downloads/files?fileType=csv&fileName=OmicsExpressionProteinCodingGenesTPMLogp1.csv
wget https://depmap.org/portal/api/downloads/files?fileType=csv&fileName=OmicsSomaticMutationsMatrixDamaging.csv
```

---

## 6. DRKG (Drug Repurposing Knowledge Graph) — April 2026

**Location**: `DRKG_DIR = ROOT_DIR / "data" / "external" / "drkg"`

| File | Config Key | Description |
|---|---|---|
| `drkg.tsv` | `DRKG_TRIPLES_TSV` | 5.8M triples: head, relation, tail |

**Download**:
```bash
# From GitHub: https://github.com/gnn4dr/DRKG
# Direct: https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tsv.gz
wget https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tsv.gz
gunzip drkg.tsv.gz
mkdir -p data/external/drkg && mv drkg.tsv data/external/drkg/
```

**Schema**: `head\trelation\ttail` (tab-separated, 5.87M rows)

**Key relation types**:
- `DRUGBANK::ddi-interactor-in::Compound:Disease` — drug-disease links
- `GNBR::T::Compound:Gene` — drug-target links (from text mining)
- `STRING::IPTM+LV::Gene:Gene` — protein-protein interactions

---

## 7. AlphaFold3 Weights (Phase 5 — April 2026)

**Location**: `AF3_MODEL_DIR = ROOT_DIR / "data" / "external" / "alphafold3_weights"`

**Source**: `github.com/google-deepmind/alphafold3`

**Download**: Requires registration at `https://forms.gle/svvpY4u2jsHEwWYS6`

**Installation**:
```bash
git clone https://github.com/google-deepmind/alphafold3
cd alphafold3
pip install -e .
# After receiving weights (email from DeepMind), place at:
# data/external/alphafold3_weights/
```

**Minimum requirements**: NVIDIA GPU with ≥10 GB VRAM (V100/A100 preferred).
CPU fallback available but very slow (~hours per structure).

---

## 8. Data Summary Table

| Dataset | Phase | Records | Key Features | Labels |
|---|---|---|---|---|
| TCGA-LIHC clinical | 1 | ~371 patients | Age, stage, vital_status | Cluster (derived) |
| TCGA-LIHC RNA-seq | 1, 2 | ~371 × 60,660 genes | Log-normalized counts | Subtype |
| TCGA-LIHC CNV | 1 | ~371 samples | Chromosome segment copy numbers | Subtype |
| E-MTAB-5902 | 2 | 1 sample (~50M SNVs) | Per-chrom variant counts | Biomarkers |
| LINCS modzs.gctx | 3 | ~978K perturbation profiles | 978-gene z-scores | — |
| GDSC2 IC50 | 3 | ~310K responses | Cell line + drug | LN_IC50 |
| CCLE (3 tables) | 3 | ~1800 cell lines | Expression, mutation, CNV | — |
| ChEMBL-35 | 4, 5 | 2.4M compounds | SMILES, bioactivity, targets | — |
| DRKG | 4 | 5.87M triples | Drug-Gene-Disease edges | — |
| AF3 weights | 5 | — | Protein-ligand co-folding | Binding score |
