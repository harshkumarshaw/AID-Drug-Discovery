# Data Loading Rules — Updated April 2026
## Critical rules for loading large biomedical datasets safely

---

## Rule 1: Always Validate Shape Before Proceeding

Every data load operation MUST check that the result is non-empty:

```python
df = pd.read_csv(path)
if df.shape[0] == 0:
    raise ValueError(f"[Data Loader] {path} loaded 0 rows. Check file integrity.")
if df.shape[1] == 0:
    raise ValueError(f"[Data Loader] {path} has 0 columns after loading.")
logger.info(f"Loaded {path.name}: {df.shape}")
```

---

## Rule 2: LINCS Files — DO NOT Concat All TSVs Together ⚠️ BUG 001

The four consensus TSV files have different perturbagen sets.
`pd.concat([drugbank, knockdown, overexpression, pert_id], axis=1).dropna()`
produces **0 rows** (empty intersection).

**Correct approach:**
```python
# Use each file separately for its specific purpose:
from config import LINCS_DRUGBANK_TSV, LINCS_PERT_TSV, LINCS_GCTX
from cmapPy.pandasGEXpr import parse

# For drug response: use drugbank signatures only
drugbank_sigs = pd.read_csv(LINCS_DRUGBANK_TSV, sep='\t', index_col=0)

# For full 978-gene signatures: use modzs.gctx via cmapPy
# Load in chunks to avoid OOM with 42 GB file
gctoo = parse(str(LINCS_GCTX), rid=gene_subset, cid=pert_subset)
df = gctoo.data_df
```

---

## Rule 3: ChEMBL SQLite — Must Extract Archive First ⚠️ BUG 002

```python
# WRONG — will raise "not a database" error:
# conn = sqlite3.connect("chembl_35_sqlite.tar.gz")

# CORRECT:
import tarfile
from config import CHEMBL_SQLITE_ARCHIVE, CHEMBL_DB_DIR, CHEMBL_DB_PATH

if not CHEMBL_DB_PATH.exists():
    with tarfile.open(CHEMBL_SQLITE_ARCHIVE, "r:gz") as tar:
        tar.extractall(CHEMBL_DB_DIR)

conn = sqlite3.connect(CHEMBL_DB_PATH)
```

---

## Rule 4: Large Files Must Use Chunked Loading

Any file > 500 MB must be loaded in chunks:

```python
# BAD — OOM for large files:
# df = pd.read_csv(EMTAB_VARIANT_TSV)

# GOOD — chunked loading:
from config import EMTAB_VARIANT_TSV, CHUNK_SIZE

chunks = []
for chunk in pd.read_csv(EMTAB_VARIANT_TSV, sep='\t', chunksize=CHUNK_SIZE):
    processed = chunk[chunk['FILTER'] == 'PASS']
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)
```

---

## Rule 5: TCGA GroupBy Must Exclude UUID Columns ⚠️ BUG 003

```python
# BAD — fails on UUID string columns:
# cluster_means = data.groupby('Cluster').mean()

# GOOD:
import numpy as np
numeric_data = data.select_dtypes(include=[np.number])
cluster_means = numeric_data.groupby(data['Cluster']).mean()
```

---

## Rule 6: AlphaFold3 Target File Format (April 2026)

AlphaFold3 requires a JSON input file per protein-ligand pair:

```python
import json
from config import AF3_MODEL_DIR

def prepare_af3_input(protein_seq: str, ligand_smiles: str, job_name: str) -> dict:
    """Prepare AlphaFold3 input JSON for protein-ligand co-folding."""
    return {
        "name": job_name,
        "sequences": [
            {"protein": {"id": "A", "sequence": protein_seq}},
            {"ligand": {"id": "L", "smiles": ligand_smiles}}
        ],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1
    }
# Write to input JSON → run AF3 → parse output JSON for pTM/ipTM scores
```

---

## Rule 7: DRKG Knowledge Graph Loading (April 2026)

```python
from config import DRKG_TRIPLES_TSV
import pandas as pd

# Load DRKG triples (Drug → Gene → Disease → Pathway)
drkg_df = pd.read_csv(DRKG_TRIPLES_TSV, sep='\t', header=None,
                       names=['head', 'relation', 'tail'])
# Filter to biomedically relevant relation types
drug_gene = drkg_df[drkg_df['relation'].str.contains('DRUGBANK::ddi-interactor-in')]
gene_disease = drkg_df[drkg_df['relation'].str.contains('DISEASES::')]
```

---

## Rule 8: All Paths Must Come From config.py

```python
# BAD — hardcoded paths will break on any other machine:
# df = pd.read_csv("D:\\Research Study\\AID\\Code\\data.csv")

# GOOD:
from config import TCGA_CLINICAL_CSV
df = pd.read_csv(TCGA_CLINICAL_CSV)
```

---

## Rule 9: BioGPT / LLM Inference — Use GPU If Available

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_MODEL_NAME
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME).to(device)

# Limit context length to avoid OOM
MAX_PATH_TOKENS = 512
```

---

## Rule 10: GDSC2 / CCLE Loading

```python
from config import GDSC2_CSV, CCLE_EXPRESSION_CSV

# GDSC2 — IC50 values (LN-transformed preferred)
gdsc = pd.read_csv(GDSC2_CSV)
gdsc = gdsc[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'AUC']].dropna()

# CCLE expression — gene x cell line matrix (already in LOGE TPM+1)
ccle_expr = pd.read_csv(CCLE_EXPRESSION_CSV, index_col=0)
# Match CCLE cell line names to GDSC cell line names (name harmonization required)
```
