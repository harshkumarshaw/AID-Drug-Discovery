"""
colab/colab_phase3_full_mmdrp.py
=================================
Full MMDRP-style drug response model + modzs.gctx loading.
Run this on Colab (T4 15GB) or Kaggle (P100 16GB).

WHY COLAB:
  1. modzs.gctx is 42 GB — impossible to load on 16 GB RAM
  2. Full MMDRP (3 AE branches, batch=128) needs ~8 GB VRAM

LOCAL ALTERNATIVE: Run pipeline/drug_response.py locally
  (MMDRP-lite: 2 AE branches, top 2000 genes, batch=32, AMP)

SETUP:
1. Upload modzs.gctx to Google Drive (one-time, takes 6–8 hours on stable internet)
   OR use Kaggle datasets for LINCS (already available as public dataset)
2. Upload your local GDSC2 + CCLE CSVs to Drive
3. Run cells sequentially
"""

# === CELL 1: Setup & Drive Mount ===

import os, sys, subprocess, json

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/AID_Pipeline'
DATA_DIR = f'{DRIVE_ROOT}/data'
EXTERNAL_DIR = f'{DATA_DIR}/external'
PROCESSED_DIR = f'{DATA_DIR}/processed'
MODELS_DIR = f'{DRIVE_ROOT}/models/drug_response'

for d in [DATA_DIR, EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'cmapPy', 'rdkit', 'torch', 'scikit-learn', 'shap', '--quiet'
], check=True)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# === CELL 2: Load GDSC2 + CCLE ===

import pandas as pd
import numpy as np

# GDSC2 drug sensitivity
GDSC2_PATH = f'{EXTERNAL_DIR}/gdsc2_drug_sensitivity.csv'
gdsc2 = pd.read_csv(GDSC2_PATH)
gdsc2 = gdsc2[['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50', 'AUC', 'TCGA_DESC']].dropna()

# Filter to liver cancer cell lines for LIHC-specific analysis
lihc_cell_lines = gdsc2[gdsc2['TCGA_DESC'] == 'LIHC']['CELL_LINE_NAME'].unique()
print(f"GDSC2 loaded: {len(gdsc2)} rows, {len(lihc_cell_lines)} LIHC cell lines")
print(f"Example LIHC cell lines: {lihc_cell_lines[:5]}")

# CCLE expression (genes × cell lines)
CCLE_EXPR_PATH = f'{EXTERNAL_DIR}/ccle_expression.csv'
ccle_expr = pd.read_csv(CCLE_EXPR_PATH, index_col=0)

# Match CCLE to GDSC2 cell line names (name harmonization)
# CCLE uses format: "CellLineName_TissueType"
ccle_names = ccle_expr.index.str.split('_').str[0].str.upper()
ccle_expr.index = ccle_names
gdsc2['cell_key'] = gdsc2['CELL_LINE_NAME'].str.upper().str.replace('-', '').str.replace(' ', '')

# Inner join: only cell lines present in both datasets
common_lines = set(ccle_expr.index) & set(gdsc2['cell_key'].unique())
print(f"Common cell lines (CCLE ∩ GDSC2): {len(common_lines)}")

ccle_expr = ccle_expr[ccle_expr.index.isin(common_lines)]
gdsc2_matched = gdsc2[gdsc2['cell_key'].isin(common_lines)]


# === CELL 3: Load LINCS modzs.gctx (Full Dataset) ===
# This is the key step that requires Colab — 42 GB file

GCTX_PATH = f'{DRIVE_ROOT}/LINCS/modzs.gctx'

if not os.path.exists(GCTX_PATH):
    print("ERROR: modzs.gctx not found on Drive.")
    print(f"Expected at: {GCTX_PATH}")
    print("Upload from local machine: copy LINCS/modzs.gctx to MyDrive/AID_Pipeline/LINCS/")
    print("Using LINCS drugbank TSV as fallback...")

    LINCS_TSV_PATH = f'{DATA_DIR}/raw/consensi-drugbank.tsv'
    if os.path.exists(LINCS_TSV_PATH):
        lincs_sigs = pd.read_csv(LINCS_TSV_PATH, sep='\t', index_col=0)
    else:
        print("Also no TSV found. Run locally with LINCS TSV first.")
        lincs_sigs = None
else:
    # Load gctx with cmapPy — select only LIHC-relevant cell line perturbations
    from cmapPy.pandasGEXpr import parse

    print(f"Loading modzs.gctx from {GCTX_PATH}...")
    print("This takes 3–10 minutes for full load...")

    # Load subset: only drugs present in GDSC2 (reduces from 1.3M to ~300K perts)
    gdsc2_drugs = gdsc2_matched['DRUG_NAME'].str.lower().unique()

    # Parse metadata first to identify relevant perturbation columns
    gctoo = parse(str(GCTX_PATH))
    col_meta = gctoo.col_metadata_df

    # Filter to drug perturbations matching GDSC2 drugs
    drug_mask = col_meta['pert_iname'].str.lower().isin(gdsc2_drugs)
    relevant_cids = col_meta[drug_mask].index.tolist()
    print(f"Relevant LINCS perturbations: {len(relevant_cids)} of {len(col_meta)}")

    # Reload with only relevant columns (much faster, less memory)
    if len(relevant_cids) > 0:
        gctoo_filtered = parse(str(GCTX_PATH), cid=relevant_cids[:5000])  # Cap for safety
        lincs_sigs = gctoo_filtered.data_df  # Genes × perturbations
        print(f"LINCS subset loaded: {lincs_sigs.shape}")
    else:
        print("No matching perts found — using TSV fallback")
        lincs_sigs = None


# === CELL 4: Feature Engineering ===

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler

# Top variable genes (reduce dimensionality)
N_GENES = 5000  # More genes than MVP, fewer than all 60K
gene_var = ccle_expr.var(axis=0).sort_values(ascending=False)
top_genes = gene_var.head(N_GENES).index
ccle_expr_top = ccle_expr[top_genes]

# Drug Morgan fingerprints
def get_morgan_fp(smiles: str, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits))

# Get SMILES from ChEMBL (or use LINCS pert_smiles column if available)
# For now, query PubChem for SMILES by drug name
import requests

def get_smiles_from_pubchem(drug_name: str) -> str:
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON'
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception:
        pass
    return None

# Build drug fingerprint matrix
print("Building drug fingerprint matrix...")
drug_smiles = {}
for drug in gdsc2_matched['DRUG_NAME'].unique():
    smi = get_smiles_from_pubchem(drug)
    if smi:
        drug_smiles[drug] = smi

drug_fps = {name: get_morgan_fp(smi) for name, smi in drug_smiles.items()}
print(f"Fingerprints computed for {len(drug_fps)} of {gdsc2_matched['DRUG_NAME'].nunique()} drugs")


# === CELL 5: Build Dataset (cell line features + drug features → IC50) ===

X_rows, y_rows = [], []

for _, row in gdsc2_matched.iterrows():
    cell = row['cell_key']
    drug = row['DRUG_NAME']
    ic50 = row['LN_IC50']

    # Cell line features
    if cell not in ccle_expr_top.index:
        continue
    cell_feat = ccle_expr_top.loc[cell].values

    # Drug features: fingerprint (+ LINCS signature if available)
    if drug not in drug_fps:
        continue
    drug_feat = drug_fps[drug]

    if lincs_sigs is not None:
        # Try to match LINCS signature for this drug
        drug_col = lincs_sigs.columns[lincs_sigs.columns.str.lower().str.contains(
            drug.lower().replace(' ', '')[:6])]
        if len(drug_col) > 0:
            lincs_feat = lincs_sigs[drug_col[0]].values[:978]
            drug_feat = np.concatenate([drug_feat, lincs_feat])
        else:
            drug_feat = np.concatenate([drug_feat, np.zeros(978)])
    else:
        drug_feat = np.concatenate([drug_feat, np.zeros(978)])

    X_rows.append(np.concatenate([cell_feat, drug_feat]))
    y_rows.append(ic50)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.float32)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")


# === CELL 6: Full MMDRP-Style Model ===

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

device = torch.device('cuda')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                      random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.15/0.85,
                                                    random_state=42)

# Scale
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# DataLoaders (batch=128 — more memory on T4)
train_dl = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
    batch_size=128, shuffle=True
)
val_dl = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1)),
    batch_size=256  # Larger eval batch
)

# MMDRP-style model (simplified but significantly larger than local lite version)
n_cell = len(ccle_expr_top.columns)  # Cell line expression features
n_drug = 2048 + 978                   # Morgan FP + LINCS signature

class MMDRPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Cell line encoder
        self.cell_enc = nn.Sequential(
            nn.Linear(n_cell, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        # Drug encoder
        self.drug_enc = nn.Sequential(
            nn.Linear(n_drug, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        # LMF fusion → MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x_cell = x[:, :n_cell]
        x_drug = x[:, n_cell:]
        z_cell = self.cell_enc(x_cell)
        z_drug = self.drug_enc(x_drug)
        z = torch.cat([z_cell, z_drug], dim=1)
        return self.predictor(z)

model = MMDRPModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()
scaler_amp = GradScaler()

# Training with AMP
best_val_loss = float('inf')
patience = 15
patience_ctr = 0

for epoch in range(100):
    model.train()
    train_losses = []
    for X_b, y_b in train_dl:
        X_b, y_b = X_b.to(device), y_b.to(device)
        with autocast():
            pred = model(X_b)
            loss = criterion(pred, y_b)
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        optimizer.zero_grad()
        train_losses.append(loss.item())
    scheduler.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad(), autocast():
        for X_b, y_b in val_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_losses.append(criterion(model(X_b), y_b).item())
    val_loss = np.mean(val_losses)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train RMSE: {np.sqrt(np.mean(train_losses)):.4f} "
              f"| Val RMSE: {np.sqrt(val_loss):.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), f'{MODELS_DIR}/mmdrp_full_best.pt')
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nBest Val RMSE: {np.sqrt(best_val_loss):.4f}")


# === CELL 7: Evaluate + SHAP + Validation Anchor ===

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import shap

model.load_state_dict(torch.load(f'{MODELS_DIR}/mmdrp_full_best.pt'))
model.eval()

X_test_t = torch.tensor(X_test).to(device)
with torch.no_grad(), autocast():
    y_pred = model(X_test_t).cpu().numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_r = pearsonr(y_test, y_pred)[0]
print(f"\n=== Test Set Evaluation ===")
print(f"RMSE: {rmse:.4f} (target: < 1.0)")
print(f"Pearson r: {pearson_r:.4f} (target: > 0.5)")

# Validation anchor: sorafenib on LIHC cell lines
# Sorafenib should have low IC50 (highly responsive) for HCC cell lines
sorafenib_rows = gdsc2_matched[
    (gdsc2_matched['DRUG_NAME'].str.lower() == 'sorafenib') &
    (gdsc2_matched['TCGA_DESC'] == 'LIHC')
]
print(f"\n=== Validation Anchor: Sorafenib on LIHC cell lines ===")
print(f"N LIHC cell lines: {len(sorafenib_rows)}")
if len(sorafenib_rows) > 0:
    q25 = gdsc2_matched['LN_IC50'].quantile(0.25)
    soraf_mean = sorafenib_rows['LN_IC50'].mean()
    print(f"Global 25th percentile LN_IC50: {q25:.3f}")
    print(f"Sorafenib LIHC mean LN_IC50: {soraf_mean:.3f}")
    print(f"Anchor PASS: {soraf_mean < q25}")


# === CELL 8: Save Predictions ===

# Generate predictions for all GDSC2 pairs with features
pred_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred,
})
pred_df.to_csv(f'{PROCESSED_DIR}/drug_response_predictions_full.csv', index=False)
print(f"Saved predictions: {PROCESSED_DIR}/drug_response_predictions_full.csv")

# Save scaler for use in Phase 6
import pickle
with open(f'{MODELS_DIR}/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nPhase 3 (Full MMDRP) complete. Download from Drive:")
print(f"  {MODELS_DIR}/mmdrp_full_best.pt")
print(f"  {PROCESSED_DIR}/drug_response_predictions_full.csv")
