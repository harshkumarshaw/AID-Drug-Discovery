"""
colab/colab_phase5_af3.py
=========================
AlphaFold3 protein-ligand co-folding for Phase 5 molecular simulation.
Run this file on Google Colab (T4 15GB or A100) or Kaggle (P100 16GB).

WHY COLAB: AF3 requires ≥10 GB VRAM. RTX 3050 (4GB) cannot run AF3.
Local alternative: AutoDock Vina (run pipeline/molecular_simulation.py locally).

SETUP INSTRUCTIONS:
1. Open Google Colab (colab.research.google.com)
2. Runtime → Change runtime type → T4 GPU (free) or A100 (Colab Pro)
3. Mount Google Drive:
   from google.colab import drive; drive.mount('/content/drive')
4. Copy this entire file content into a Colab code cell, OR upload as .py and run:
   !python colab_phase5_af3.py
5. Outputs are saved to DRIVE_ROOT/data/processed/interaction_scores.csv

COLAB CELL STRUCTURE: Each section marked with "# === CELL N ===" is a
separate Colab code cell for interactive execution.
"""

# === CELL 1: Setup & Drive Mount ===
# Run this cell first

import os, json, subprocess, sys

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ─── Configure paths (edit DRIVE_ROOT to match your Drive) ───────────────────
DRIVE_ROOT = '/content/drive/MyDrive/AID_Pipeline'
DATA_DIR = f'{DRIVE_ROOT}/data'
MODELS_DIR = f'{DRIVE_ROOT}/models'
PROCESSED_DIR = f'{DATA_DIR}/processed'

# Create directories
for d in [DATA_DIR, MODELS_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

print("Drive mounted. Directories created.")
print(f"Root: {DRIVE_ROOT}")


# === CELL 2: Install AlphaFold3 and Dependencies ===

subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'rdkit', 'admet-ai', 'biopython', 'requests', '--quiet'
], check=True)

# Install AlphaFold3 from GitHub (open-source since January 2026)
if not os.path.exists('/content/alphafold3'):
    subprocess.run([
        'git', 'clone', '--depth', '1',
        'https://github.com/google-deepmind/alphafold3',
        '/content/alphafold3'
    ], check=True)

os.chdir('/content/alphafold3')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
               check=True)

print("AlphaFold3 installed.")

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# === CELL 3: Load Repurposing Candidates ===
# Load the top-50 candidates from Phase 4 local run

import pandas as pd
import numpy as np

# Load candidates from Drive (uploaded from local Phase 4 run)
REPURPOSING_CSV = f'{PROCESSED_DIR}/repurposing_candidates.csv'

if os.path.exists(REPURPOSING_CSV):
    candidates = pd.read_csv(REPURPOSING_CSV)
    print(f"Loaded {len(candidates)} repurposing candidates from Phase 4")
    print(candidates[['drug_name', 'chembl_id', 'combined_score']].head(10))
else:
    # Fallback: use hardcoded LIHC validation anchors for smoke test
    print("WARNING: repurposing_candidates.csv not found. Using validation anchors.")
    candidates = pd.DataFrame({
        'drug_name': ['sorafenib', 'lenvatinib', 'regorafenib', 'cabozantinib'],
        'chembl_id': ['CHEMBL1336', 'CHEMBL3307956', 'CHEMBL1946170', 'CHEMBL2105735'],
        'smiles': [
            'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1',   # sorafenib
            'C=CC(=O)Nc1ccc(Oc2cc3c(cc2C)C(=O)NC3=O)cc1',                   # lenvatinib
            'CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2F)ccn1',  # regorafenib
            'COc1cc2nccc(Oc3ccc(NC(=O)c4cc(F)cc(F)c4)cc3)c2cc1OC',         # cabozantinib
        ],
        'combined_score': [0.90, 0.85, 0.78, 0.72],
    })

# Select top-N for simulation (Colab session limit: ~6 pairs/hour with AF3)
TOP_N_FOR_SIMULATION = 10
top_candidates = candidates.head(TOP_N_FOR_SIMULATION)
print(f"\nSimulating top {len(top_candidates)} candidates")


# === CELL 4: Define LIHC Target Proteins ===
# Key LIHC targets from Phase 2 biomarker phase + literature

LIHC_TARGETS = {
    'VEGFR2': {
        'uniprot': 'P35968',
        'description': 'Vascular endothelial growth factor receptor 2',
        'relevance': 'Angiogenesis driver in LIHC, target of sorafenib/lenvatinib',
    },
    'EGFR': {
        'uniprot': 'P00533',
        'description': 'Epidermal growth factor receptor',
        'relevance': 'Overexpressed in ~60% of LIHC tumors',
    },
    'TP53': {
        'uniprot': 'P04637',
        'description': 'Tumor protein p53',
        'relevance': 'Mutated in ~25% of LIHC; loss of function',
    },
}

def fetch_protein_sequence(uniprot_id: str) -> str:
    """Fetch protein sequence from UniProt API."""
    import requests
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta'
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        lines = resp.text.strip().split('\n')
        seq = ''.join(lines[1:])  # Skip header line
        print(f"  Fetched {uniprot_id}: {len(seq)} aa")
        return seq
    else:
        raise ValueError(f"UniProt fetch failed for {uniprot_id}: {resp.status_code}")

print("Fetching LIHC target protein sequences from UniProt...")
for name, info in LIHC_TARGETS.items():
    info['sequence'] = fetch_protein_sequence(info['uniprot'])
    print(f"  {name} ({info['uniprot']}): {len(info['sequence'])} amino acids")


# === CELL 5: Prepare AlphaFold3 Input JSONs ===

import os

def prepare_af3_input(protein_name: str, protein_seq: str,
                       drug_name: str, smiles: str) -> dict:
    """
    Prepare AlphaFold3 input JSON for protein-ligand co-folding.
    AF3 format: protein sequence + ligand SMILES in same prediction job.
    """
    return {
        "name": f"{protein_name}__{drug_name}",
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_seq,
                }
            },
            {
                "ligand": {
                    "id": "L",
                    "smiles": smiles,
                }
            }
        ],
        "modelSeeds": [42],
        "dialect": "alphafold3",
        "version": 1,
    }

# Create AF3 input JSONs for all drug-target pairs
AF3_INPUT_DIR = f'{DRIVE_ROOT}/af3_inputs'
AF3_OUTPUT_DIR = f'{DRIVE_ROOT}/af3_outputs'
os.makedirs(AF3_INPUT_DIR, exist_ok=True)
os.makedirs(AF3_OUTPUT_DIR, exist_ok=True)

pairs = []
for _, drug_row in top_candidates.iterrows():
    for target_name, target_info in LIHC_TARGETS.items():
        if 'smiles' not in drug_row or pd.isna(drug_row.get('smiles')):
            continue

        input_json = prepare_af3_input(
            protein_name=target_name,
            protein_seq=target_info['sequence'],
            drug_name=drug_row['drug_name'],
            smiles=drug_row['smiles'],
        )
        pair_name = f"{target_name}__{drug_row['drug_name']}"
        input_path = f'{AF3_INPUT_DIR}/{pair_name}.json'

        with open(input_path, 'w') as f:
            json.dump(input_json, f, indent=2)

        pairs.append({
            'drug_name': drug_row['drug_name'],
            'target': target_name,
            'input_json': input_path,
            'output_dir': f'{AF3_OUTPUT_DIR}/{pair_name}',
        })

print(f"Prepared {len(pairs)} protein-ligand pairs for AF3")


# === CELL 6: Run AlphaFold3 Co-Folding ===
# NOTE: AF3 weights must be downloaded separately and placed at AF3_WEIGHTS_DIR
# Weights: https://forms.gle/svvpY4u2jsHEwWYS6 (email from DeepMind)

AF3_WEIGHTS_DIR = f'{DRIVE_ROOT}/alphafold3_weights'  # Must have weights here

results = []

for pair in pairs:
    os.makedirs(pair['output_dir'], exist_ok=True)
    pair_name = f"{pair['target']}__{pair['drug_name']}"
    print(f"\nRunning AF3: {pair_name}...")

    try:
        cmd = [
            sys.executable, '/content/alphafold3/run_alphafold.py',
            f'--json_path={pair["input_json"]}',
            f'--model_dir={AF3_WEIGHTS_DIR}',
            f'--output_dir={pair["output_dir"]}',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Parse AF3 output JSON for confidence metrics
        output_jsons = [f for f in os.listdir(pair['output_dir'])
                        if f.endswith('_summary_confidences.json')]

        if output_jsons:
            with open(f'{pair["output_dir"]}/{output_jsons[0]}') as f:
                confidence = json.load(f)

            ptm = confidence.get('ptm', 0)
            iptm = confidence.get('iptm', 0)
            binding_pass = iptm >= 0.5

            results.append({
                'drug_name': pair['drug_name'],
                'target': pair['target'],
                'ptm': ptm,
                'iptm': iptm,
                'binding_score': iptm,
                'binding_pass': binding_pass,
                'status': 'success',
            })
            print(f"  pTM={ptm:.3f}, ipTM={iptm:.3f} → {'PASS' if binding_pass else 'FAIL'}")
        else:
            print(f"  WARNING: No output JSON found — AF3 may have failed")
            results.append({'drug_name': pair['drug_name'], 'target': pair['target'],
                           'status': 'failed', 'binding_pass': False})

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {pair_name}")
        results.append({'drug_name': pair['drug_name'], 'target': pair['target'],
                       'status': 'timeout', 'binding_pass': False})
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({'drug_name': pair['drug_name'], 'target': pair['target'],
                       'status': 'error', 'error': str(e), 'binding_pass': False})


# === CELL 7: ADMET-AI Filtering ===

try:
    from admet_ai import ADMETModel

    admet_model = ADMETModel()
    admet_results = []

    for _, drug_row in top_candidates.iterrows():
        if 'smiles' not in drug_row or pd.isna(drug_row.get('smiles')):
            continue
        preds = admet_model.predict(drug_row['smiles'])
        admet_results.append({
            'drug_name': drug_row['drug_name'],
            'admet_lipinski': preds.get('Lipinski', True),
            'admet_herg': preds.get('hERG', False),
            'admet_ames': preds.get('AMES', False),
            'admet_pass': (preds.get('Lipinski', True) and
                          not preds.get('hERG', False) and
                          not preds.get('AMES', False)),
        })
    admet_df = pd.DataFrame(admet_results)
    print(f"ADMET results: {admet_df['admet_pass'].sum()}/{len(admet_df)} drugs pass")

except ImportError:
    print("ADMET-AI not installed. Using Lipinski filter only.")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    admet_results = []
    for _, drug_row in top_candidates.iterrows():
        if 'smiles' not in drug_row or pd.isna(drug_row.get('smiles')):
            continue
        mol = Chem.MolFromSmiles(drug_row['smiles'])
        if mol:
            lip_pass = (Descriptors.MolWt(mol) <= 550 and
                        Descriptors.MolLogP(mol) <= 6 and
                        Descriptors.NumHDonors(mol) <= 5)
        else:
            lip_pass = False
        admet_results.append({'drug_name': drug_row['drug_name'],
                               'admet_lipinski': lip_pass, 'admet_pass': lip_pass})
    admet_df = pd.DataFrame(admet_results)


# === CELL 8: Merge Results and Save ===

af3_df = pd.DataFrame(results)
merged = pd.merge(top_candidates, af3_df, on='drug_name', how='left')
merged = pd.merge(merged, admet_df, on='drug_name', how='left')

# Overall pass: binding_pass AND admet_pass
merged['final_pass'] = (
    merged.get('binding_pass', False).fillna(False) &
    merged.get('admet_pass', True).fillna(True)
)

# Save to Drive
output_path = f'{PROCESSED_DIR}/interaction_scores_af3.csv'
merged.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Summary
print("\n=== Simulation Summary ===")
print(f"Total candidates simulated: {len(merged)}")
af3_results = merged[merged.get('status', '') != 'failed']
print(f"AF3 successful: {(af3_results.get('status','')=='success').sum()}")
print(f"Binding pass (ipTM > 0.5): {merged.get('binding_pass', pd.Series()).sum()}")
print(f"ADMET pass: {merged.get('admet_pass', pd.Series()).sum()}")
print(f"Final pass (both): {merged['final_pass'].sum()}")

print("\nTop candidates:")
top = merged[merged['final_pass']].sort_values('binding_score', ascending=False)
print(top[['drug_name', 'target', 'binding_score', 'combined_score']].to_string())
