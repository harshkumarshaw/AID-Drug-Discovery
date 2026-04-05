"""
scripts/download_datasets.py
==============================
One-script data download for the AI Drug Discovery Pipeline.
Downloads all required and recommended datasets in priority order.

Usage:
    python scripts/download_datasets.py --tier mvp       # MVP datasets only
    python scripts/download_datasets.py --tier tier2     # All recommended
    python scripts/download_datasets.py --tier all       # Everything including optional
    python scripts/download_datasets.py --list           # List what will be downloaded

Requirements: requests, tqdm, gzip, pathlib (all stdlib or pip install requests tqdm)
"""

import argparse
import gzip
import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from config import (
        EXTERNAL_DIR, PROCESSED_DIR, TARGETS_DIR, DRKG_DIR,
        GDSC2_CSV, CCLE_EXPRESSION_CSV, CCLE_MUTATION_CSV,
        CCLE_CNV_CSV, CCLE_MODEL_INFO_CSV,
        STRING_PPI_TSV, MSIGDB_KEGG_GMT, HPA_PATHOLOGY_TSV,
        LIHC_TARGET_PDBS,
    )
except ImportError:
    print("ERROR: config.py not found. Run from the project root directory.")
    sys.exit(1)

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing requests and tqdm...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests', 'tqdm', '--quiet'])
    import requests
    from tqdm import tqdm


# ─── Download Helpers ────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, description: str = "", force: bool = False):
    """Download a file with progress bar."""
    if dest.exists() and not force:
        print(f"  ✓ Already exists: {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading: {description or dest.name}")
    print(f"     From: {url}")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))

        with open(dest, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=dest.name, ncols=70
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"  ✓ Saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def gunzip_file(gz_path: Path, out_path: Path):
    """Extract .gz file."""
    if out_path.exists():
        print(f"  ✓ Already extracted: {out_path.name}")
        return
    print(f"  📦 Extracting: {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  ✓ Extracted: {out_path.name}")


def install_pip_package(package: str):
    """Install a pip package."""
    print(f"  📦 Installing: {package}")
    subprocess.run([sys.executable, '-m', 'pip', 'install', package, '--quiet'],
                   check=True)
    print(f"  ✓ Installed: {package}")


# ─── Dataset Definitions ─────────────────────────────────────────────────────

DATASETS = {

    # ── MVP TIER ─────────────────────────────────────────────────────────────
    "drkg": {
        "tier": "mvp",
        "description": "Drug Repurposing Knowledge Graph (5.87M triples)",
        "phase": "Phase 4 (repurposing)",
        "size_mb": 380,
        "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/DRKG/drkg.tsv.gz",
        "dest_gz": DRKG_DIR / "drkg.tsv.gz",
        "dest": DRKG_DIR / "drkg.tsv",
        "action": "download_gunzip",
    },

    "ccle_expression": {
        "tier": "mvp",
        "description": "CCLE Gene Expression (DepMap 24Q4, TPM log1p, ~1800 cell lines)",
        "phase": "Phase 3 (drug response)",
        "size_mb": 600,
        "url": "https://depmap.org/portal/api/downloads/files?fileName=OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "dest": CCLE_EXPRESSION_CSV,
        "action": "download",
    },

    "ccle_model_info": {
        "tier": "mvp",
        "description": "CCLE Model Metadata (cell line name harmonization)",
        "phase": "Phase 3 (drug response) — cell line name matching",
        "size_mb": 5,
        "url": "https://depmap.org/portal/api/downloads/files?fileName=Model.csv",
        "dest": CCLE_MODEL_INFO_CSV,
        "action": "download",
    },

    "hpa_pathology": {
        "tier": "mvp",
        "description": "Human Protein Atlas — Liver Cancer Pathology Data",
        "phase": "Phase 2 (biomarker validation)",
        "size_mb": 10,
        "url": "https://www.proteinatlas.org/download/pathology.tsv.zip",
        "dest": HPA_PATHOLOGY_TSV,
        "action": "download_note",  # Zip, needs manual extraction
        "note": "Download ZIP from https://www.proteinatlas.org/download/pathology.tsv.zip and extract pathology.tsv",
    },

    # PDB Target Structures
    **{
        f"pdb_{name}": {
            "tier": "mvp",
            "description": f"PDB structure: {name} (PDB: {dest.stem})",
            "phase": "Phase 5 (molecular simulation)",
            "size_mb": 1,
            "url": f"https://files.rcsb.org/download/{dest.stem}.pdb",
            "dest": dest,
            "action": "download",
        }
        for name, dest in LIHC_TARGET_PDBS.items()
    },

    "pip_vina": {
        "tier": "mvp",
        "description": "AutoDock Vina Python bindings (local docking)",
        "phase": "Phase 5 (molecular simulation — local)",
        "size_mb": 50,
        "action": "pip",
        "package": "vina",
    },

    "pip_admet_ai": {
        "tier": "mvp",
        "description": "ADMET-AI (42 ADMET endpoints from SMILES)",
        "phase": "Phase 5 (ADMET filtering)",
        "size_mb": 100,
        "action": "pip",
        "package": "admet-ai",
    },

    "pip_pykeen": {
        "tier": "mvp",
        "description": "PyKEEN (Knowledge Graph Embeddings)",
        "phase": "Phase 4 (drug repurposing)",
        "size_mb": 50,
        "action": "pip",
        "package": "pykeen",
    },

    "pip_lifelines": {
        "tier": "mvp",
        "description": "lifelines (Kaplan-Meier survival analysis)",
        "phase": "Phase 1 (subtyping validation)",
        "size_mb": 10,
        "action": "pip",
        "package": "lifelines",
    },

    "pip_cmappy": {
        "tier": "mvp",
        "description": "cmapPy (LINCS gctx loading)",
        "phase": "Phase 3 (drug response — Colab only for gctx)",
        "size_mb": 5,
        "action": "pip",
        "package": "cmapPy",
    },

    # ── TIER 2 ───────────────────────────────────────────────────────────────
    "ccle_mutations": {
        "tier": "tier2",
        "description": "CCLE Somatic Mutations (binary matrix, DepMap 24Q4)",
        "phase": "Phase 3 (drug response — mutation AE branch)",
        "size_mb": 250,
        "url": "https://depmap.org/portal/api/downloads/files?fileName=OmicsSomaticMutationsMatrixDamaging.csv",
        "dest": CCLE_MUTATION_CSV,
        "action": "download",
    },

    "ccle_cnv": {
        "tier": "tier2",
        "description": "CCLE Copy Number Variation (DepMap 24Q4)",
        "phase": "Phase 3 (drug response — CNV AE branch)",
        "size_mb": 80,
        "url": "https://depmap.org/portal/api/downloads/files?fileName=OmicsCNSegmentsProfile.csv",
        "dest": CCLE_CNV_CSV,
        "action": "download",
    },

    "string_ppi": {
        "tier": "tier2",
        "description": "STRING Protein-Protein Interactions (Homo sapiens, v12)",
        "phase": "Phase 4 (KG augmentation)",
        "size_mb": 1200,
        "url": "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz",
        "dest_gz": EXTERNAL_DIR / "string_ppi.txt.gz",
        "dest": STRING_PPI_TSV,
        "action": "download_gunzip",
    },

    "msigdb_kegg": {
        "tier": "tier2",
        "description": "MSigDB KEGG Pathway Gene Sets (Homo sapiens)",
        "phase": "Phase 2 (biomarker enrichment analysis)",
        "size_mb": 2,
        "url": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/c2.cp.kegg.v2023.2.Hs.symbols.gmt",
        "dest": MSIGDB_KEGG_GMT,
        "action": "download",
    },

    # ── REGISTRATION REQUIRED (provide instructions only) ─────────────────────
    "gdsc2": {
        "tier": "mvp",
        "description": "GDSC2 Drug Sensitivity (LN-IC50, AUC for ~300 drugs × ~900 cell lines)",
        "phase": "Phase 3 (drug response labels)",
        "size_mb": 20,
        "action": "manual",
        "instructions": [
            "1. Go to: https://www.cancerrxgene.org/downloads/bulk_download",
            "2. Under 'Drug Sensitivity', download: 'GDSC2 fitted dose-response parameters'",
            "3. File: GDSC2_fitted_dose_response_27Oct23.xlsx",
            "4. Convert to CSV:",
            "   python -c \"import pandas as pd; pd.read_excel('GDSC2_fitted_dose_response_27Oct23.xlsx').to_csv('gdsc2_drug_sensitivity.csv', index=False)\"",
            f"5. Move to: {GDSC2_CSV}",
        ],
    },

    "tcga_rnaseq": {
        "tier": "mvp",
        "description": "TCGA-LIHC RNA-seq (STAR Counts, ~371 patients × 60,660 genes)",
        "phase": "Phase 1 + Phase 2",
        "size_mb": 200,
        "action": "manual",
        "instructions": [
            "1. Go to: https://portal.gdc.cancer.gov/",
            "2. Click 'Exploration' → Cases → Filter: Project = TCGA-LIHC",
            "3. Files tab → Data Category = Transcriptome Profiling",
            "   → Workflow Type = STAR - Counts",
            "4. Add to Cart → Download Cart (Manifest)",
            "5. Install GDC client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool",
            "6. Run: gdc-client download -m gdc_manifest.txt -d TCGA/raw/",
            "7. Merge individual htseq files into matrix using scripts/merge_tcga_rnaseq.py (TBD)",
        ],
    },

    "tcga_mutation": {
        "tier": "mvp",
        "description": "TCGA-LIHC Somatic Mutations (Masked MAF, ~374 patients)",
        "phase": "Phase 2 (biomarker discovery — mutation features)",
        "size_mb": 50,
        "action": "manual",
        "instructions": [
            "1. Same GDC portal as RNA-seq",
            "2. Files tab → Data Category = Simple Nucleotide Variation",
            "   → Data Type = Masked Somatic Mutation",
            "3. Download MAF file",
            f"4. Place at: {ROOT_DIR / 'data/processed/tcga_lihc_mutations.maf'}",
        ],
    },

    "cosmic": {
        "tier": "tier2",
        "description": "COSMIC Cancer Gene Census (known cancer genes with LIHC annotation)",
        "phase": "Phase 2 validation",
        "size_mb": 1,
        "action": "manual",
        "instructions": [
            "1. Register free at: https://cancer.sanger.ac.uk/census",
            "2. Download: cancer_gene_census.csv",
            f"3. Place at: {ROOT_DIR / 'data/external/cosmic_census.csv'}",
        ],
    },

    "disgenet": {
        "tier": "tier2",
        "description": "DisGeNET gene-disease associations (for KG augmentation)",
        "phase": "Phase 4 (drug repurposing — KG edges)",
        "size_mb": 500,
        "action": "manual",
        "instructions": [
            "1. Register free at: https://www.disgenet.com/",
            "2. Download: 'curated gene-disease associations' (all_gene_disease_associations.tsv.gz)",
            "3. Extract and place at: data/external/disgenet_gene_disease.tsv",
        ],
    },

    "alphafold3_weights": {
        "tier": "tier2",
        "description": "AlphaFold3 model weights (for Colab Phase 5)",
        "phase": "Phase 5 (molecular simulation — Colab only)",
        "size_mb": 1500,
        "action": "manual",
        "instructions": [
            "1. Fill DeepMind form: https://forms.gle/svvpY4u2jsHEwWYS6",
            "2. Wait 1–7 days for email with download link",
            "3. Download weights (~1.5 GB)",
            "4. Upload to Google Drive: MyDrive/AID_Pipeline/alphafold3_weights/",
            "5. Access from Colab via Drive mount",
        ],
    },
}


# ─── Main Execution ───────────────────────────────────────────────────────────

def get_datasets_for_tier(tier: str) -> dict:
    """Return datasets for the given tier (mvp, tier2, all)."""
    if tier == "all":
        return DATASETS
    tiers = {"mvp": {"mvp"}, "tier2": {"mvp", "tier2"}}
    allowed = tiers.get(tier, {"mvp"})
    return {k: v for k, v in DATASETS.items() if v.get("tier") in allowed}


def list_datasets(tier: str):
    """Print a formatted list of datasets."""
    datasets = get_datasets_for_tier(tier)
    print(f"\n{'='*70}")
    print(f"Dataset Inventory ({tier.upper()} tier — {len(datasets)} items)")
    print(f"{'='*70}\n")

    auto = {k: v for k, v in datasets.items() if v["action"] in ("download", "download_gunzip", "pip")}
    manual = {k: v for k, v in datasets.items() if v["action"] == "manual"}
    note = {k: v for k, v in datasets.items() if v["action"] == "download_note"}

    print(f"📥 AUTO-DOWNLOADABLE ({len(auto)} items):")
    for k, v in auto.items():
        size = f"~{v.get('size_mb', '?')} MB"
        print(f"  [{v['tier'].upper()}] {v['description']} ({size})")

    print(f"\n📝 MANUAL DOWNLOAD REQUIRED ({len(manual)} items):")
    for k, v in manual.items():
        print(f"  [{v['tier'].upper()}] {v['description']}")

    total_auto = sum(v.get('size_mb', 0) for v in auto.values())
    total_manual = sum(v.get('size_mb', 0) for v in manual.values())
    print(f"\nTotal auto-download: ~{total_auto / 1000:.1f} GB")
    print(f"Total manual download: ~{total_manual / 1000:.1f} GB")


def run_downloads(tier: str, force: bool = False):
    """Execute all automatic downloads for the given tier."""
    datasets = get_datasets_for_tier(tier)

    # Create directories
    for d in [EXTERNAL_DIR, PROCESSED_DIR, TARGETS_DIR, DRKG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    successes, failures, skipped_manual = [], [], []

    for key, ds in datasets.items():
        print(f"\n{'─'*60}")
        print(f"[{ds['tier'].upper()}] {ds['description']}")
        print(f"Phase: {ds['phase']}")

        action = ds["action"]

        if action == "pip":
            try:
                install_pip_package(ds["package"])
                successes.append(key)
            except Exception as e:
                print(f"  ✗ pip install {ds['package']} failed: {e}")
                failures.append(key)

        elif action == "download":
            ok = download_file(ds["url"], ds["dest"], ds["description"], force)
            (successes if ok else failures).append(key)

        elif action == "download_gunzip":
            ok = download_file(ds["url"], ds["dest_gz"], ds["description"], force)
            if ok:
                gunzip_file(ds["dest_gz"], ds["dest"])
            (successes if ok else failures).append(key)

        elif action in ("manual", "download_note"):
            print(f"  ⚠️  MANUAL DOWNLOAD REQUIRED:")
            for step in ds.get("instructions", [ds.get("note", "See docs/data_inventory.md")]):
                print(f"     {step}")
            skipped_manual.append(key)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  ✓ Success: {len(successes)}")
    print(f"  ✗ Failed:  {len(failures)}")
    print(f"  ⚠️  Manual: {len(skipped_manual)}")
    if failures:
        print(f"\n  Failed items: {', '.join(failures)}")
    if skipped_manual:
        print(f"\n  Manual items to complete:")
        for key in skipped_manual:
            print(f"    - {DATASETS[key]['description']}")
        print(f"\n  Full instructions: docs/data_inventory.md")
    print(f"{'='*60}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for the AI Drug Discovery Pipeline"
    )
    parser.add_argument(
        "--tier",
        choices=["mvp", "tier2", "all"],
        default="mvp",
        help="Download tier: mvp (required for MVP), tier2 (all recommended), all"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets without downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets(args.tier)
    else:
        print(f"\nStarting {args.tier.upper()} tier downloads...")
        print(f"Data will be saved to: {ROOT_DIR / 'data'}\n")
        run_downloads(args.tier, force=args.force)
