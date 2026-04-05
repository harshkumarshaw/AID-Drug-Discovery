"""
scripts/filter_drkg.py
========================
Filter the DRKG knowledge graph from 5.87M triples to ~500K relevant triples
for LIHC drug repurposing (Gap #3 Fix).

WHY THIS IS NECESSARY:
  Full DRKG has 5.87M triples across many entity types unrelated to LIHC:
  symptoms, anatomy, demographics, species, etc.
  PyKEEN on 5.87M triples needs ~8–16 GB RAM. On 16 GB RAM, OOM likely.
  After filtering: ~500K–1M triples → fits in 8 GB RAM, fast to train.

WHAT WE KEEP:
  - Drug ↔ Gene edges (inhibition, activation, targets — from DRUGBANK, DGIDB)
  - Gene ↔ Disease edges (associated_with — from GNBR, DISEASES)
  - Drug ↔ Disease edges (treats, contraindicated — from DRUGBANK)
  - Gene ↔ Gene edges (PPI with score ≥ 700 — from STRING)
  - Drug ↔ Pathway edges (affects — from HETIONET)

RUN:
  python scripts/filter_drkg.py              # default output
  python scripts/filter_drkg.py --stats      # show relation stats before filtering
  python scripts/filter_drkg.py --max 200000 # cap at 200K for very low RAM
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DRKG_TRIPLES_TSV, DRKG_KEEP_PREFIXES, DRKG_KEEP_ENTITY_TYPES,
    DRKG_MAX_TRIPLES_LOCAL, EXTERNAL_DIR, MVP_MODE,
)

OUTPUT_TSV = EXTERNAL_DIR / "drkg" / "drkg_lihc_filtered.tsv"


def load_drkg_stats(path: Path) -> dict:
    """Count triples per relation prefix without loading all into memory."""
    stats = {}
    total = 0
    print("Counting triples per relation prefix (streaming)...")
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            rel = parts[1]
            prefix = rel.split('::')[0] + '::'
            stats[prefix] = stats.get(prefix, 0) + 1
            total += 1
    print(f"Total triples: {total:,}")
    return stats, total


def entity_type_ok(entity: str, keep_types: list) -> bool:
    """Check if entity type is in the keep list."""
    # DRKG entity format: "EntityType::EntityId" e.g., "Gene::1234", "Compound::DB00001"
    entity_type = entity.split('::')[0] if '::' in entity else entity
    return any(entity_type.lower() == kt.lower() for kt in keep_types)


def filter_drkg(
    input_path: Path,
    output_path: Path,
    keep_prefixes: list,
    keep_entity_types: list,
    max_triples: int = 500_000,
) -> int:
    """
    Stream DRKG TSV and write only relevant triples to output.
    Returns: number of triples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_relation = 0
    skipped_entity = 0

    print(f"Filtering DRKG...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Keep prefixes: {keep_prefixes}")
    print(f"  Max triples: {max_triples:,} (0 = no limit)")

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i % 500_000 == 0 and i > 0:
                print(f"  Processed: {i:,} | Kept: {kept:,} | "
                      f"Skip(rel): {skipped_relation:,} | Skip(ent): {skipped_entity:,}")

            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue

            head, rel, tail = parts

            # ── Filter 1: Relation prefix ──
            rel_prefix = rel.split('::')[0] + '::'
            if not any(rel.startswith(p) for p in keep_prefixes):
                skipped_relation += 1
                continue

            # ── Filter 2: Entity types ──
            head_ok = entity_type_ok(head, keep_entity_types)
            tail_ok = entity_type_ok(tail, keep_entity_types)
            if not (head_ok and tail_ok):
                skipped_entity += 1
                continue

            # ── Write ──
            f_out.write(line)
            kept += 1

            # ── Max triples cap ──
            if max_triples > 0 and kept >= max_triples:
                print(f"  Reached max triples cap ({max_triples:,})")
                break

    return kept


def validate_output(output_path: Path, n_triples: int):
    """Quick sanity check on the filtered output."""
    print(f"\nValidating output...")

    # Check drug-disease triples exist
    drugbank_count = 0
    gnbr_count = 0
    with open(output_path, 'r') as f:
        for line in f:
            if 'DRUGBANK' in line:
                drugbank_count += 1
            if 'GNBR' in line:
                gnbr_count += 1

    print(f"  Total triples: {n_triples:,}")
    print(f"  DRUGBANK triples: {drugbank_count:,}")
    print(f"  GNBR triples: {gnbr_count:,}")

    # Check known entities are present
    found_sorafenib = False
    found_vegfr2 = False
    with open(output_path, 'r') as f:
        content = f.read(5_000_000)  # Check first 5MB
        found_sorafenib = 'sorafenib' in content.lower() or 'DB00398' in content
        found_vegfr2 = 'VEGFR2' in content or 'KDR' in content or '3791' in content

    print(f"  Sorafenib present: {'✅' if found_sorafenib else '⚠️ NOT FOUND'}")
    print(f"  VEGFR2/KDR present: {'✅' if found_vegfr2 else '⚠️ NOT FOUND'}")

    if not found_sorafenib:
        print("  WARNING: Sorafenib not found in filtered DRKG. Phase 4 validation "
              "anchor may fail. Check DRKG_KEEP_PREFIXES includes DRUGBANK.")
    if not found_vegfr2:
        print("  WARNING: VEGFR2 not found. Check entity type filtering.")


def main():
    parser = argparse.ArgumentParser(
        description="Filter DRKG to LIHC-relevant triples"
    )
    parser.add_argument("--stats", action="store_true",
                        help="Show relation stats without filtering")
    parser.add_argument("--max", type=int, default=None,
                        help=f"Max triples (default: {DRKG_MAX_TRIPLES_LOCAL:,})")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip output validation step")
    args = parser.parse_args()

    if not DRKG_TRIPLES_TSV.exists():
        print(f"ERROR: DRKG file not found at {DRKG_TRIPLES_TSV}")
        print("Download it first:")
        print("  python scripts/download_datasets.py --tier mvp")
        sys.exit(1)

    if args.stats:
        stats, total = load_drkg_stats(DRKG_TRIPLES_TSV)
        print("\nRelation prefix counts (sorted by frequency):")
        for prefix, count in sorted(stats.items(), key=lambda x: -x[1]):
            keep = "✅ KEEP" if any(prefix.startswith(p) for p in DRKG_KEEP_PREFIXES) else "  skip"
            print(f"  {keep}  {prefix:<25} {count:>10,}  ({count/total*100:.1f}%)")
        return

    max_triples = args.max if args.max is not None else DRKG_MAX_TRIPLES_LOCAL

    if OUTPUT_TSV.exists():
        # Count existing lines
        with open(OUTPUT_TSV, 'r') as f:
            existing = sum(1 for _ in f)
        print(f"Filtered DRKG already exists: {OUTPUT_TSV} ({existing:,} triples)")
        print("To re-filter, delete the file and run again.")
        return

    n_kept = filter_drkg(
        input_path=DRKG_TRIPLES_TSV,
        output_path=OUTPUT_TSV,
        keep_prefixes=DRKG_KEEP_PREFIXES,
        keep_entity_types=DRKG_KEEP_ENTITY_TYPES,
        max_triples=max_triples,
    )

    print(f"\n✅ Filtered DRKG saved: {OUTPUT_TSV}")
    print(f"   {n_kept:,} triples (from 5,874,261 original)")
    print(f"   Reduction: {5874261 - n_kept:,} triples removed "
          f"({(5874261 - n_kept) / 5874261 * 100:.0f}% removed)")

    if not args.no_validate:
        validate_output(OUTPUT_TSV, n_kept)

    # Update config reference note
    print(f"\nNext step: In pipeline/drug_repurposing.py, load:")
    print(f"  from config import EXTERNAL_DIR")
    print(f"  drkg_path = EXTERNAL_DIR / 'drkg' / 'drkg_lihc_filtered.tsv'")
    print(f"  # Pass to PyKEEN as training triples")


if __name__ == "__main__":
    main()
