"""Second filter pass on data/mp_perovskites/metadata_verified.csv: drop
structurally-unverified rows and reject anything too far above the convex
hull.

Multiple MP entries can share the same formula_pretty while being genuinely
different structures/polymorphs -- e.g. Er3AlC appears twice, once as the
hull-stable phase (mp-568021, energy_above_hull=0) and once ~1.9 eV/atom
above the hull (mp-10038). The energy_above_hull threshold below handles this
on its own: the far-above-hull duplicate is dropped for being unstable, not
for being a duplicate. Two entries with the same formula are both kept when
both are legitimately near the hull -- distinct stable/metastable polymorphs
of the same composition are not duplicates to collapse.

Condition applied: energy_above_hull <= --max-e-above-hull (default 0.05
eV/atom, a common DFT metastability cutoff -- 0.01 eV/atom is essentially
hull-stable modulo numerical noise, 1.0 eV/atom is not a real candidate
material).

Applied only to verified_perovskite == True rows (verified_perovskite ==
"ERROR" or False rows are dropped regardless of energy).

Usage:
    uv run python3 scripts/dataset/filter_stable_mp_perovskites.py
    uv run python3 scripts/dataset/filter_stable_mp_perovskites.py --max-e-above-hull 0.1
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mp_metadata import filter_stable_rows
from src.paths import MP_METADATA_STABLE, MP_METADATA_VERIFIED


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-e-above-hull",
        type=float,
        default=0.05,
        help="Reject rows with energy_above_hull (eV/atom) greater than this (default: 0.05)",
    )
    parser.add_argument(
        "--in", dest="in_path", type=Path, default=MP_METADATA_VERIFIED, help="Input CSV (default: metadata_verified.csv)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MP_METADATA_STABLE,
        help="Output path (default: metadata_stable.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.in_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept, stats = filter_stable_rows(rows, args.max_e_above_hull)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"{len(rows)} input rows")
    print(f"  dropped {stats['not_verified']} not verified as perovskite")
    print(f"  dropped {stats['above_threshold']} with energy_above_hull > {args.max_e_above_hull} eV/atom")
    print(f"kept {len(kept)} near-hull, verified perovskites")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
