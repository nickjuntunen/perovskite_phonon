"""Add independent structural verification to data/mp_perovskites/metadata.csv.

mp_perovskites.py's candidate search (anonymized-formula + robocrys keyword)
identifies stoichiometry/text matches only; formula-pattern hits in
particular can be non-perovskites that merely reduce to an ABX3 / A2BB'X6
ratio (e.g. YNiSi3, an intermetallic silicide -- see
src/data/perovskite_structure_check.py's docstring). This re-analyzes every CIF
already on disk with that module's corner-sharing-BX6-octahedra check and
appends the result as new columns, so rows can be filtered on actual
structure rather than composition or text alone.

No Materials Project API calls -- everything needed is already in
data/mp_perovskites/structures/.

Always writes to a new file (default metadata_verified.csv), never back onto
metadata.csv itself -- a checkpointing bug once did that and clobbered the
source file with a partial run before it could be safely inspected.

Usage:
    uv run python3 scripts/dataset/verify_mp_perovskites.py
    uv run python3 scripts/dataset/verify_mp_perovskites.py --max-rows 50   # smoke test
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.paths import MP_DIR, MP_METADATA, MP_METADATA_VERIFIED
from src.data.perovskite_structure_check import VERIFICATION_FIELDNAMES, verification_fields


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows processed, for a quick test run")
    parser.add_argument(
        "--checkpoint-every", type=int, default=250, help="Write partial progress to disk every N rows"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MP_METADATA_VERIFIED,
        help="Output path; never metadata.csv itself, to avoid clobbering the source file mid-run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(MP_METADATA, newline="") as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        rows = list(reader)

    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out_path = args.out
    if out_path.resolve() == MP_METADATA.resolve():
        raise SystemExit("--out must not be metadata.csv itself; pick a different output path")

    fieldnames = original_fieldnames + [f for f in VERIFICATION_FIELDNAMES if f not in original_fieldnames]

    n_true = n_false = n_error = 0
    start = time.time()
    for i, row in enumerate(rows):
        row.update(verification_fields(MP_DIR / row["cif_path"]))
        verdict = row["verified_perovskite"]
        if verdict == "ERROR":
            n_error += 1
        elif bool(verdict):
            n_true += 1
        else:
            n_false += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(rows) - (i + 1)) / rate if rate > 0 else float("nan")
            print(
                f"  {i + 1}/{len(rows)} done "
                f"({rate:.1f}/s, ~{remaining / 60:.1f} min left) "
                f"-- verified={n_true} rejected={n_false} errors={n_error}",
                flush=True,
            )

        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows[: i + 1])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone in {(time.time() - start) / 60:.1f} min.")
    print(f"verified_perovskite: True={n_true}, False={n_false}, ERROR={n_error}, total={len(rows)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
