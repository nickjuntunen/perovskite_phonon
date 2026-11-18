"""Fetch a perovskite-compound dataset from the Materials Project.

Second, independent dataset alongside data/cubic_structures: instead of
synthetically built ideal-cubic ABX3 structures, this pulls real, relaxed MP
entries identified as perovskites. See src/data/mp_perovskites.py for the
traversal strategy (robocrys keyword search + anonymized-formula search,
merged and deduplicated) and why it deviates from the provenance-endpoint
approach in the linked forum discussion.

Requires a Materials Project API key: pass --api-key, set MP_API_KEY, or save
the key to ~/.mp_api_key. Get one at
https://next-gen.materialsproject.org/api#api-key

Usage:
    python scripts/dataset/fetch_mp_perovskites.py
    python scripts/dataset/fetch_mp_perovskites.py --max-materials 20   # quick smoke test
    python scripts/dataset/fetch_mp_perovskites.py --formula-patterns ABC3,ABC2D6 --keywords perovskite
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pymatgen.io.cif import CifWriter

from mp_api.client import MPRester
from src.paths import MP_DIR
from src.data.mp_metadata import mp_run_name
from src.data.mp_perovskites import (
    METADATA_FIELDNAMES,
    DEFAULT_FORMULA_PATTERNS,
    DEFAULT_KEYWORDS,
    collect_candidate_ids,
    fetch_summary_docs,
    resolve_api_key,
    summary_row,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-key", default=None, help="MP API key (else MP_API_KEY env var or ~/.mp_api_key)"
    )
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated robocrys text-search keywords (empty string to skip this source)",
    )
    parser.add_argument(
        "--formula-patterns",
        default=",".join(DEFAULT_FORMULA_PATTERNS),
        help="Comma-separated anonymized-formula patterns, e.g. ABC3,ABC2D6 "
        "(empty string to skip this source)",
    )
    parser.add_argument("--out-dir", type=Path, default=MP_DIR)
    parser.add_argument(
        "--include-deprecated", action="store_true", help="Keep MP-flagged deprecated entries"
    )
    parser.add_argument(
        "--max-materials",
        type=int,
        default=None,
        help="Cap the number of materials fetched, for a quick test run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    formula_patterns = [p.strip() for p in args.formula_patterns.split(",") if p.strip()]

    api_key = resolve_api_key(args.api_key)

    structures_dir = args.out_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key) as mpr:
        sources = collect_candidate_ids(mpr, keywords=keywords, formula_patterns=formula_patterns)

        all_ids: set[str] = set()
        print("Candidates per source:")
        for name, ids in sources.items():
            print(f"  {name}: {len(ids)}")
            all_ids |= ids
        overlap = sum(len(ids) for ids in sources.values()) - len(all_ids)
        print(f"Union: {len(all_ids)} unique material_ids ({overlap} duplicate hits across sources)")

        if args.max_materials is not None:
            all_ids = set(sorted(all_ids)[: args.max_materials])
            print(f"Capped to {len(all_ids)} material_ids (--max-materials)")

        docs = fetch_summary_docs(mpr, list(all_ids))

    if not args.include_deprecated:
        before = len(docs)
        docs = [d for d in docs if not d.deprecated]
        if before != len(docs):
            print(f"Dropped {before - len(docs)} deprecated entries (--include-deprecated to keep)")

    print(f"Fetched {len(docs)} summary documents; writing to {args.out_dir}")

    membership = {mid: sorted(name for name, ids in sources.items() if mid in ids) for mid in all_ids}

    rows = []
    for doc in docs:
        cif_path = structures_dir / f"{mp_run_name(doc.material_id, doc.formula_pretty)}.cif"
        CifWriter(doc.structure).write_file(str(cif_path))
        rows.append(summary_row(doc, membership.get(doc.material_id, []), str(cif_path.relative_to(args.out_dir))))

    metadata_path = args.out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {metadata_path}")
    print(f"Wrote {len(rows)} CIFs to {structures_dir}")


if __name__ == "__main__":
    main()
