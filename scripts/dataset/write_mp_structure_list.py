"""Write the structure list for the MP-perovskite MLIP phonon sweep
(scripts/mlip/run_mlip_phonons.py --structure-list): every structure in
metadata_stable.csv -- structurally verified perovskites within the hull
threshold (scripts/dataset/filter_stable_mp_perovskites.py) -- one CIF path
per line.

Membership depends only on structure and stability, not on descriptors. The
list used to be restricted to the structures cluster_compositions.py could
build a descriptor vector for, which silently dropped 1190 of 3949 valid
entries -- among them every Sn2+ halide perovskite (mendeleev has no Sn2+
Shannon radius), and Eu/Yb compounds (no Pauling electronegativity). A
missing descriptor is an analysis problem, to impute or exclude downstream;
a structure missing from the sweep is never computed at all.
--require-clustered restores the old restriction.

--exclude-done skips structures that already have a finished run (meta.json)
under a sweep root, giving a list of only the new work: run_mlip_phonons.py
skips finished structures itself, but an array over the full list would then
spend most tasks skipping, and the cost would pile up in the rest.

Paths are written relative to the list's directory (structures/<name>.cif),
so syncing data/mp_perovskites/{structures/,<list>} anywhere keeps it valid.

Lines are sorted by nsites, largest first: run_mlip_phonons.py hands array
task i every NUM_TASKS-th line starting at i, so this deals the expensive
cells round-robin across tasks instead of piling them into one slice.

Usage:
    uv run python3 scripts/dataset/write_mp_structure_list.py
    # -> data/mp_perovskites/mlip_structures.txt (all 3949)
    uv run python3 scripts/dataset/write_mp_structure_list.py \\
        --exclude-done mnt/mlip_phonons_mp/MACE-matpes-r2scan-omat-ft --out data/mp_perovskites/mlip_structures_new.txt

--exclude drops named structures: use it for ones that fail deterministically
(job 46929436: Sr2CoSbO6 and NdCuO3 never reach fmax in 1000 relaxation
steps, SrLa2(CoO3)3 fails spglib's Niggli reduction), so an array doesn't keep
spending GPU time on them and marking their tasks FAILED.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlip.phonon_results import unfinished
from src.paths import MP_COMPOSITION_CLUSTERS_CSV, MP_DIR, MP_METADATA_STABLE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=MP_METADATA_STABLE)
    parser.add_argument("--out", type=Path, default=MP_DIR / "mlip_structures.txt")
    parser.add_argument("--require-clustered", action="store_true",
                        help="Only structures in --clusters-csv (the old behaviour: drops entries without descriptors)")
    parser.add_argument("--clusters-csv", type=Path, default=MP_COMPOSITION_CLUSTERS_CSV)
    parser.add_argument("--exclude", nargs="+", default=[], metavar="NAME",
                        help="Leave these structures out, e.g. ones that fail deterministically (by name, "
                             "mp-xxx_<formula>)")
    parser.add_argument("--exclude-done", type=Path, default=None, metavar="SWEEP_ROOT",
                        help="Skip structures with a finished run, <SWEEP_ROOT>/<name>/meta.json (one model's output dir)")
    args = parser.parse_args()

    with args.metadata.open() as f:
        cif_by_name = {Path(r["cif_path"]).stem: (r["cif_path"], int(r["nsites"])) for r in csv.DictReader(f)}
    names = list(cif_by_name)

    if args.require_clustered:
        with args.clusters_csv.open() as f:
            clustered = [r["name"] for r in csv.DictReader(f)]
        missing = [n for n in clustered if n not in cif_by_name]
        if missing:
            raise SystemExit(f"{len(missing)} clustered name(s) not in {args.metadata}, e.g. {missing[:3]}")
        print(f"--require-clustered: {len(clustered)} of {len(names)} structures")
        names = clustered

    if args.exclude_done is not None:
        try:
            todo = unfinished(names, args.exclude_done)
        except NotADirectoryError as exc:
            raise SystemExit(str(exc))
        print(f"--exclude-done: {len(names) - len(todo)} of {len(names)} already finished under {args.exclude_done}")
        names = todo

    if args.exclude:
        unknown = sorted(set(args.exclude) - set(names))
        if unknown:
            raise SystemExit(f"--exclude name(s) not in the list: {unknown}")
        names = [n for n in names if n not in set(args.exclude)]
        print(f"--exclude: {len(args.exclude)} left out")

    if any("." in n for n in names):
        # run_mlip_phonons.py names a run after the file up to its first dot
        raise SystemExit(f"structure name(s) containing '.': {[n for n in names if '.' in n][:3]}")
    # cif_path in metadata is relative to MP_DIR; re-express it relative to the list.
    rows = sorted(((cif_by_name[n][1], MP_DIR / cif_by_name[n][0]) for n in names), key=lambda r: -r[0])
    absent = [p for _, p in rows if not p.exists()]
    if absent:
        raise SystemExit(f"{len(absent)} CIF(s) missing, e.g. {absent[:3]}")

    out_dir = args.out.resolve().parent
    # relpath rather than Path.relative_to: the list may sit outside MP_DIR (e.g. a scratch copy)
    lines = [os.path.relpath(p.resolve(), out_dir) for _, p in rows]
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    if rows:
        print(f"{len(lines)} structures ({rows[0][0]} to {rows[-1][0]} sites) -> {args.out}")
    else:
        print(f"nothing to do; wrote an empty {args.out}")


if __name__ == "__main__":
    main()
