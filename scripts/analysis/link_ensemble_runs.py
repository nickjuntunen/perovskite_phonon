"""One run root for an SQS-ensemble set whose cells partly reuse earlier runs.

The ensemble set (data/mixed_halides_ensemble, generate_mixed_halides.py
--mode noise-check) computed only the cells that no finished run already had
(dedupe: same cell, species and positions); reuse.csv names, for each of the
others, the run directory that holds it (relative to $SCRATCH/perovskite_phonon,
i.e. to mnt/). This builds <set>/runs/ with one symlink per manifest cell, named
by the cell, pointing either to the ensemble sweep's run or to the reused one.
select_stable_polymorphs.py and extract_design_descriptors.py take the run name
from the directory name, so they see every cell under its ensemble name and the
manifest weights join as for any noise-check set.

The links are rebuilt from scratch on every call. A cell whose run directory
doesn't exist yet is left out and counted; one whose run hasn't finished (no
meta.json) is linked, and the selection skips it.

Usage:
    uv run python3 scripts/analysis/link_ensemble_runs.py            # -> data/mixed_halides_ensemble/runs
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

MODEL = "MACE-matpes-r2scan-omat-ft"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, default=PROJECT_ROOT / "data" / "mixed_halides_ensemble")
    parser.add_argument("--scratch", type=Path, default=PROJECT_ROOT / "mnt",
                        help="$SCRATCH/perovskite_phonon as seen from here (default: the sshfs mount mnt/)")
    parser.add_argument("--sweep", default="mlip_phonons_mixed_ensemble",
                        help="The ensemble's own sweep dir under --scratch (its OUT_ROOT)")
    args = parser.parse_args()

    scratch = args.scratch.resolve()
    sweep = scratch / args.sweep / MODEL
    if not sweep.is_dir():
        raise SystemExit(f"{sweep} not found: is mnt/ mounted?")
    names = pd.read_csv(args.set_dir / "manifest.csv")["name"]
    reuse = pd.read_csv(args.set_dir / "reuse.csv").set_index("name")["reused_from"]

    out = args.set_dir / "runs"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    n = {"ensemble": 0, "reused": 0, "missing": 0, "unfinished": 0}
    for name in names:
        src = scratch / reuse[name] if name in reuse.index else sweep / name
        if not src.is_dir():
            n["missing"] += 1
            continue
        (out / name).symlink_to(src, target_is_directory=True)
        n["reused" if name in reuse.index else "ensemble"] += 1
        n["unfinished"] += not (src / "meta.json").exists()
    print(f"{len(names)} manifest cells -> {out}: {n['ensemble']} from {args.sweep}, {n['reused']} reused; "
          f"{n['unfinished']} linked but unfinished (no meta.json), {n['missing']} with no run dir yet")


if __name__ == "__main__":
    main()
