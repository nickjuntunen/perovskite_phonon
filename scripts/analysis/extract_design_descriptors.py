"""Build the descriptor table descriptor_selection.ipynb works from: phonon
and structure descriptors (src/analysis/design_descriptors.py) for every selected
polymorph, plus lattice-instability descriptors from the other states the
MLIP sweep found for the same formula.

Polymorph mode (default), one row per stable_polymorphs.csv row:
    phonon / structure  from <root>/<dir>/phonopy_params.yaml (the saved force
                        constants; no MLIP calls), site lists from the CSV's
                        mp_a_sites / mp_b_sites / mp_x_site
    instability         from candidates.csv:
        distortion_energy_meV_per_atom   E(this run's symmetric state) - E(this
                        polymorph): how far the polymorph relaxed below the MP
                        structure it started from (0 for a symmetric pick)
        omega_min_symmetric_cm1   that symmetric state's lowest frequency:
                        < 0 is the depth of the instability it rolled off
        cubic_*         the same against the formula's Pm-3m state, where the
                        sweep has one (NaN otherwise): the aristotype's double-
                        well depth and soft-mode frequency

Benchmark mode (--mlip-root): every <mlip-root>/<--pattern>/ dir with a
phonopy_params.yaml, one row per (model, structure), model being the first
path component -- the cross-model spread that descriptor_selection.ipynb uses
as a noise floor. Sublattices there come from coordination alone (no site
lists).

Cost: almost all of it is reading phonopy_params.yaml -- 224 GB over the MP
sweep's selected states, nearly all of it full force constants -- so the files
go through mlip_phonons.load_compact, which skips the rows the descriptors don't
need instead of handing ~100 MB of floats to PyYAML. Over the mnt/ sshfs mount
that's ~60 MB/s per worker; the network, not the CPU, sets the pace.

Results are written every --checkpoint-every states. --resume keeps the rows of
an existing --out-csv that succeeded and only redoes the rest -- for an interrupted run, e.g. an sshfs mount that dropped (mount
with -o reconnect,ServerAliveInterval=15 to make that less likely).

Outputs: --out-csv, and next to it <stem>_quantiles.npz with the whole-cell
vDOS quantile function per row ("levels", "quantiles_cm1", "keys" -- keys
match the CSV's key column).

Keys are "<run_name>/<state>" (benchmark mode: the directory under
--mlip-root). Not "<material_id>/<state>": material_id is the run name up to
its first "_", unique for MP entries but shared by every composition of a
generated set (all production mixed halides are "sqs0-2x2x2_<formula>"), so
those rows overwrote each other. Duplicate keys are refused before any work.

Usage:
    uv run python3 scripts/analysis/extract_design_descriptors.py \\
        --root mnt/mlip_phonons_mp/MACE-matpes-r2scan-omat-ft --workers 16
    uv run python3 scripts/analysis/extract_design_descriptors.py \\
        --mlip-root data/mlip_phonons --out-csv data/phonon_benchmark/design_descriptors_models.csv
    # the gauntlet's relaxed (stable, Pnma) CsPbBr3, all models
    uv run python3 scripts/analysis/extract_design_descriptors.py --mlip-root out/gauntlet/mlip \\
        --pattern "*/CsPbBr3/soft_mode" --out-csv data/phonon_benchmark/design_descriptors_gauntlet.csv
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per worker (the per-q diagonalizations are small); must be
# set before numpy loads, or forked workers inherit a full-width thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.paths import MP_SWEEP_DIR, STABLE_POLYMORPHS_DIR
from src.analysis.design_descriptors import (
    QUANTILE_LEVELS,
    describe_state_dir,
    instability_descriptors,
    load_quantiles,
    quantiles_path,
    save_descriptor_table,
)

# Columns main() adds from the CSVs rather than from describe_state_dir()
META_COLUMNS = ["formula", "material_id", "state", "run_name", "model", "structure",
                "distortion_energy_meV_per_atom", "omega_min_symmetric_cm1",
                "cubic_distortion_energy_meV_per_atom", "cubic_omega_min_cm1"]


def _sites(field) -> set[str]:
    return {s for s in str(field).split(";") if s} if isinstance(field, str) else set()


def run(jobs: list[dict], workers: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(describe_state_dir, jobs, chunksize=4))
    rows = pd.DataFrame([r for r, _ in results])
    ok = [(r["key"], q) for r, q in results if q is not None]
    quantiles = np.array([q for _, q in ok]) if ok else np.empty((0, len(QUANTILE_LEVELS)))
    return rows, quantiles, [k for k, _ in ok]


def previous_results(args, jobs: list[dict]) -> tuple[pd.DataFrame, dict]:
    """--resume: the successful rows (descriptor columns only -- main() re-attaches
    the metadata) and quantile functions of an existing --out-csv."""
    npz = quantiles_path(args.out_csv)
    if not (args.resume and args.out_csv.exists() and npz.exists()):
        return pd.DataFrame(columns=["key"]), {}
    done = load_quantiles(npz)  # only successful rows have quantiles
    # round_trip: pandas' default float parser can be 1 ULP off, and this table is
    # rewritten at every checkpoint -- kept rows would drift with each cycle
    old = pd.read_csv(args.out_csv, float_precision="round_trip")
    old = old[old["key"].isin(done) & old["key"].isin({j["key"] for j in jobs})]
    return old.drop(columns=[c for c in META_COLUMNS if c in old]), done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polymorphs-csv", type=Path, default=STABLE_POLYMORPHS_DIR / "stable_polymorphs.csv")
    parser.add_argument("--candidates-csv", type=Path, default=STABLE_POLYMORPHS_DIR / "candidates.csv")
    parser.add_argument("--root", type=Path, default=MP_SWEEP_DIR,
                        help="The run_mlip_phonons.py output the CSV's dir column is relative to")
    parser.add_argument("--mlip-root", type=Path, default=None,
                        help="Benchmark mode: <mlip-root>/<model>/<name>/ dirs instead of the polymorph CSV")
    parser.add_argument("--pattern", default="*/*", help="Benchmark mode: dir glob under --mlip-root (default: */*)")
    parser.add_argument("--out-csv", type=Path, default=STABLE_POLYMORPHS_DIR / "design_descriptors.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Keep the successful rows of an existing --out-csv and only redo the rest")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Write --out-csv after every this many states, so an interrupted run loses at most "
                             "that many (default: 100)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows (for a quick test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.mlip_root is not None:
        dirs = sorted(p.parent for p in args.mlip_root.glob(f"{args.pattern}/phonopy_params.yaml"))[: args.limit]
        rel = [d.relative_to(args.mlip_root) for d in dirs]
        jobs = [{"key": str(r), "dir": d} for r, d in zip(rel, dirs)]
        meta = pd.DataFrame({"key": [j["key"] for j in jobs], "model": [r.parts[0] for r in rel],
                             "structure": ["/".join(r.parts[1:]) for r in rel]})
    else:
        polymorphs = pd.read_csv(args.polymorphs_csv).head(args.limit)
        polymorphs["key"] = polymorphs["run_name"] + "/" + polymorphs["state"]
        jobs = [{"key": r.key, "dir": args.root / r.dir, "a_sites": _sites(r.mp_a_sites),
                 "b_sites": _sites(r.mp_b_sites), "x_site": r.mp_x_site if isinstance(r.mp_x_site, str) else None}
                for r in polymorphs.itertuples()]
        meta = polymorphs[["key", "formula", "material_id", "state", "run_name"]].join(
            instability_descriptors(polymorphs, pd.read_csv(args.candidates_csv)))

    keys = pd.Series([j["key"] for j in jobs])
    if keys.duplicated().any():
        raise SystemExit(f"duplicate keys, rows would overwrite each other: {sorted(set(keys[keys.duplicated()]))[:5]}")

    kept, quantiles = previous_results(args, jobs)
    todo = [j for j in jobs if j["key"] not in quantiles]
    if args.resume:
        print(f"--resume: {len(jobs) - len(todo)} done, {len(todo)} to go", flush=True)
    rows = [kept]
    failed = pd.DataFrame(columns=["key", "error"])
    for start in range(0, len(todo), args.checkpoint_every):
        batch_rows, batch_q, batch_keys = run(todo[start:start + args.checkpoint_every], args.workers)
        rows.append(batch_rows)
        quantiles.update(zip(batch_keys, batch_q))
        if "error" in batch_rows:
            failed = pd.concat([failed, batch_rows.loc[batch_rows["error"].notna(), ["key", "error"]]])
        save_descriptor_table(meta, rows, quantiles, args.out_csv)
        print(f"  {min(start + args.checkpoint_every, len(todo))}/{len(todo)} processed, "
              f"{len(failed)} failed so far", flush=True)
    if not todo:
        save_descriptor_table(meta, rows, quantiles, args.out_csv)

    if len(failed):
        print(f"{len(failed)} state(s) failed (rerun with --resume to retry):\n  "
              + "\n  ".join(f"{k}: {e}" for k, e in zip(failed["key"], failed["error"])))
    print(f"{len(quantiles)}/{len(jobs)} state(s) described -> {args.out_csv}")


if __name__ == "__main__":
    main()
