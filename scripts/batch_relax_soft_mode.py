"""Batch-run scripts/relax_soft_mode.py's logic over every material that has
completed DFPT phonon output and shows imaginary (unstable) modes.

Unlike invoking relax_soft_mode.py once per material, this loads the MACE
calculator once and reuses it across all structures, and is resumable: any
material that already has data/tilted_structures/<name>_tilted.cif is
skipped, so a killed/interrupted run can just be restarted.

Recipe: freeze in the single deepest unstable branch at each distinct q-point
sampled by the DFPT run (--auto-per-q 1), keeping only the --auto-top overall
strongest of those, into the smallest supercell commensurate with all of them
at once.

(An earlier version of this script combined every distinct unstable q-point
at once, uncapped -- for the ~176/220 materials whose DFPT run found all 10
sampled q-points unstable, that meant freezing in all 10 simultaneously into
one 320-atom supercell. That landscape turned out to be too frustrated for
LBFGS to relax in a reasonable step budget: 119 of those 176 didn't reach
fmax=0.01 in 300 steps, vs. only 3/33 failures for materials needing <=8
seeds. Whether a material was unstable at 9-10 q-points didn't correlate with
convergence trouble at all beyond that -- --auto-top 2 (matching a real
two-tilt Glazer system) fixes this by construction. The previous run's output
is archived under data/archive_all_modes_v1/.)

Usage:
    uv run python3 scripts/batch_relax_soft_mode.py --ph-root sherlock_outputs/ph_first
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ase.io
import torch
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from mace.calculators import mace_mp

from generate_qe_input import QE_INPUTS_DIR, generate_inputs_for_structure, pbesol_config_for
from paths import PH_FIRST_DIR, TILTED_STRUCTURES_DIR
from soft_mode_distortion import ModeSeed, build_multi_mode_distorted_supercell
from unstable_modes import rank_unstable_modes, top_modes_per_q

LOG_PATH = TILTED_STRUCTURES_DIR / "batch_log.csv"


def already_done(name: str) -> bool:
    cif = TILTED_STRUCTURES_DIR / f"{name}_tilted.cif"
    qe_dir = QE_INPUTS_DIR / f"{name}_tilted"
    return cif.exists() and (qe_dir / f"{name}_tilted.scf.in").exists()


def process_one(
    name: str, dyn_dir: Path, calc, fmax: float, steps: int, amplitude: float, auto_top: int | None
) -> dict:
    start = time.time()
    unstable = top_modes_per_q(rank_unstable_modes(dyn_dir), n_per_q=1)
    if not unstable:
        return {"name": name, "status": "no_unstable_modes", "seconds": 0.0}
    if auto_top is not None:
        unstable = unstable[:auto_top]

    seeds = [ModeSeed(dyn=m.dyn, mode_index=m.mode_index, amplitude_angstrom=amplitude) for m in unstable]
    structure = build_multi_mode_distorted_supercell(seeds)
    n_atoms = structure.get_global_number_of_atoms()

    structure.calc = calc
    optimizer = LBFGS(FrechetCellFilter(structure), logfile=None)
    optimizer.run(fmax=fmax, steps=steps)
    final_fmax = max((f**2).sum() ** 0.5 for f in structure.get_forces())
    n_steps = optimizer.nsteps

    TILTED_STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    cif_path = TILTED_STRUCTURES_DIR / f"{name}_tilted.cif"
    ase.io.write(cif_path, structure)

    out_dir = QE_INPUTS_DIR / f"{name}_tilted"
    generate_inputs_for_structure(cif_path, pbesol_config_for, out_dir)

    return {
        "name": name,
        "status": "converged" if final_fmax <= fmax else "step_limit",
        "n_atoms": n_atoms,
        "n_seeds": len(seeds),
        "n_steps": n_steps,
        "final_fmax": round(final_fmax, 5),
        "seconds": round(time.time() - start, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ph-root", type=Path, default=PH_FIRST_DIR)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument(
        "--auto-top", type=int, default=2,
        help="Keep only the N overall strongest instabilities per material (default: 2)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these material names")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MACE (medium, {device})...", flush=True)
    calc = mace_mp(model="medium", device=device, default_dtype="float64")

    materials = sorted(d.name for d in args.ph_root.iterdir() if d.is_dir())
    if args.only:
        materials = [m for m in materials if m in set(args.only)]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as log_file:
        fieldnames = ["name", "status", "n_atoms", "n_seeds", "n_steps", "final_fmax", "seconds"]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, name in enumerate(materials, 1):
            if already_done(name):
                print(f"[{i}/{len(materials)}] {name}: already done, skipping", flush=True)
                continue
            dyn_dir = args.ph_root / name
            print(f"[{i}/{len(materials)}] {name}: starting...", flush=True)
            try:
                result = process_one(name, dyn_dir, calc, args.fmax, args.steps, args.amplitude, args.auto_top)
            except Exception as exc:
                result = {"name": name, "status": f"error: {exc}", "seconds": 0.0}
            for key in fieldnames:
                result.setdefault(key, "")
            writer.writerow(result)
            log_file.flush()
            print(f"[{i}/{len(materials)}] {name}: {result['status']} ({result['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
