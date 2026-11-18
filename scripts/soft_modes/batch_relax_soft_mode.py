"""Batch-run scripts/soft_modes/relax_soft_mode.py's logic over every material that has
completed DFPT phonon output and shows imaginary (unstable) modes.

Unlike invoking relax_soft_mode.py once per material, this loads the MACE
(+ DFT-D3(BJ), see src/mlip/mlip_calculators.py) calculator once and reuses it across all structures, and is resumable: any
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
    uv run python3 scripts/soft_modes/batch_relax_soft_mode.py --ph-root sherlock_outputs/ph_first
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlip import mlip_phonons
from src.mlip.mlip_calculators import default_device, load_calculator
from src.paths import PH_FIRST_DIR, TILTED_STRUCTURES_DIR
from src.qe.generate_qe_input import QE_INPUTS_DIR, write_structure_and_inputs
from src.qe.qe_phonon_io import run_names
from src.qe.soft_mode_distortion import build_multi_mode_distorted_supercell
from src.qe.unstable_modes import as_mode_seed, rank_unstable_modes, select_modes

LOG_PATH = TILTED_STRUCTURES_DIR / "batch_log.csv"


def already_done(name: str) -> bool:
    cif = TILTED_STRUCTURES_DIR / f"{name}_tilted.cif"
    qe_dir = QE_INPUTS_DIR / f"{name}_tilted"
    return cif.exists() and (qe_dir / f"{name}_tilted.scf.in").exists()


def process_one(
    name: str, dyn_dir: Path, calc, fmax: float, steps: int, amplitude: float, auto_top: int | None
) -> dict:
    start = time.time()
    unstable = select_modes(rank_unstable_modes(dyn_dir), per_q=1, top=auto_top)
    if not unstable:
        return {"name": name, "status": "no_unstable_modes", "seconds": 0.0}

    seeds = [as_mode_seed(m, amplitude) for m in unstable]
    structure = build_multi_mode_distorted_supercell(seeds)
    structure.calc = calc
    converged, n_steps = mlip_phonons.optimize(structure, mode="full", fmax=fmax, steps=steps)
    final_fmax = mlip_phonons.max_force(structure)
    write_structure_and_inputs(structure, f"{name}_tilted", TILTED_STRUCTURES_DIR)

    return {
        "name": name,
        "status": "converged" if converged else "step_limit",
        "n_atoms": len(structure),
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
    parser.add_argument(
        "--model", default="mace:medium",
        help="MLIP spec, see src/mlip/mlip_calculators.py; D3(BJ) is added (default: %(default)s)",
    )
    args = parser.parse_args()

    device = default_device()
    print(f"Loading {args.model} (+ D3, {device})...", flush=True)
    calc = load_calculator(args.model, device=device)

    materials = run_names(args.ph_root)
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
