"""Freeze the most unstable phonon mode from a QE .dyn file into a supercell,
relax it with a MACE foundation-model potential, and regenerate QE inputs.

This is the follow-up to a completed ph.x DFPT run that found imaginary
zone-boundary frequencies (octahedral tilting etc. -- see
src/soft_mode_distortion.py for why the original small cell can't relax into
that on its own): cheaply pre-relax a symmetry-broken supercell with MACE
before paying for another expensive QE run on it.

Usage:
    uv run python3 scripts/relax_soft_mode.py \\
        sherlock_outputs/ph_first/CsPbI3/CsPbI3.dyn10 CsPbI3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import ase.io
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from mace.calculators import mace_mp

from generate_qe_input import QE_INPUTS_DIR, generate_inputs_for_structure, pbesol_config_for
from soft_mode_distortion import build_distorted_supercell, most_unstable_mode, parse_dyn_file

TILTED_STRUCTURES_DIR = PROJECT_ROOT / "data" / "tilted_structures"


def relax_with_mace(atoms, fmax: float, steps: int):
    atoms.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    optimizer = LBFGS(FrechetCellFilter(atoms))
    optimizer.run(fmax=fmax, steps=steps)
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dyn_file", type=Path, help="Path to the .dyn file at the unstable q-point")
    parser.add_argument("name", help="Base structure name, e.g. CsPbI3 (used for output filenames)")
    parser.add_argument("--mode-index", type=int, default=None, help="Override auto-picked (most unstable) mode")
    parser.add_argument("--supercell", type=int, nargs=3, default=(2, 2, 2), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--amplitude", type=float, default=0.15, help="Max single-atom seed displacement, angstrom")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold, eV/angstrom")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    dyn = parse_dyn_file(args.dyn_file)
    mode_index = args.mode_index if args.mode_index is not None else most_unstable_mode(dyn)
    frequency = dyn.frequencies_cm1[mode_index]
    if frequency >= 0:
        print(f"Warning: mode {mode_index} has frequency {frequency:.2f} cm-1 (not unstable) -- continuing anyway.")
    print(f"Freezing mode {mode_index} ({frequency:.2f} cm-1) at q={dyn.q_frac} into a {tuple(args.supercell)} supercell")

    structure = build_distorted_supercell(dyn, mode_index, tuple(args.supercell), args.amplitude)

    print("Relaxing with MACE (mace_mp, medium, float64)...")
    relax_with_mace(structure, fmax=args.fmax, steps=args.steps)

    TILTED_STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{args.name}_tilted"
    cif_path = TILTED_STRUCTURES_DIR / f"{out_name}.cif"
    ase.io.write(cif_path, structure)
    print(f"Relaxed structure written to {cif_path}")

    out_dir = QE_INPUTS_DIR / out_name
    scf_path, ph_path = generate_inputs_for_structure(cif_path, pbesol_config_for, out_dir)
    print(f"Regenerated QE inputs: {scf_path}, {ph_path}")


if __name__ == "__main__":
    main()
