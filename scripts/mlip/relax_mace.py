"""Relax a cubic ABX3 perovskite with a MACE foundation potential before QE SCF+DFPT.

The structures in data/cubic_structures/ come from summing Shannon ionic
radii, not from an energy minimization -- they are a reasonable starting
guess, not an equilibrium geometry. DFPT assumes the reference structure has
~zero Hellmann-Feynman forces; residual forces there show up downstream as
spurious (often imaginary) phonon modes rather than a clean error. Relaxing
positions + cell with a cheap MLIP first gets close to the DFT minimum before
handing the structure to pw.x/ph.x.

The potential is MACE + DFT-D3(BJ), like every MLIP calculation here (see
src/mlip/mlip_calculators.py; --model takes any spec that module accepts).

Usage: python scripts/mlip/relax_mace.py <STRUCTURE> [--fmax 0.01]

Regenerates data/qe_inputs/<STRUCTURE>/ in place from the relaxed structure,
via the same generate_qe_input.pbesol_config_for path scripts/qe/generate_inputs.py
uses, so cutoffs/pseudopotentials stay consistent with the rest of the pipeline.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ase.io

from src.mlip import mlip_phonons
from src.mlip.mlip_calculators import load_calculator
from src.paths import CUBIC_STRUCTURES_DIR, RELAXED_STRUCTURES_DIR
from src.qe.generate_qe_input import write_structure_and_inputs

# Same foundation checkpoint already in use for this group's MACE work
# (see ~/mat_tmmc/scripts/subs/sherlock/train_distill_mace.sh).
FOUNDATION_MODEL = "/home/groups/rotskoff/nick/mace-mpa-0-medium.model"
STEPS = 500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structure")
    parser.add_argument("--fmax", type=float, default=0.01, help="eV/Angstrom")
    parser.add_argument(
        "--model", default=FOUNDATION_MODEL,
        help="MLIP spec, see src/mlip/mlip_calculators.py; D3(BJ) is added (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="A few-atom cell doesn't need a GPU (default: cpu)")
    args = parser.parse_args()

    atoms = ase.io.read(CUBIC_STRUCTURES_DIR / f"{args.structure}.cif")
    atoms.calc = load_calculator(args.model, device=args.device)
    converged, _ = mlip_phonons.optimize(atoms, mode="full", fmax=args.fmax, steps=STEPS, logfile="-")
    if not converged:
        raise RuntimeError(f"{args.structure}: MLIP relaxation did not converge to fmax={args.fmax} in {STEPS} steps")

    relaxed_path, scf_path, ph_path = write_structure_and_inputs(atoms, args.structure, RELAXED_STRUCTURES_DIR)
    print(f"Relaxed structure written to {relaxed_path}")
    print(f"Regenerated QE inputs: {scf_path}, {ph_path}")


if __name__ == "__main__":
    main()
