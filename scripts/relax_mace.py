"""Relax a cubic ABX3 perovskite with a MACE foundation potential before QE SCF+DFPT.

The structures in data/cubic_structures/ come from summing Shannon ionic
radii, not from an energy minimization -- they are a reasonable starting
guess, not an equilibrium geometry. DFPT assumes the reference structure has
~zero Hellmann-Feynman forces; residual forces there show up downstream as
spurious (often imaginary) phonon modes rather than a clean error. Relaxing
positions + cell with a cheap MLIP first gets close to the DFT minimum before
handing the structure to pw.x/ph.x.

Usage: python scripts/relax_mace.py <STRUCTURE> [--fmax 0.01]

Regenerates data/qe_inputs/<STRUCTURE>/ in place from the relaxed structure,
via the same generate_qe_input.pbesol_config_for path scripts/generate_inputs.py
uses, so cutoffs/pseudopotentials stay consistent with the rest of the pipeline.
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

from generate_qe_input import QE_INPUTS_DIR, STRUCTURES_DIR, generate_inputs_for_structure, pbesol_config_for
from paths import RELAXED_STRUCTURES_DIR as RELAXED_DIR


def relax(name: str, fmax: float) -> Path:
    cif_path = STRUCTURES_DIR / f"{name}.cif"
    atoms = ase.io.read(cif_path)
    # mace_mp(model="medium", ...) auto-downloads/caches the standard MACE-MP-0
    # "medium" foundation model matched to the installed mace-torch version --
    # same as relax_soft_mode.py/batch_relax_soft_mode.py. A previous version of
    # this function pointed at a local pinned checkpoint
    # ($GROUP_HOME/nick/mace-mpa-0-medium.model) shared with another project;
    # that file was pickled by a different mace-torch version than what's
    # currently installed (AttributeError on mace.modules.radial.AgnesiTransform
    # while unpickling) and isn't usable as-is.
    atoms.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

    optimizer = LBFGS(FrechetCellFilter(atoms))
    optimizer.run(fmax=fmax, steps=500)
    if not optimizer.converged():
        raise RuntimeError(f"{name}: MACE relaxation did not converge to fmax={fmax} in 500 steps")

    RELAXED_DIR.mkdir(parents=True, exist_ok=True)
    relaxed_path = RELAXED_DIR / f"{name}.cif"
    ase.io.write(relaxed_path, atoms)
    return relaxed_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("--fmax", type=float, default=0.01, help="eV/Angstrom")
    args = parser.parse_args()

    relaxed_path = relax(args.structure, args.fmax)
    print(f"Relaxed structure written to {relaxed_path}")

    scf_path, ph_path = generate_inputs_for_structure(
        relaxed_path, pbesol_config_for, QE_INPUTS_DIR / args.structure
    )
    print(f"Regenerated QE inputs: {scf_path}, {ph_path}")


if __name__ == "__main__":
    main()
