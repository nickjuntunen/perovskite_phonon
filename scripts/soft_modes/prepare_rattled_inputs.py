"""DFT counterpart of run_mlip_phonons.py's rattle retry: for every material
whose completed DFPT run shows imaginary modes, rattle that run's geometry,
relax it on the DFT PES without symmetry, and redo DFPT on the result.

Writes data/qe_inputs/<name>_rattled/ from the first run's own inputs
(<scf-root>/<name>/<name>.scf.in and <ph-root>/<name>/<name>.ph.in, i.e. the
post-MACE-relax geometry and whatever smearing/magnetism/k-point settings that
run actually used), changed only where the retry needs it:

  - positions rattled (ase.Atoms.rattle, Gaussian, --stdev Angstrom)
  - calculation 'scf' -> 'relax' (fixed cell, BFGS) with a tight
    forc_conv_thr: DFPT needs ~zero forces, and the soft directions this is
    descending along are exactly where a loose threshold stops early -- and
    nstep raised from pw.x's default 50, for the same reason
  - prefix / fildyn renamed to <name>_rattled

The relax is DFT's, not MACE's, on purpose: the instability lives on the DFT
PES, and ph.x at a MACE minimum that isn't a DFT minimum would just report
the residual forces as (more) imaginary modes. The rattle breaks every
symmetry, so pw.x finds none and the relax is unconstrained -- and ph.x then
has to do the full (time-reversal-only) q-grid, several times the cost of the
symmetric first run.

What this can and can't fix: a rattled unit cell can only distort
commensurately with itself, i.e. remove Gamma-point instabilities. Zone-
boundary ones (R/M octahedral tilts) need a supercell -- that's
batch_relax_soft_mode.py's job, not this one's.

Submit on Sherlock without the MACE relax step (it would read the cubic
structure, not these inputs):
    scripts/jobs/sherlock/submit_all.sh --no-mace-relax CsPbBr3_rattled

Usage:
    uv run python3 scripts/soft_modes/prepare_rattled_inputs.py               # every unstable run
    uv run python3 scripts/soft_modes/prepare_rattled_inputs.py --only CsPbBr3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ase.io
import numpy as np

from src.paths import PH_FIRST_DIR, QE_INPUTS_DIR, RATTLED_STRUCTURES_DIR, SCF_FIRST_DIR
from src.qe.edit_inputs import relax_input_from_scf, renamed_ph_input
from src.qe.qe_phonon_io import run_names
from src.qe.unstable_modes import rank_unstable_modes

SUFFIX = "_rattled"
def prepare_one(name: str, args) -> str:
    dyn_dir = args.ph_root / name
    unstable = rank_unstable_modes(dyn_dir, freq_tol_cm1=-args.imag_tol)
    if not unstable:
        return "stable, skipped"
    scf_in = args.scf_root / name / f"{name}.scf.in"
    ph_in = dyn_dir / f"{name}.ph.in"
    for path in (scf_in, ph_in):
        if not path.exists():
            raise FileNotFoundError(path)

    atoms = ase.io.read(scf_in, format="espresso-in")
    atoms.rattle(stdev=args.stdev, seed=args.seed)

    prefix = f"{name}{SUFFIX}"
    out_dir = args.qe_inputs_dir / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    scf_text = relax_input_from_scf(scf_in.read_text(), atoms, prefix, args.forc_conv_thr)
    (out_dir / f"{prefix}.scf.in").write_text(scf_text)
    (out_dir / f"{prefix}.ph.in").write_text(renamed_ph_input(ph_in.read_text(), name, prefix))

    # Round-trip check: what pw.x will read is the rattled cell, nothing else changed.
    check = ase.io.read(out_dir / f"{prefix}.scf.in", format="espresso-in")
    assert np.allclose(check.cell, atoms.cell) and np.allclose(check.positions, atoms.positions, atol=1e-6)

    RATTLED_STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    ase.io.write(RATTLED_STRUCTURES_DIR / f"{prefix}.cif", atoms)
    worst = unstable[0]
    return f"{len(unstable)} unstable branch(es), worst {worst.frequency_cm1:.1f} cm-1 at {worst.q_label} -> {out_dir}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ph-root", type=Path, default=PH_FIRST_DIR)
    parser.add_argument("--scf-root", type=Path, default=SCF_FIRST_DIR)
    parser.add_argument("--qe-inputs-dir", type=Path, default=QE_INPUTS_DIR)
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these material names")
    parser.add_argument("--stdev", type=float, default=0.05, help="Rattle amplitude, Angstrom (default: 0.05, as run_mlip_phonons.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--imag-tol", type=float, default=1.0,
        help="Branches below -IMAG_TOL cm-1 count as imaginary (default: 1.0, as run_mlip_phonons.py)",
    )
    parser.add_argument(
        "--forc-conv-thr", type=float, default=1e-5,
        help="pw.x relax force threshold, Ry/Bohr (default: 1e-5, ~2.6e-4 eV/A)",
    )
    args = parser.parse_args()

    materials = run_names(args.ph_root, exclude_suffix=SUFFIX)
    if args.only:
        materials = [m for m in materials if m in set(args.only)]
    failures = 0
    for name in materials:
        try:
            print(f"{name}: {prepare_one(name, args)}")
        except Exception as exc:
            failures += 1
            print(f"{name}: FAILED -- {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
