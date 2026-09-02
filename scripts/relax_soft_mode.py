"""Freeze one or more unstable phonon modes from QE .dyn file(s) into a supercell,
relax it with a MACE foundation-model potential, and regenerate QE inputs.

This is the follow-up to a completed ph.x DFPT run that found imaginary
zone-boundary frequencies (octahedral tilting etc. -- see
src/soft_mode_distortion.py for why the original small cell can't relax into
that on its own): cheaply pre-relax a symmetry-broken supercell with MACE
before paying for another expensive QE run on it.

Give one --seed to freeze in a single mode (the common case). Give more than
one --seed, from different .dyn files (different q-points, same DFPT run) to
combine simultaneous instabilities into one supercell -- e.g. an R-point
out-of-phase tilt together with an M-point in-phase tilt, as in real Glazer
tilt systems. Each seed defaults to that q-point's most unstable mode and the
shared --amplitude; override either with :MODE_INDEX and/or :AMPLITUDE.

Instead of picking --seed(s) by hand, --auto-seed scans a directory of .dyn
files (via src/unstable_modes.py) and freezes in the strongest instabilities
it finds automatically -- see its own --auto-top/--auto-per-q options below.

Usage:
    # single mode, picked by hand (equivalent to the old single-.dyn-file interface)
    uv run python3 scripts/relax_soft_mode.py CsPbI3 \\
        --seed sherlock_outputs/ph_first/CsPbI3/CsPbI3.dyn10

    # two simultaneous instabilities at different q-points, different amplitudes
    uv run python3 scripts/relax_soft_mode.py CsPbI3 \\
        --seed sherlock_outputs/ph_first/CsPbI3/CsPbI3.dyn10:0:0.15 \\
        --seed sherlock_outputs/ph_first/CsPbI3/CsPbI3.dyn4:0:0.10

    # let it find the two strongest instabilities (one per q-point) itself
    uv run python3 scripts/relax_soft_mode.py CsPbI3 \\
        --auto-seed sherlock_outputs/ph_first/CsPbI3 --auto-top 2
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

from src.generate_qe_input import QE_INPUTS_DIR, generate_inputs_for_structure, pbesol_config_for
from src.soft_mode_distortion import (
    ModeSeed,
    build_multi_mode_distorted_supercell,
    most_unstable_mode,
    parse_dyn_file,
)
from src.unstable_modes import rank_unstable_modes, top_modes_per_q

TILTED_STRUCTURES_DIR = PROJECT_ROOT / "data" / "tilted_structures"


def relax_with_mace(atoms, fmax: float, steps: int):
    atoms.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    optimizer = LBFGS(FrechetCellFilter(atoms))
    optimizer.run(fmax=fmax, steps=steps)
    return atoms


def _parse_seed_arg(raw: str) -> tuple[Path, int | None, float | None]:
    """Parse one --seed argument: DYN_FILE[:MODE_INDEX[:AMPLITUDE]].

    MODE_INDEX and AMPLITUDE are optional and independently omittable (e.g.
    "file.dyn10::0.2" overrides only the amplitude); a bare path uses this
    q-point's most unstable mode and the shared --amplitude default.
    """
    parts = raw.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(f"Too many ':'-separated fields in --seed {raw!r}")
    parts += [""] * (3 - len(parts))
    path_str, mode_str, amplitude_str = parts
    mode_index = int(mode_str) if mode_str else None
    amplitude = float(amplitude_str) if amplitude_str else None
    return Path(path_str), mode_index, amplitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Base structure name, e.g. CsPbI3 (used for output filenames)")

    selection = parser.add_argument_group(
        "mode selection (give --seed one or more times, or --auto-seed -- at least one is required)"
    )
    selection.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        default=[],
        metavar="DYN_FILE[:MODE_INDEX[:AMPLITUDE]]",
        help=(
            "A phonon mode to freeze in: a .dyn file at the unstable q-point, "
            "optionally with :MODE_INDEX (default: most unstable mode at that "
            "q) and :AMPLITUDE (default: --amplitude). Repeat --seed, with "
            ".dyn files from different q-points of the same DFPT run, to "
            "combine simultaneous instabilities into one supercell."
        ),
    )
    selection.add_argument(
        "--auto-seed",
        type=Path,
        default=None,
        metavar="DYN_DIR",
        help=(
            "Instead of (or alongside) --seed, automatically pick the "
            "strongest instabilities from every .dyn file in DYN_DIR (see "
            "src/unstable_modes.py / scripts/find_unstable_modes.py)."
        ),
    )
    selection.add_argument(
        "--auto-top", type=int, default=None,
        help="With --auto-seed: keep only the N overall strongest instabilities (default: all found)",
    )
    selection.add_argument(
        "--auto-per-q", type=int, default=1,
        help="With --auto-seed: at most this many branches per distinct q-point (default: 1, the deepest at each)",
    )

    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Override the auto-computed supercell (smallest commensurate with every seed's q)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.15,
        help="Default max single-atom seed displacement (angstrom) for any seed without its own amplitude",
    )
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold, eV/angstrom")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if not args.seeds and args.auto_seed is None:
        parser.error("Provide at least one --seed or --auto-seed")

    mode_seeds = []
    for raw_seed in args.seeds:
        dyn_path, mode_index_override, amplitude_override = _parse_seed_arg(raw_seed)
        dyn = parse_dyn_file(dyn_path)
        mode_index = mode_index_override if mode_index_override is not None else most_unstable_mode(dyn)
        amplitude = amplitude_override if amplitude_override is not None else args.amplitude
        frequency = dyn.frequencies_cm1[mode_index]
        if frequency >= 0:
            print(
                f"Warning: mode {mode_index} at q={tuple(dyn.q_frac)} has frequency "
                f"{frequency:.2f} cm-1 (not unstable) -- continuing anyway."
            )
        print(f"Seeding mode {mode_index} ({frequency:.2f} cm-1) at q={tuple(dyn.q_frac)}, amplitude {amplitude} A")
        mode_seeds.append(ModeSeed(dyn=dyn, mode_index=mode_index, amplitude_angstrom=amplitude))

    if args.auto_seed is not None:
        unstable = rank_unstable_modes(args.auto_seed)
        if args.auto_per_q is not None:
            unstable = top_modes_per_q(unstable, n_per_q=args.auto_per_q)
        if args.auto_top is not None:
            unstable = unstable[: args.auto_top]
        if not unstable:
            parser.error(f"--auto-seed found no unstable branches in {args.auto_seed}")
        for mode in unstable:
            print(
                f"Auto-selected mode {mode.mode_index} at q={mode.q_label} "
                f"({mode.frequency_cm1:.2f} cm-1) from {mode.dyn_path.name}, amplitude {args.amplitude} A"
            )
            mode_seeds.append(ModeSeed(dyn=mode.dyn, mode_index=mode.mode_index, amplitude_angstrom=args.amplitude))

    supercell = tuple(args.supercell) if args.supercell is not None else None
    structure = build_multi_mode_distorted_supercell(mode_seeds, supercell=supercell)
    print(f"Built a {structure.get_global_number_of_atoms()}-atom supercell from {len(mode_seeds)} seed(s)")

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