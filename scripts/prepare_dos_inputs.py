"""Add q2r.x + matdyn.x (vDOS) inputs to every structure that has a completed,
physically meaningful phonon dispersion run.

'Physically meaningful' means: the cubic aristotype cell itself, if its own
DFPT dispersion came out fully real (no unstable zone-boundary modes) -- or,
if it didn't, the corresponding <name>_tilted structure that
scripts/batch_relax_soft_mode.py already builds by freezing in the strongest
instabilities and relaxing with MACE. A vDOS computed on a dynamically
unstable cubic cell isn't the spectrum of anything that exists; the tilted,
relaxed structure is the one worth scoring against a vDOS target.

This script only decides *which* directory's completed DFPT run to build DOS
inputs from and writes q2r.in/matdyn_dos.in there -- it does not invoke
q2r.x/matdyn.x itself. Run those on the cluster the same way
run_scf.sbatch/run_ph.sbatch are already run for each STRUCTURE name (wire up
run_q2r.sbatch/run_matdyn.sbatch analogously; they aren't part of this
script).

Usage:
    # first pass: writes DOS inputs for every cubic-stable structure, and
    # reports which unstable ones are still waiting on a tilted-cell DFPT run
    uv run python3 scripts/prepare_dos_inputs.py \\
        --cubic-ph-root sherlock_outputs/ph_first \\
        --qe-inputs-dir data/qe_inputs

    # after batch_relax_soft_mode.py + a tilted-cell run_scf/run_ph pass has
    # finished for some of the previously-pending materials, rerun the same
    # command -- already-done structures are just overwritten harmlessly.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.paths import PH_FIRST_DIR, QE_INPUTS_DIR
from src.phonon_dos import generate_dos_inputs
from src.unstable_modes import rank_unstable_modes


def _unstable_status(dyn_dir: Path) -> bool | None:
    """True/False, or None if this material's DFPT run hasn't completed yet
    (no numbered .dyn files found) -- kept distinct from False so the caller
    can report "not ready" rather than silently treating it as stable."""
    try:
        modes = rank_unstable_modes(dyn_dir)
    except FileNotFoundError:
        return None
    return bool(modes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cubic-ph-root", type=Path, default=PH_FIRST_DIR,
        help="ph_first/ root for the original cubic-cell DFPT runs (one subdirectory per structure name)",
    )
    parser.add_argument(
        "--tilted-ph-root", type=Path, default=None,
        help="ph_first/ root for <name>_tilted DFPT runs (default: same as --cubic-ph-root)",
    )
    parser.add_argument(
        "--qe-inputs-dir", type=Path, default=QE_INPUTS_DIR,
        help="qe_inputs/ -- where each structure's *.scf.in/*.ph.in (and now *.matdyn_dos.in) live",
    )
    parser.add_argument("--nk", type=int, nargs=3, default=(20, 20, 20), help="Dense q-mesh for the DOS interpolation (default: 20x20x20)")
    parser.add_argument("--deltaE", type=float, default=0.5, help="DOS bin width, cm-1 (default: 0.5)")
    args = parser.parse_args()

    print(args.cubic_ph_root)
    tilted_ph_root = args.tilted_ph_root or args.cubic_ph_root

    ready, pending_tilt, unstable_after_tilt, not_done = [], [], [], []
    materials = sorted(
        d.name for d in args.cubic_ph_root.iterdir()
        if d.is_dir() and not d.name.endswith("_tilted")
    )

    for name in materials:
        cubic_unstable = _unstable_status(args.cubic_ph_root / name)
        if cubic_unstable is None:
            not_done.append(name)
            continue

        if not cubic_unstable:
            out_dir = args.qe_inputs_dir / name
            generate_dos_inputs(out_dir, name, nk=tuple(args.nk), delta_e_cm1=args.deltaE)
            ready.append(name)
            continue

        tilted_name = f"{name}_tilted"
        tilted_unstable = _unstable_status(tilted_ph_root / tilted_name)
        if tilted_unstable is None:
            pending_tilt.append(name)
            continue

        out_dir = args.qe_inputs_dir / tilted_name
        generate_dos_inputs(out_dir, tilted_name, nk=tuple(args.nk), delta_e_cm1=args.deltaE)
        ready.append(tilted_name)
        if tilted_unstable:
            unstable_after_tilt.append(tilted_name)

    print(f"DOS inputs written for {len(ready)} structure(s):")
    for name in ready:
        print(f"  {name}")

    if unstable_after_tilt:
        print(
            f"\nWARNING: {len(unstable_after_tilt)} structure(s) are STILL dynamically "
            "unstable even after soft-mode relaxation -- DOS inputs were written anyway, "
            "but their vDOS won't be that of a true minimum. Treat these as a stronger "
            "synthesizability red flag than an unstable *cubic* cell (see "
            "vdos_features.py's module docstring), consider more/different --seed(s) in "
            "relax_soft_mode.py, or exclude them from downstream scoring:"
        )
        for name in unstable_after_tilt:
            print(f"  {name}")

    if pending_tilt:
        print(
            f"\n{len(pending_tilt)} material(s) are unstable in the cubic cell but don't yet "
            "have a completed tilted-cell DFPT run (run scripts/batch_relax_soft_mode.py, then "
            "submit run_scf.sbatch/run_ph.sbatch for <name>_tilted, before retrying this script):"
        )
        for name in pending_tilt:
            print(f"  {name}")

    if not_done:
        print(f"\n{len(not_done)} material(s) have no completed cubic DFPT output yet, skipped:")
        for name in not_done:
            print(f"  {name}")


if __name__ == "__main__":
    main()
