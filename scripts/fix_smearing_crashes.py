"""Find and fix Quantum ESPRESSO SCF runs that crashed with QE's

    charge is wrong: smearing is needed

by regenerating those structures' SCF+DFPT inputs with occupations='smearing'
instead of the default 'fixed' -- see src/scf_diagnostics.py for why this
error shows up (predominantly for open d/f-shell B-site ions: Co, Cu, Mn, Cr,
Eu, Sm, ...) and what it does and doesn't fix.

Usage:
    # auto-detect from existing SCF logs, then regenerate in place
    uv run python3 scripts/fix_smearing_crashes.py \\
        --scf-dir /scratch/users/juntunen/perovskite_phonon/scf_first

    # or supply structure names directly (e.g. from your own directory scan) --
    # useful to double-check a name really hit *this* error before fixing it
    uv run python3 scripts/fix_smearing_crashes.py \\
        --names CsCoBr3 CsCuCl3 CsEuF3 CsMnI3

    # see what would be touched without writing anything
    uv run python3 scripts/fix_smearing_crashes.py --scf-dir ... --dry-run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.generate_qe_input import QE_INPUTS_DIR
from src.paths import STRUCTURES_DIR
from src.scf_diagnostics import find_scf_charge_errors, regenerate_with_smearing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scf-dir", type=Path, default=None,
        help="Root of scf_first/ output to scan for the error (one subdirectory per structure)",
    )
    parser.add_argument("--names", nargs="+", default=None, help="Structure names to fix directly, skipping the log scan")
    parser.add_argument(
        "--structures-dir", type=Path, default=None,
        help="Directory holding <name>.cif to rebuild from (default: STRUCTURES_DIR, the pristine "
        "un-relaxed cubic cell). Pass relaxed_structures/ (or tilted_structures/ for a _tilted name) "
        "so the regenerated input keeps the already-relaxed geometry instead of reverting it -- see "
        "orchestrate_pipeline.py's regenerate_smearing().",
    )
    parser.add_argument("--degauss", type=float, default=0.01, help="Smearing width, Ry (default: 0.01, QEInputConfig's default)")
    parser.add_argument("--smearing", default="gaussian", help="Smearing type (default: gaussian)")
    parser.add_argument("--dry-run", action="store_true", help="Print which structures would be fixed without writing anything")
    parser.add_argument(
        "--metallic", action="store_true",
        help="Structures are genuinely metallic once smearing lets SCF converge -- also set "
        "DFPT epsil=False, zeu=False (Born charges + LO-TO splitting are only valid for insulators)",
    )
    args = parser.parse_args()

    if args.names:
        names = args.names
    elif args.scf_dir:
        names = find_scf_charge_errors(args.scf_dir)
        print(f"Found {len(names)} structure(s) with the 'charge is wrong: smearing is needed' error in {args.scf_dir}")
    else:
        parser.error("Provide either --scf-dir (to scan logs) or --names (explicit list)")

    if not names:
        print("Nothing to fix.")
        return

    for name in names:
        print(f"  {name}")

    if args.dry_run:
        print("\n--dry-run: not writing anything.")
        return

    done = regenerate_with_smearing(
        names, args.structures_dir or STRUCTURES_DIR, QE_INPUTS_DIR, smearing=args.smearing, degauss=args.degauss,
        epsil=not args.metallic, zeu=not args.metallic,
    )
    print(f"\nRegenerated {len(done)}/{len(names)} structures' SCF+DFPT inputs with occupations='smearing' (degauss={args.degauss} Ry).")
    if args.metallic:
        print("DFPT inputs regenerated with epsil=False, zeu=False (metallic). Resubmit just their DFPT (ph-*) jobs -- SCF is unaffected.")
    else:
        print("Resubmit their SCF (then DFPT) jobs the same way as before -- same STRUCTURE names, same run_scf.sbatch/run_ph.sbatch.")
        print(
            "\nNote: if any of these turn out genuinely metallic once smearing lets SCF finish, "
            "the DFPT step's epsil/zeu (Born charges + LO-TO splitting) are only valid for "
            "insulators -- rerun with --metallic for those."
        )


if __name__ == "__main__":
    main()