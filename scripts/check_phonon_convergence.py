"""Generate and check convergence tests for QE SCF+DFPT phonon calculations.

Sweeps one parameter -- ecutwfc, kpts, tr2_ph, or qpts -- across several
values (see src/convergence.py for why ecutwfc/kpts/tr2_ph are cheap
Gamma-only runs while qpts must run the real dispersion each time), and
reports how much the computed frequencies move between consecutive values.

Usage:
    # 1. generate the sweep's SCF+DFPT inputs (written into data/qe_inputs/,
    #    under sweep-specific STRUCTURE names)
    uv run python3 scripts/check_phonon_convergence.py generate \\
        data/cubic_structures/CsPbI3.cif ecutwfc

    # 2. submit each printed sweep name's SCF then DFPT job -- directly, NOT
    #    through submit_all.sh: skip the MACE-relax step here, since a
    #    convergence test needs the *same* starting geometry at every sweep
    #    point, with only the parameter under test changing.
    #        sbatch --export=ALL,STRUCTURE=<name>,PROJECT_ROOT=$(pwd) run_scf.sbatch
    #        sbatch --export=ALL,STRUCTURE=<name>,PROJECT_ROOT=$(pwd) run_ph.sbatch
    #    (repeat per sweep name; run_ph.sbatch after its matching run_scf.sbatch finishes)

    # 3. once every sweep point has finished, sync
    #    $SCRATCH/perovskite_phonon/ph_first/ locally and check convergence
    uv run python3 scripts/check_phonon_convergence.py check \\
        CsPbI3 ecutwfc /path/to/synced/ph_first
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from convergence import STANDARD_SWEEPS, check_convergence, generate_convergence_test, read_convergence_sweep
from generate_qe_input import pbesol_config_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Write the SCF+DFPT inputs for one convergence sweep")
    gen.add_argument("structure", type=Path, help="Structure file, e.g. data/cubic_structures/CsPbI3.cif")
    gen.add_argument("parameter", choices=sorted(STANDARD_SWEEPS), help="Which parameter to sweep")

    chk = subparsers.add_parser("check", help="Report frequency convergence from a completed sweep")
    chk.add_argument("structure_name", help="Base structure name the sweep was generated for, e.g. CsPbI3")
    chk.add_argument("parameter", choices=sorted(STANDARD_SWEEPS))
    chk.add_argument(
        "results_root", type=Path,
        help="Local directory holding each sweep point's ph.x output (synced from $SCRATCH/perovskite_phonon/ph_first/)",
    )
    chk.add_argument("--tol", type=float, default=1.0, help="Convergence threshold, cm-1 (default: 1.0)")

    args = parser.parse_args()

    if args.command == "generate":
        parameter = STANDARD_SWEEPS[args.parameter]
        sweep_names = generate_convergence_test(args.structure, parameter, base_config=pbesol_config_for)
        print(f"Wrote {len(sweep_names)} sweep points for '{parameter.name}' into data/qe_inputs/:")
        for name in sweep_names:
            print(f"  {name}")
        print(
            "\nSubmit each one directly (skip submit_all.sh's MACE-relax step -- "
            "every sweep point should start from the same fixed geometry):\n"
            "  sbatch --export=ALL,STRUCTURE=<name>,PROJECT_ROOT=$(pwd) run_scf.sbatch\n"
            "  sbatch --export=ALL,STRUCTURE=<name>,PROJECT_ROOT=$(pwd) run_ph.sbatch\n"
            "Then sync $SCRATCH/perovskite_phonon/ph_first/ locally and run 'check'."
        )

    elif args.command == "check":
        points = read_convergence_sweep(args.structure_name, args.parameter, args.results_root)
        rows = check_convergence(points, tol_cm1=args.tol)

        print(f"{'value':>10}  {'max |dfreq|, cm-1':>18}  converged?")
        for row in rows:
            diff_str = f"{row['max_diff_cm1']:.3f}" if row["max_diff_cm1"] is not None else "--"
            flag = "<-- converged from here" if row["converged_from_here"] else ""
            print(f"{row['formatted_value']:>10}  {diff_str:>18}  {flag}")
            if row["per_q"] and len(row["per_q"]) > 1:
                breakdown = ", ".join(f"{label}: {d:.3f}" for label, d in sorted(row["per_q"].items()))
                print(f"{'':>10}  {'':>18}  ({breakdown})")

        if not any(row["converged_from_here"] for row in rows):
            print(f"\nNot converged within tol={args.tol} cm-1 across this sweep -- consider extending it.")


if __name__ == "__main__":
    main()