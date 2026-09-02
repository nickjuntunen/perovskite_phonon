"""Scan a directory of Quantum ESPRESSO ph.x .dyn files and report the
strongest phonon instabilities, ranked most-negative-frequency first.

Usage:
    uv run python3 scripts/find_unstable_modes.py sherlock_outputs/ph_first/CsPbI3

    # only the single deepest branch at each distinct q-point:
    uv run python3 scripts/find_unstable_modes.py sherlock_outputs/ph_first/CsPbI3 --per-q 1

    # the 2 strongest instabilities overall, formatted as relax_soft_mode.py
    # --seed arguments ready to paste in (or pipe straight into --auto-seed,
    # which does this same selection automatically):
    uv run python3 scripts/find_unstable_modes.py sherlock_outputs/ph_first/CsPbI3 \\
        --per-q 1 --top 2 --as-seeds
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from unstable_modes import as_seed_arg, rank_unstable_modes, summarize_by_q, top_modes_per_q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dyn_dir", type=Path, help="Directory containing .dyn files from one DFPT run")
    parser.add_argument("--pattern", default="*.dyn*", help="Glob pattern for .dyn files within dyn_dir")
    parser.add_argument("--freq-tol", type=float, default=-1e-3, help="Only branches below this frequency (cm-1) count as unstable")
    parser.add_argument("--per-q", type=int, default=None, help="Report only the N most unstable branches at each distinct q (default: all)")
    parser.add_argument("--top", type=int, default=None, help="After --per-q grouping (if given), keep only the overall top N")
    parser.add_argument("--amplitude", type=float, default=0.15, help="Amplitude to use when formatting --as-seeds output")
    parser.add_argument("--as-seeds", action="store_true", help="Print relax_soft_mode.py --seed arguments instead of a table")
    args = parser.parse_args()

    all_modes = rank_unstable_modes(args.dyn_dir, pattern=args.pattern, freq_tol_cm1=args.freq_tol)
    if not all_modes:
        print(f"No unstable branches found (tol={args.freq_tol} cm-1) in {args.dyn_dir}")
        return

    selected = all_modes
    if args.per_q is not None:
        selected = top_modes_per_q(selected, n_per_q=args.per_q)
    if args.top is not None:
        selected = selected[: args.top]

    if args.as_seeds:
        for mode in selected:
            print(f"--seed {as_seed_arg(mode, args.amplitude)}")
        return

    print(f"{len(selected)} of {len(all_modes)} unstable branch(es) found in {args.dyn_dir}:\n")
    print(f"{'q':>12}  {'mode':>4}  {'freq (cm-1)':>12}  file")
    for mode in selected:
        print(f"{mode.q_label:>12}  {mode.mode_index:>4}  {mode.frequency_cm1:>12.2f}  {mode.dyn_path.name}")

    print("\nBy q-point (all unstable branches, not just those selected above):")
    for label, group in summarize_by_q(all_modes).items():
        print(f"  {label:>8}: {len(group)} branch(es), deepest {group[0].frequency_cm1:.2f} cm-1")


if __name__ == "__main__":
    main()