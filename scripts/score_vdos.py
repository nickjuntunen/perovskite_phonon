"""Score every structure with a completed vDOS against a feature-based target
(a phonon gap, or a specific mode frequency) and rank by fitness.

Usage:
    # target: a phonon gap centered near 150 cm-1, ~40 cm-1 wide
    uv run python3 scripts/score_vdos.py --dos-root sherlock_outputs/dos \\
        --target gap --center 150 --width 40

    # target: a single (e.g. soft/tilt) mode near 45 cm-1
    uv run python3 scripts/score_vdos.py --dos-root sherlock_outputs/dos \\
        --target mode --freq 45
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phonon_dos import parse_matdyn_dos
from vdos_features import extract_features, gap_fitness, mode_fitness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dos-root", type=Path, required=True,
        help="Directory holding each structure's <name>.dos (synced matdyn.x fldos output; see scripts/prepare_dos_inputs.py)",
    )
    parser.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "data" / "vdos_scores.csv")
    parser.add_argument("--target", choices=["gap", "mode"], required=True)
    parser.add_argument("--center", type=float, help="Target gap center, cm-1 (--target gap)")
    parser.add_argument("--width", type=float, help="Target gap width, cm-1 (--target gap)")
    parser.add_argument("--freq", type=float, help="Target mode frequency, cm-1 (--target mode)")
    parser.add_argument("--dos-tol", type=float, default=1e-3, help="DOS threshold below which a frequency window counts as 'empty' for gap detection")
    parser.add_argument("--peak-prominence", type=float, default=1e-3)
    args = parser.parse_args()

    if args.target == "gap" and (args.center is None or args.width is None):
        parser.error("--target gap requires --center and --width")
    if args.target == "mode" and args.freq is None:
        parser.error("--target mode requires --freq")

    rows = []
    for dos_path in sorted(Path(args.dos_root).glob("*.dos")):
        name = dos_path.stem
        freq_cm1, dos = parse_matdyn_dos(dos_path)
        features = extract_features(freq_cm1, dos, dos_tol=args.dos_tol, peak_prominence=args.peak_prominence)

        fitness = (
            gap_fitness(features, args.center, args.width)
            if args.target == "gap"
            else mode_fitness(features, args.freq)
        )

        rows.append({
            "name": name,
            "fitness": fitness,
            "has_imaginary": features.has_imaginary,
            "imaginary_fraction": round(features.imaginary_fraction, 5),
            "gap_low_cm1": features.gap_low_cm1,
            "gap_high_cm1": features.gap_high_cm1,
            "gap_width_cm1": features.gap_width_cm1,
        })

    if not rows:
        print(f"No *.dos files found in {args.dos_root} -- has scripts/prepare_dos_inputs.py been run, and q2r.x/matdyn.x submitted?")
        return

    rows.sort(key=lambda r: r["fitness"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} structure(s), wrote {args.out_csv}")
    print("\nTop 10 closest to target:")
    for row in rows[:10]:
        flag = "  [IMAGINARY MODES PRESENT]" if row["has_imaginary"] else ""
        fitness_str = f"{row['fitness']:.2f}" if row["fitness"] != float("inf") else "inf (no gap/peak found)"
        print(f"  {row['name']:>20}  fitness={fitness_str}{flag}")


if __name__ == "__main__":
    main()
