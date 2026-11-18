"""Soft-mode arrow diagram + animation and the dispersion (+ vDOS) figure for one
structure's completed ph.x DFPT run -- the figures are src/utils/visualize.py's.

Usage:
    uv run python3 scripts/qe/visualize_modes.py -s CsPbI3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.paths import CUBIC_STRUCTURES_DIR, PH_FIRST_DIR, QE_OUT_DIR
from src.qe.soft_mode_distortion import parse_dyn_file
from src.utils.plotting import use_project_style
from src.utils.visualize import animate_mode_gif, plot_dispersion, plot_mode_arrows


def main() -> None:
    use_project_style()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--s", type=str, default="CsCuCl3", help="Structure name (used for output filenames)")
    parser.add_argument("--cif", type=str, default=str(CUBIC_STRUCTURES_DIR), help="path to CIF structures")
    parser.add_argument("--out", type=str, default=str(QE_OUT_DIR), help="Output directory")
    parser.add_argument("--dyn", type=str, default=str(PH_FIRST_DIR), help="path to QE .dynN file for the q-point of interest")
    parser.add_argument("--freq", type=str, default=str(PH_FIRST_DIR), help="path to matdyn.x .freq file for the dispersion plot")
    parser.add_argument("--dos", type=str, default=str(PH_FIRST_DIR), help="path to matdyn.x .dos file for the vDOS panel")
    parser.add_argument("-q", "--q", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="q-point (qx, qy, qz) in the units the .dynN file uses (2*pi/alat)")

    args = parser.parse_args()
    cif_file = Path(args.cif) / f"{args.s}.cif"
    dyn_file = Path(args.dyn) / f"{args.s}" / f"{args.s}.dyn1"
    freq_file = Path(args.freq) / f"{args.s}" / f"{args.s}.freq"
    dos_file = Path(args.dos) / f"{args.s}" / f"{args.s}.dos"

    dyn = parse_dyn_file(dyn_file)
    if not np.allclose(dyn.q_frac, args.q, atol=1e-4):
        raise ValueError(f"{dyn_file}: q-point mismatch, file has {tuple(dyn.q_frac)}, expected {tuple(args.q)}")
    freqs, vectors = dyn.frequencies_cm1, list(dyn.eigenvectors.real)
    print("Gamma-point frequencies (cm-1):", freqs)
    lowest = int(np.argmin(freqs))
    plot_mode_arrows(cif_file, vectors, lowest, freqs, out_path=args.out + "/gamma_softmode_arrows.png")
    animate_mode_gif(cif_file, vectors, lowest, freqs, out_path=args.out + "/gamma_softmode.gif")

    plot_dispersion(
        freq_file,
        labels=["G", "X", "M", "G", "R", "X"],
        dos_path=dos_file if dos_file.exists() else None,
        out_path=args.out + f"/{args.s}_dispersion.png",
    )


if __name__ == "__main__":
    main()
