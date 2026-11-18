"""Plot MLIP phonon dispersions + vDOS straight from scripts/mlip/run_mlip_phonons.py
output -- no QE reference needed (for that, see scripts/mlip/compare_phonons.py).

One figure per structure, every model found under --mlip-root overlaid, on
the ASE standard path each run saved in band.npz. Models are only overlaid
when their paths agree (same lattice, same q-points); a model whose run used
a different cell setting is plotted in its own figure instead.

Usage:
    uv run python3 scripts/mlip/plot_mlip_phonons.py                  # every structure
    uv run python3 scripts/mlip/plot_mlip_phonons.py CsPbI3 --mlip-root data/mlip_phonons
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paths import MLIP_PHONONS_DIR
from src.mlip.phonon_results import discover_models, load_band_run
from src.qe.vdos_features import unit_area_smoothed
from src.utils.plotting import GUIDE_LINE, ZERO_LINE, grid_shape, series_colors, use_project_style

use_project_style()

def plot(name: str, runs: dict[str, dict], colors: dict[str, str], sigma: float, out_path: Path) -> None:
    """One panel per model -- dispersion + vDOS -- on a shared frequency axis."""
    labels = list(runs)
    nrows, ncols = grid_shape(len(labels))
    fig = plt.figure(figsize=(4.6 * ncols, 3.4 * nrows + 0.6))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.18)

    all_freqs = np.concatenate([r["freqs"].ravel() for r in runs.values()])
    pad = 0.05 * (all_freqs.max() - all_freqs.min())
    ylim = (all_freqs.min() - pad, all_freqs.max() + pad)

    for k, label in enumerate(labels):
        run = runs[label]
        inner = outer[k // ncols, k % ncols].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
        ax_band = fig.add_subplot(inner[0])
        ax_dos = fig.add_subplot(inner[1], sharey=ax_band)

        ax_band.plot(run["x"], run["freqs"], color=colors[label], lw=1.1)
        ax_band.axhline(0, color=ZERO_LINE, lw=0.6)
        for x in run["special_x"]:
            ax_band.axvline(x, color=GUIDE_LINE, lw=0.6)
        ax_band.set_xticks(run["special_x"], run["labels"], fontsize=8)
        ax_band.set_xlim(run["x"][0], run["x"][-1])
        ax_band.set_ylim(*ylim)
        ax_band.tick_params(axis="y", labelsize=8)
        if k % ncols == 0:
            ax_band.set_ylabel("Frequency (cm$^{-1}$)", fontsize=9)
        else:
            ax_band.tick_params(axis="y", labelleft=False)
        ax_band.set_title(label, fontsize=8, loc="left")

        if run["dos"] is not None:
            freq, g = run["dos"]
            ax_dos.plot(unit_area_smoothed(freq, g, sigma), freq, color=colors[label], lw=1.0)
        ax_dos.axhline(0, color=ZERO_LINE, lw=0.6)
        ax_dos.set_xticks([])
        ax_dos.set_xlim(left=0)
        ax_dos.tick_params(axis="y", labelleft=False)

    fig.suptitle(name, y=0.995)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structures", nargs="*", help="Default: every structure with at least one model run")
    parser.add_argument("--mlip-root", type=Path, default=MLIP_PHONONS_DIR)
    parser.add_argument("--models", nargs="*", default=None, help="Model labels (subdirectories of --mlip-root); default: all")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <mlip-root>/plots")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian broadening for the vDOS panel, cm-1 (default: 2)")
    args = parser.parse_args()

    models = args.models or discover_models(args.mlip_root)
    try:
        colors = series_colors(models)
    except ValueError as exc:
        parser.error(f"{exc}; pass --models")
    structures = args.structures or sorted({
        d.name for m in models for d in (args.mlip_root / m).iterdir() if (d / "band.npz").exists()
    })
    out_dir = args.out_dir or args.mlip_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in structures:
        runs = {m: r for m in models if (r := load_band_run(args.mlip_root / m / name, name)) is not None}
        if not runs:
            print(f"{name}: no MLIP runs found, skipped")
            continue
        # Group models by q-path so only like-for-like dispersions share axes.
        groups: list[dict[str, dict]] = []
        for label, run in runs.items():
            for group in groups:
                other = next(iter(group.values()))
                if other["q_frac"].shape == run["q_frac"].shape and np.allclose(other["q_frac"], run["q_frac"]):
                    group[label] = run
                    break
            else:
                groups.append({label: run})
        for i, group in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"_{i + 1}"
            out_path = out_dir / f"{name}_mlip_phonons{suffix}.png"
            plot(name, group, colors, args.sigma, out_path)
            print(f"{name}: {', '.join(group)} -> {out_path}")


if __name__ == "__main__":
    main()
