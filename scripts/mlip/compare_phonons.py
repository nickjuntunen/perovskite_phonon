"""Benchmark MLIP phonon dispersions + vDOS against QE DFPT.

Reference (QE), per structure <name> under --qe-ph-root/<name>/ (the layout
run_ph.sbatch leaves behind):
    <name>.fc        q2r.x force constants (cell, masses -- needed to map q-points)
    <name>.freq      matdyn.x dispersion on the band path (required)
    matdyn.in        that path's vertices (optional; for axis labels)
    <name>.dos       matdyn.x vDOS (optional; see --matdyn-x to generate it)

MLIP, per model under --mlip-root/<model>/<name>/, from scripts/mlip/run_mlip_phonons.py:
    phonopy_params.yaml, <name>.dos, meta.json

The MLIP force constants are re-evaluated at exactly QE's q-points, so the
dispersion comparison is point-for-point with no interpolation between codes.
See src/analysis/phonon_benchmark.py for what each metric means.

Output in --out-dir:
    metrics.csv                 one row per (model, structure)
    summary.csv                 per-model means over structures
    <name>_benchmark.png        dispersion + vDOS overlay, all models
    parity.png                  MLIP vs QE frequency, every q/branch/structure
    band_mae.png                dispersion MAE per structure and model
    metrics_overview.png        bias, vDOS shape and Cv/S/F errors per structure and model

Usage:
    uv run python3 scripts/mlip/compare_phonons.py \\
        --qe-ph-root sherlock_outputs/ph_first \\
        --mlip-root data/mlip_phonons \\
        --matdyn-x ~/software/qe-7.5/bin/matdyn.x   # fill in missing QE vDOS
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
import pandas as pd

from src.paths import DATA_DIR, MLIP_PHONONS_DIR
from src.analysis.phonon_benchmark import band_metrics, dos_metrics, load_mlip_on_reference_path, load_qe_reference
from src.mlip.phonon_results import discover_models
from src.qe.vdos_features import smooth_dos, unit_area_smoothed
from src.utils.plotting import (
    GUIDE_LINE,
    REFERENCE_COLOR,
    ZERO_LINE,
    grid_shape,
    series_colors,
    use_project_style,
)

use_project_style()

def _grid(n: int, cell_w: float, cell_h: float):
    nrows, ncols = grid_shape(n)
    return nrows, ncols, (cell_w * ncols, cell_h * nrows)


def plot_structure(name: str, qe: dict, mlips: dict[str, dict], colors: dict[str, str], out_path: Path, sigma: float) -> None:
    """One panel per model -- dispersion (left) + vDOS (right) -- each with QE
    underneath in neutral ink, sharing one frequency axis."""
    labels = list(mlips)
    nrows, ncols, figsize = _grid(len(labels), 4.6, 3.4)
    fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.6))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.18)

    qe_dos = qe["dos"]
    if qe_dos is not None:
        qe_dos = (qe_dos[0], smooth_dos(qe_dos[0], np.clip(qe_dos[1], 0, None), sigma_cm1=sigma))
    all_freqs = np.concatenate([qe["freqs"].ravel()] + [m["freqs"].ravel() for m in mlips.values()])
    pad = 0.05 * (all_freqs.max() - all_freqs.min())
    ylim = (all_freqs.min() - pad, all_freqs.max() + pad)

    x = qe["x"]
    for k, label in enumerate(labels):
        m = mlips[label]
        inner = outer[k // ncols, k % ncols].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
        ax_band = fig.add_subplot(inner[0])
        ax_dos = fig.add_subplot(inner[1], sharey=ax_band)

        ax_band.plot(x, qe["freqs"], color=REFERENCE_COLOR, lw=1.3)
        ax_band.plot(x, m["freqs"], color=colors[label], lw=1.1)
        ax_band.axhline(0, color=ZERO_LINE, lw=0.6)
        for tick_x, _ in qe["ticks"]:
            ax_band.axvline(tick_x, color=GUIDE_LINE, lw=0.6)
        if qe["ticks"]:
            ax_band.set_xticks([t[0] for t in qe["ticks"]], [t[1] for t in qe["ticks"]], fontsize=8)
        ax_band.set_xlim(x[0], x[-1])
        ax_band.set_ylim(*ylim)
        ax_band.tick_params(axis="y", labelsize=8)
        if k % ncols == 0:
            ax_band.set_ylabel("Frequency (cm$^{-1}$)", fontsize=9)
        else:
            ax_band.tick_params(axis="y", labelleft=False)
        mae = np.abs(m["freqs"] - qe["freqs"]).mean()
        ax_band.set_title(f"{label}  (MAE {mae:.1f} cm$^{{-1}}$)", fontsize=8, loc="left")

        for dos, color in ((qe_dos, REFERENCE_COLOR), (m["dos"], colors[label])):
            if dos is None:
                continue
            freq, g = dos
            g = g / np.trapezoid(g, freq) if color == REFERENCE_COLOR else unit_area_smoothed(freq, g, sigma)
            ax_dos.plot(g, freq, color=color, lw=1.0)
        ax_dos.axhline(0, color=ZERO_LINE, lw=0.6)
        ax_dos.set_xticks([])
        ax_dos.set_xlim(left=0)
        ax_dos.tick_params(axis="y", labelleft=False)

    handles = [plt.Line2D([], [], color=REFERENCE_COLOR, lw=1.3, label="QE DFPT (every panel)")]
    fig.suptitle(name, y=0.995)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.995), fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_parity(pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]], colors: dict[str, str], out_path: Path) -> None:
    labels = list(pairs)
    nrows, ncols, figsize = _grid(len(labels), 3.4, 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    all_vals = np.concatenate([np.concatenate([np.ravel(a) for pair in v for a in pair]) for v in pairs.values()])
    lo, hi = all_vals.min(), all_vals.max()
    pad = 0.05 * (hi - lo)
    for k, label in enumerate(labels):
        ax = axes[k // ncols, k % ncols]
        ref = np.concatenate([np.ravel(r) for r, _ in pairs[label]])
        mlip = np.concatenate([np.ravel(m) for _, m in pairs[label]])
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ZERO_LINE, lw=0.8)
        ax.scatter(ref, mlip, s=6, color=colors[label], alpha=0.4, linewidths=0)
        mae = np.abs(mlip - ref).mean()
        ax.set_title(f"{label}\nMAE {mae:.1f} cm$^{{-1}}$", fontsize=8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        if k // ncols == nrows - 1 or k + ncols >= len(labels):
            ax.set_xlabel("QE DFPT (cm$^{-1}$)", fontsize=9)
            ax.tick_params(labelbottom=True)
        if k % ncols == 0:
            ax.set_ylabel("MLIP (cm$^{-1}$)", fontsize=9)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _grouped_bars(ax, metrics: pd.DataFrame, column: str, colors: dict[str, str]) -> None:
    table = metrics.pivot(index="structure", columns="model", values=column)
    models = [m for m in colors if m in table.columns]
    width = 0.8 / len(models)
    base = np.arange(len(table))
    for i, model in enumerate(models):
        ax.bar(base + (i - (len(models) - 1) / 2) * width, table[model].values, width * 0.9,
               color=colors[model], label=model)
    ax.axhline(0, color=ZERO_LINE, lw=0.6)
    ax.set_xticks(base, table.index, rotation=45, ha="right")
    ax.grid(axis="y", color="#e4e4df", lw=0.6)
    ax.set_axisbelow(True)


def plot_band_mae(metrics: pd.DataFrame, colors: dict[str, str], out_path: Path) -> None:
    n_struct, n_model = metrics["structure"].nunique(), metrics["model"].nunique()
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * n_struct * max(1, n_model / 2) + 2), 3.8))
    _grouped_bars(ax, metrics, "band_mae_cm1", colors)
    ax.set_ylabel("Dispersion MAE vs QE (cm$^{-1}$)")
    ax.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(n_model, 3))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# (column, y-label) for the vDOS/thermodynamics overview -- one metric per
# panel, each on its own axis, since they don't share units.
OVERVIEW_PANELS = [
    ("band_mean_signed_err_cm1", "Dispersion bias (cm$^{-1}$)\n$<0$: MLIP too soft"),
    ("dos_wasserstein_cm1", "vDOS Wasserstein (cm$^{-1}$)"),
    ("dos_overlap", "vDOS overlap (1 = identical)"),
    ("s_kB_err", "$\\Delta S$ (k$_B$/atom)"),
    ("cv_kB_err", "$\\Delta C_V$ (k$_B$/atom)"),
    ("f_meV_err", "$\\Delta F$ (meV/atom)"),
]


def plot_overview(metrics: pd.DataFrame, colors: dict[str, str], temperature: float, out_path: Path) -> None:
    """Small multiples: one panel per metric, MLIP minus QE for the
    thermodynamic errors (so 0 = perfect), at `temperature`."""
    panels = [(col, label) for col, label in OVERVIEW_PANELS if col in metrics.columns]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (column, label) in zip(axes.flat, panels):
        _grouped_bars(ax, metrics, column, colors)
        ax.set_ylabel(label, fontsize=9)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"MLIP vs QE DFPT (thermodynamics at {temperature:.0f} K)", y=0.99, fontsize=11)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(len(labels), 4),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


SUMMARY_COLUMNS = [
    "band_mae_cm1", "band_rmse_cm1", "band_max_abs_err_cm1", "band_mean_signed_err_cm1",
    "dos_wasserstein_cm1", "dos_overlap", "cv_kB_err", "s_kB_err", "f_meV_err", "zpe_meV_err",
    "stability_agrees", "force_eval_wall_time_s",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qe-ph-root", type=Path, required=True)
    parser.add_argument("--mlip-root", type=Path, default=MLIP_PHONONS_DIR)
    parser.add_argument("--models", nargs="*", default=None, help="Model labels (subdirectories of --mlip-root); default: all")
    parser.add_argument("--structures", nargs="*", default=None, help="Default: every structure with QE and >= 1 MLIP result")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "phonon_benchmark")
    parser.add_argument("--matdyn-x", type=Path, default=None, help="matdyn.x binary, to generate missing QE <name>.dos files")
    parser.add_argument("--nk", type=int, nargs=3, default=(20, 20, 20), help="q-mesh for generated QE vDOS (default: 20 20 20)")
    parser.add_argument("--deltaE", type=float, default=0.5, help="Bin width for generated QE vDOS, cm-1 (default: 0.5)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian broadening for vDOS comparison, cm-1 (default: 2)")
    parser.add_argument("--temperature", type=float, default=300.0, help="K, for Cv/S/F (default: 300)")
    args = parser.parse_args()

    models = args.models or discover_models(args.mlip_root)
    try:
        colors = series_colors(models)
    except ValueError as exc:
        parser.error(f"{exc}; pass --models")
    structures = args.structures or sorted(
        d.name for d in args.qe_ph_root.iterdir()
        if d.is_dir() and any((args.mlip_root / m / d.name / "phonopy_params.yaml").exists() for m in models)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    parity: dict[str, list] = {m: [] for m in models}
    for name in structures:
        qe = load_qe_reference(args.qe_ph_root / name, name, args.matdyn_x, tuple(args.nk), args.deltaE)
        if qe is None:
            print(f"{name}: no QE .fc/.freq under {args.qe_ph_root / name}, skipped")
            continue
        n_atoms = len(qe["header"].symbols)
        mlips = {}
        for model in models:
            m = load_mlip_on_reference_path(args.mlip_root / model, name, qe)
            if m is None:
                continue
            mlips[model] = m
            row = {"structure": name, "model": model, "n_atoms": n_atoms}
            row.update(band_metrics(qe["freqs"], m["freqs"]))
            if qe["dos"] is not None and m["dos"] is not None:
                row.update(dos_metrics(*qe["dos"], *m["dos"], n_atoms, sigma_cm1=args.sigma, temperature_k=args.temperature))
            for key in ("relax", "supercell", "nac", "pressure_GPa", "force_eval_wall_time_s"):
                if key in m["meta"]:
                    row[key] = str(m["meta"][key]) if isinstance(m["meta"][key], list) else m["meta"][key]
            rows.append(row)
            parity[model].append((qe["freqs"], m["freqs"]))

        if not mlips:
            continue
        if qe["dos"] is None:
            print(f"{name}: no QE vDOS ({name}.dos) -- dispersion metrics only (pass --matdyn-x to generate it)")
        plot_structure(name, qe, mlips, colors, args.out_dir / f"{name}_benchmark.png", args.sigma)

    if not rows:
        sys.exit("No (structure, model) pairs with both QE and MLIP results found.")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    summary_cols = [c for c in SUMMARY_COLUMNS if c in metrics.columns]
    summary = metrics.groupby("model")[summary_cols].mean(numeric_only=False)
    summary.insert(0, "n_structures", metrics.groupby("model").size())
    summary.to_csv(args.out_dir / "summary.csv")

    plot_parity({m: v for m, v in parity.items() if v}, colors, args.out_dir / "parity.png")
    plot_band_mae(metrics, colors, args.out_dir / "band_mae.png")
    plot_overview(metrics, colors, args.temperature, args.out_dir / "metrics_overview.png")

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3g}".format):
        per_struct_cols = ["structure", "model", "band_mae_cm1", "band_mean_signed_err_cm1",
                           "omega_min_ref_cm1", "omega_min_mlip_cm1", "stability_agrees"]
        per_struct_cols += [c for c in ("dos_wasserstein_cm1", "dos_overlap", "s_kB_err") if c in metrics.columns]
        print(metrics[per_struct_cols].to_string(index=False))
        print("\nPer-model means:")
        print(summary.to_string())
    print(f"\nWrote metrics, summary and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
