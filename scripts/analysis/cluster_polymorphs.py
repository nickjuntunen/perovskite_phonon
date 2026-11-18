"""Cluster the selected stable polymorphs (scripts/analysis/select_stable_polymorphs.py's
stable_polymorphs.csv) by their MLIP vDOS, and cross-reference the clusters
against chemistry (cluster_compositions.py's clusters, X anion) and structure
(crystal system, space group).

Each vDOS (per-cell, from the run_mlip_phonons.py state the polymorph came
from) is normalized to unit area and represented by one of:

    quantile  its quantile function -- the frequency below which a fraction
              p of the modes lie -- at --n-features evenly spaced p. Euclidean
              distance between two of these is sqrt(--n-features) x the
              Wasserstein-2 distance between the spectra, so a rigid shift of a band costs its
              frequency difference (not, as with a binned DOS, the same as any
              non-overlapping shift) and gaps show up as jumps. Dominated by
              the overall frequency scale, i.e. by X mass and bond stiffness.
    dos       the Gaussian-broadened DOS itself on a common frequency grid --
              spectral *shape* at fixed frequency; sensitive to peak
              positions to within --sigma, blind to how far apart they are.
    log-quantile  the quantile function of log(frequency): Wasserstein-2 on a
              log axis, i.e. a uniform frequency scaling (heavier X, softer
              lattice) is a constant offset and the shape matters more.

The vectors are standardized per feature only for 'dos' (for the quantile
representations the raw cm-1 distances are the point), then clustered with
KMeans.

Outputs, in --out-dir (default: next to the input CSV):
    polymorph_clusters.csv        stable_polymorphs.csv + vdos_cluster + vDOS summary stats
    polymorph_clusters_vdos.png   per-cluster median vDOS with 10-90% band
    polymorph_clusters_pca.png    PCA of the feature vectors, one panel per cluster
    polymorph_clusters_k_scan.csv silhouette / inertia for k = 2..--k-max (--scan)

Usage:
    uv run python3 scripts/analysis/cluster_polymorphs.py
    uv run python3 scripts/analysis/cluster_polymorphs.py --representation log-quantile --n-clusters 8 --scan
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.paths import MP_SWEEP_DIR, STABLE_POLYMORPHS_DIR
from src.analysis.polymorphs import non_framework_reason
from src.qe.phonon_dos import load_vdos
from src.qe.vdos_features import quantile_function, smooth_dos, vdos_summary
from src.utils.plotting import CONTEXT, HIGHLIGHT, grid_shape, plot_band_panels, use_project_style

use_project_style()

DEFAULT_CSV = STABLE_POLYMORPHS_DIR / "stable_polymorphs.csv"


def featurize(dos_by_row: list, representation: str, n_features: int, grid: np.ndarray, sigma: float) -> np.ndarray:
    levels = (np.arange(n_features) + 0.5) / n_features
    if representation == "quantile":
        return np.array([quantile_function(f, d, levels) for f, d in dos_by_row])
    if representation == "log-quantile":
        # log(1 + omega) so the acoustic tail at omega -> 0 doesn't dominate
        return np.log1p(np.array([quantile_function(f, d, levels) for f, d in dos_by_row]))
    smoothed = [np.interp(grid, f, smooth_dos(f, d, sigma), left=0, right=0) for f, d in dos_by_row]
    return StandardScaler().fit_transform(np.array(smoothed))


def top_counts(series: pd.Series, n: int = 3) -> str:
    counts = series.value_counts()
    return ", ".join(f"{k} {v}" for k, v in counts.head(n).items())


def plot_cluster_pca(X: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100
    k = labels.max() + 1
    nrows, ncols = grid_shape(k, 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c >= k:
            ax.set_visible(False)
            continue
        mask = labels == c
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=4, color=CONTEXT, edgecolors="none")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, color=HIGHLIGHT, edgecolors="none")
        ax.set_title(f"cluster {c} (n={mask.sum()})", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}\\% var.)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}\\% var.)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polymorphs-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--root", type=Path, default=MP_SWEEP_DIR,
                        help="The run_mlip_phonons.py output the CSV's dir column is relative to")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: the CSV's directory")
    parser.add_argument("--representation", choices=["quantile", "log-quantile", "dos"], default="quantile")
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--n-features", type=int, default=100,
                        help="Quantile levels for the quantile representations (default: 100)")
    parser.add_argument("--grid-max", type=float, default=None,
                        help="Upper end of the common frequency grid, cm-1 (default: the largest vDOS max)")
    parser.add_argument("--grid-step", type=float, default=2.0, help="Common grid spacing, cm-1 (default: 2)")
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian broadening for 'dos' and plots, cm-1")
    parser.add_argument("--min-gap", type=float, default=20.0,
                        help="Gap width counted as 'gapped' in the printed summary, cm-1 (default: 20)")
    parser.add_argument("--include-unstable", action="store_true",
                        help="Also cluster formulas whose pick is not dynamically stable (default: stable only)")
    parser.add_argument("--include-non-framework", action="store_true",
                        help="Also cluster H-containing and molecular-oxyanion (CO3, NO3, BO3, ...) compounds, "
                             "whose top band is internal modes (default: left out)")
    parser.add_argument("--scan", action="store_true", help="Also report silhouette/inertia for k = 2..--k-max")
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or args.polymorphs_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.polymorphs_csv).rename(columns={"cluster": "chem_cluster"})
    if not args.include_unstable:
        n_before = len(df)
        df = df[df["dynamically_stable"]]
        print(f"{n_before - len(df)} not dynamically stable pick(s) left out (--include-unstable to keep)")
    if not args.include_non_framework:
        reason = pd.Series({i: non_framework_reason(args.root / p) for i, p in df["structure_path"].items()})
        out = reason.dropna()
        print(f"{len(out)} non-framework pick(s) left out (--include-non-framework to keep): "
              f"{out.value_counts().to_dict()}, e.g. {', '.join(df.loc[out.index[:8], 'formula'])}")
        df = df.drop(index=out.index)

    dos_by_row, summaries, missing = [], [], []
    for i, row in df.iterrows():
        try:
            freq, dos = load_vdos(args.root / row["dir"])
        except FileNotFoundError as exc:
            missing.append(f"{row['material_id']}/{row['state']}: {exc}")
            continue
        summaries.append({"index": i, **vdos_summary(freq, dos)})
        dos_by_row.append((freq, dos / np.trapezoid(dos, freq)))
    if missing:
        print(f"{len(missing)} polymorph(s) without a vDOS, skipped:\n  " + "\n  ".join(missing))
    df = df.loc[[s["index"] for s in summaries]]
    df = df.join(pd.DataFrame(summaries).set_index("index"))

    grid_max = args.grid_max or max(f[d > 0].max() for f, d in dos_by_row)
    grid = np.arange(0, grid_max + args.grid_step, args.grid_step)
    X = featurize(dos_by_row, args.representation, args.n_features, grid, args.sigma)

    if args.scan:
        scan = []
        for k in range(2, args.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=args.random_state).fit(X)
            scan.append({"k": k, "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_})
        scan = pd.DataFrame(scan)
        scan.to_csv(out_dir / "polymorph_clusters_k_scan.csv", index=False)
        print("k scan (higher silhouette = better separated):")
        print(scan.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    labels = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.random_state).fit_predict(X)
    # Relabel by ascending median vDOS mean, so cluster 0 is the softest lattice
    order = np.argsort([np.median(df["vdos_mean_cm1"].to_numpy()[labels == c]) for c in range(args.n_clusters)])
    labels = np.argsort(order)[labels]
    df.insert(1, "vdos_cluster", labels)
    print(f"\n{len(df)} polymorphs, {args.representation} representation, k={args.n_clusters}, "
          f"silhouette {silhouette_score(X, labels):.3f}")

    x_anion = df["mp_x_site"].fillna("?")
    for c in range(args.n_clusters):
        sub = df[labels == c]
        print(
            f"\ncluster {c} (n={len(sub)}): <omega> {sub['vdos_mean_cm1'].median():.0f} cm-1 "
            f"[{sub['vdos_mean_cm1'].min():.0f}-{sub['vdos_mean_cm1'].max():.0f}], "
            f"omega_max {sub['vdos_max_cm1'].median():.0f}, gap > {args.min_gap:g} cm-1 in {(sub['vdos_gap_width_cm1'] > args.min_gap).mean():.0%}"
            f"\n  X: {top_counts(x_anion[labels == c])} | system: {top_counts(sub['crystal_system'])}"
            f" | chem cluster: {top_counts(sub['chem_cluster'].astype('Int64').astype(str))}"
            f"\n  e.g. {', '.join(sub.sample(min(6, len(sub)), random_state=0)['formula'])}"
        )

    df.to_csv(out_dir / "polymorph_clusters.csv", index=False)
    curves = np.array([np.interp(grid, f, smooth_dos(f, d, args.sigma), left=0, right=0) for f, d in dos_by_row])
    titles = {c: f"cluster {c} (n={(labels == c).sum()})" for c in range(args.n_clusters)}
    plot_band_panels(grid, curves, labels, titles, r"$\omega$ (cm$^{-1}$)", out_dir / "polymorph_clusters_vdos.png",
                     sharey=True, title_fontsize=9)
    plot_cluster_pca(X, labels, out_dir / "polymorph_clusters_pca.png",
                     f"KMeans on vDOS ({args.representation}), k={args.n_clusters}")
    print(f"\n-> {out_dir / 'polymorph_clusters.csv'}, polymorph_clusters_vdos.png, polymorph_clusters_pca.png")


if __name__ == "__main__":
    main()
