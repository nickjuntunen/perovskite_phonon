"""How well do the chemical composition clusters (cluster_compositions.py
--mp-metadata, data/mp_composition_clusters.csv) organize the MLIP vDOS of the
selected stable polymorphs (cluster_polymorphs.py's polymorph_clusters.csv)?

Distances are Wasserstein-2 between unit-area vDOS (Euclidean on quantile
functions, see cluster_polymorphs.py), in cm-1, and on a log-frequency axis
(log-quantile), where a uniform frequency scaling -- heavier X, softer
lattice -- is a constant offset and only spectral shape is left.

Reported, per chemical cluster: size, anion(s), mean vDOS frequency, omega_max,
gapped fraction, and the within-cluster RMS W2 spread (cm-1, and relative to
the cluster's mean frequency). Overall: the fraction of vDOS variance each
labeling explains (eta^2 = 1 - within/total sum of squares, in both metrics),
for the chemical clusters, the X anion alone, and the vDOS KMeans clusters
(the k=10 optimum this metric allows); plus silhouette, and ARI/NMI between the
chemical and vDOS clusterings.

Outputs, next to --clusters-csv:
    vdos_by_chem_cluster.csv    per-chemical-cluster table
    vdos_by_chem_cluster.png    median vDOS + 10-90% band per chemical cluster
    vdos_by_chem_cluster_log.png  the same on log(omega), each normalized to its own mean frequency

Usage:
    uv run python3 scripts/analysis/vdos_by_chem_cluster.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.paths import MP_SWEEP_DIR, STABLE_POLYMORPHS_DIR
from src.analysis.polymorphs import eta_squared, rms_spread
from src.qe.phonon_dos import load_vdos
from src.qe.vdos_features import quantile_function, smooth_dos
from src.utils.plotting import plot_band_panels, use_project_style

use_project_style()

DEFAULT_CSV = STABLE_POLYMORPHS_DIR / "polymorph_clusters.csv"
LEVELS = (np.arange(100) + 0.5) / 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--root", type=Path, default=MP_SWEEP_DIR)
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian broadening for the plots, cm-1")
    args = parser.parse_args()

    df = pd.read_csv(args.clusters_csv)
    spectra = [load_vdos(args.root / d) for d in df["dir"]]
    spectra = [(f, d / np.trapezoid(d, f)) for f, d in spectra]
    # / sqrt(n_levels): Euclidean distance between these rows is then W2 itself (a mean over levels)
    Q = np.array([quantile_function(f, d, LEVELS) for f, d in spectra])
    L = np.log(np.maximum(Q, 1.0)) / np.sqrt(len(LEVELS))  # W2 on log(omega)
    Q = Q / np.sqrt(len(LEVELS))                            # W2, cm-1
    chem = df["chem_cluster"].astype(int).to_numpy()
    anion = df["mp_x_site"].fillna("?").to_numpy()

    print(f"{len(df)} stable polymorphs. Fraction of vDOS variance explained (eta^2):")
    print(f"  {'labeling':<26}{'groups':>7}{'W2 (cm-1)':>11}{'W2 log(w)':>11}")
    for name, lab in [("chemical clusters", chem), ("X anion", anion),
                      ("chem cluster x anion", np.char.add(chem.astype(str), anion.astype(str))),
                      ("vDOS KMeans (W2)", df["vdos_cluster"].to_numpy())]:
        print(f"  {name:<26}{len(np.unique(lab)):>7}{eta_squared(Q, lab):>11.3f}{eta_squared(L, lab):>11.3f}")
    big = np.isin(chem, [c for c in np.unique(chem) if (chem == c).sum() > 2])  # silhouette needs n > 1
    print(f"silhouette of chemical clusters in vDOS space: W2 {silhouette_score(Q[big], chem[big]):.3f}, "
          f"log {silhouette_score(L[big], chem[big]):.3f}")
    print(f"chemical vs vDOS clusters: ARI {adjusted_rand_score(chem, df['vdos_cluster']):.3f}, "
          f"NMI {normalized_mutual_info_score(chem, df['vdos_cluster']):.3f}")
    print("\nchemical cluster (rows) x vDOS cluster (cols):")
    print(pd.crosstab(df["chem_cluster"].astype(int), df["vdos_cluster"]).to_string())

    rows, titles = [], {}
    for c in np.unique(chem):
        m = chem == c
        sub = df[m]
        mean_w = sub["vdos_mean_cm1"]
        anions = sub["mp_x_site"].value_counts()
        rows.append({
            "chem_cluster": c, "n": int(m.sum()),
            "anions": ", ".join(f"{k} {v}" for k, v in anions.items()),
            "tolerance_factor": sub["tolerance_factor"].mean(), "octahedral_factor": sub["octahedral_factor"].mean(),
            "vdos_mean_cm1_median": mean_w.median(), "vdos_mean_cm1_q10": mean_w.quantile(0.1),
            "vdos_mean_cm1_q90": mean_w.quantile(0.9), "vdos_max_cm1_median": sub["vdos_max_cm1"].median(),
            "gapped_fraction": (sub["vdos_gap_width_cm1"] > 20).mean(),
            "w2_spread_cm1": rms_spread(Q[m]),
            "w2_spread_relative": rms_spread(Q[m]) / mean_w.mean(),
            "log_w2_spread": rms_spread(L[m]),
            "top_vdos_clusters": ", ".join(f"{k}:{v}" for k, v in sub["vdos_cluster"].value_counts().head(3).items()),
            "top_crystal_systems": ", ".join(f"{k} {v}" for k, v in sub["crystal_system"].value_counts().head(2).items()),
        })
        titles[c] = f"chem {c} ({'/'.join(anions.index[:2])}, n={m.sum()})"
    table = pd.DataFrame(rows)
    out_dir = args.clusters_csv.parent
    table.to_csv(out_dir / "vdos_by_chem_cluster.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + table.drop(columns=["top_crystal_systems"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    grid = np.arange(0, max(f[-1] for f, _ in spectra) + 2, 2.0)
    curves = np.array([np.interp(grid, f, smooth_dos(f, d, args.sigma), left=0, right=0) for f, d in spectra])
    plot_band_panels(grid, curves, chem, titles, r"$\omega$ (cm$^{-1}$)", out_dir / "vdos_by_chem_cluster.png")
    # Shape only: each spectrum on omega / <omega>, density per unit log(omega/<omega>)
    u = np.linspace(-3, 1.2, 300)
    scaled = []
    for (f, d), w in zip(spectra, df["vdos_mean_cm1"]):
        x = np.log(np.maximum(f, 1e-3) / w)
        scaled.append(np.interp(u, x, smooth_dos(f, d, args.sigma) * f, left=0, right=0))
    plot_band_panels(np.exp(u), np.array(scaled), chem, titles, r"$\omega / \langle\omega\rangle$",
                out_dir / "vdos_by_chem_cluster_log.png", logx=True)
    print(f"\n-> {out_dir}/vdos_by_chem_cluster.{{csv,png}}, vdos_by_chem_cluster_log.png")


if __name__ == "__main__":
    main()
