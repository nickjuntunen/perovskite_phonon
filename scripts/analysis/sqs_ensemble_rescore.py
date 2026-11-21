"""SQS-ensemble targets, and the series models re-scored against them.

Every production mixed-halide value comes from one SQS cell, so part of what
the series models "miss" along a series is that cell's disorder noise, not
physics. The ensemble set (data/mixed_halides_ensemble, 5 SQS seeds per
Cl/Br and Br/I production composition, postprocess_sqs_ensemble.sh steps 1-4)
gives each composition's configurational average. This script:

1. Ensemble targets. The notebook's targets (phonon_targets, shape PCs in the
   MP basis) of every dynamically stable realization with all sites assigned,
   then the count-weighted mean per composition (symmetry-equivalent cells
   count as repeated draws, as in sqs_convergence.py), its seed std and
   standard error sqrt(var / N). -> ensemble_targets.csv, one row per
   composition, with the production (single-SQS) value alongside.

2. Re-scoring. The predictions the series models already made (no refit):
       linear                 the end-member Vegard line
       MP->mixed GP / tree    trained on MP polymorphs only   (gp_vs_tree/pred_<t>.csv)
       held-out pair GP/tree  trained on MP + the other pairs  (same files)
       <variant> <scheme> GP/tree   gp_disorder.py's fold predictions
                              (gp_vs_tree/disorder/parts; variant base / dis / dis_delta,
                              scheme pair = held-out pair, series = held-out series)
       ... [<tier>, <labels>, <geometry>]   the same refit with ensemble labels and/or
                              ensemble-averaged geometry inputs, or composition-only inputs
                              (gp_disorder.py --labels / --geometry / --tier ->
                              gp_vs_tree/disorder_<tier>_<labels>L_<geometry>G), where present
   each scored by R2(series) on the same rows against the single-SQS target and
   against the ensemble mean. Rows: production compositions with an ensemble
   mean from >= --min-stable stable realizations, in series that keep >= 2 such
   rows; the series mean is over those rows. dR2 = R2(ensemble) - R2(single),
   with a 95% interval from resampling series (--n-boot). Two reference
   lines per target:
       single->ens   the single-SQS value itself as the "prediction" of the
                     ensemble mean: how far a perfect single-SQS model would
                     fall short of the averaged truth
       ceiling       1 - mean(sem^2) / within-series variance of the ensemble
                     means: the residual disorder noise left after averaging
   -> rescore.csv (all), rescore_summary.csv (the headline models, all pairs)

3. Figures (in <set>/analysis/figures): rescore_r2.png (R2 single -> ensemble
   per target for the headline models) and rescore_series.png (example series:
   single-SQS values, ensemble mean +- sem, held-out-pair GP prediction).

Usage (after steps 1-4 of postprocess_sqs_ensemble.sh):
    uv run python3 scripts/analysis/sqs_ensemble_rescore.py
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

SET_DIR = PROJECT_ROOT / "data" / "mixed_halides_ensemble"
GP_DIR = PROJECT_ROOT / "gp_vs_tree"
HEADLINE = ["linear", "held-out pair GP", "held-out pair tree", "base series GP", "base series tree"]
GREY, BLUE, ORANGE, AQUA = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a"


def notebook_tables() -> dict:
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        g: dict = {}
        exec(open(GP_DIR / "gp_setup.py").read(), g)
    finally:
        os.chdir(cwd)
    return g


def realization_targets(set_dir: Path, g: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(dn, Tn, targets): the noise-check / ensemble realizations that are
    dynamically stable with every site assigned, with a 'count' column (manifest
    weight x number of seeds: symmetry-equivalent cells pooled), and their
    notebook targets, shape PCs in the MP basis."""
    from src.analysis.sqs_ensemble import realizations
    dn = realizations(set_dir)
    dn = dn[dn["dynamically_stable"].astype(bool) & (dn["sublattice_unassigned"] == 0)].reset_index(drop=True)
    n_sqs = int(pd.read_csv(set_dir / "manifest.csv")["seed"].max()) + 1
    dn["count"] = (dn["weight"] * n_sqs).round()
    Tn = g["phonon_targets"](dn)
    nq = np.load(set_dir / "analysis" / "design_descriptors_quantiles.npz")
    qmap = dict(zip(nq["keys"], nq["quantiles_cm1"]))
    Q = np.array([qmap[k] for k in dn["key"]])
    S = g["shape_pca"].transform(Q / Q.mean(axis=1, keepdims=True))
    for k in range(g["n_shape"]):
        Tn[f"shape_pc{k + 1}"] = S[:, k]
    return dn, Tn, [t for t in g["HALIDE_TARGETS"] if t in Tn]


def ensemble_targets(dn: pd.DataFrame, Tn: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for f, idx in dn.groupby("manifest_formula").groups.items():
        c = dn.loc[idx, "count"].to_numpy(float)
        row = {"formula": f, "n_cells": len(idx), "n_stable": int(c.sum())}
        for t in targets:
            v = Tn.loc[idx, t].to_numpy(float)
            ok = np.isfinite(v)
            n = c[ok].sum()
            if n == 0:
                continue
            m = (c[ok] * v[ok]).sum() / n
            var = (c[ok] * (v[ok] - m) ** 2).sum() / (n - 1) if n > 1 else np.nan
            row |= {f"{t}_mean": m, f"{t}_std": np.sqrt(var), f"{t}_sem": np.sqrt(var / n)}
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_sources(g: dict, targets: list[str]) -> dict[tuple[str, str], pd.Series]:
    """{(source, target): prediction indexed by run_name}."""
    dm, end_value = g["dm"], g["end_value"]
    out = {}
    for t in targets:
        lin = ((1 - dm["x2"]) * end_value[t].reindex(dm["end_X1"]).to_numpy()
               + dm["x2"] * end_value[t].reindex(dm["end_X2"]).to_numpy())
        out[("linear", t)] = pd.Series(lin.to_numpy(), index=dm["run_name"]).dropna()
        p = GP_DIR / f"pred_{t}.csv"
        if p.exists():
            pr = pd.read_csv(p).set_index("run_name")
            for col, name in [("mp GP", "MP->mixed GP"), ("mp tree", "MP->mixed tree"),
                              ("pair GP", "held-out pair GP"), ("pair tree", "held-out pair tree")]:
                out[(name, t)] = pr[col].dropna()
        for root in sorted(GP_DIR.glob("disorder*")):
            # gp_disorder.py's output dirs: disorder (C+G, single labels and geometry) or
            # disorder_<tier>_<labels>L_<geometry>G -> tag " [<tier>, <labels> labels, <geometry> geometry]"
            tag = "" if root.name == "disorder" else " [" + ", ".join(
                x.replace("L", " labels").replace("G", " geometry") if x[-1] in "LG" and x[:-1] in
                ("single", "ensemble") else x for x in root.name.split("_")[1:]) + "]"
            parts = root / "parts" / t
            if not parts.is_dir():
                continue
            for v in ("base", "dis", "dis_delta"):
                for scheme in ("pair", "series"):
                    files = sorted(parts.glob(f"{v}__{scheme}__*.csv"))
                    if not files:
                        continue
                    pr = pd.concat([pd.read_csv(f, index_col=0) for f in files]).set_index("run_name")
                    for m in ("GP", "tree"):
                        out[(f"{v} {scheme} {m}{tag}", t)] = pr[m]
    return out


def r2_series(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float:
    d = pd.DataFrame({"y": y, "s": s})
    sm = d.groupby("s")["y"].transform("mean").to_numpy()
    den = ((y - sm) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / den) if den > 0 else np.nan


def boot_dr2(y1, y2, p, s, n_boot: int, rng) -> tuple[float, float]:
    """95% interval of dR2, resampling whole series with replacement (each draw
    relabelled so a series drawn twice counts as two series)."""
    groups = list(pd.Series(np.arange(len(s))).groupby(s).indices.values())
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        lab = np.concatenate([np.full(len(groups[i]), j) for j, i in enumerate(pick)])
        out[b] = r2_series(y2[idx], p[idx], lab) - r2_series(y1[idx], p[idx], lab)
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, default=SET_DIR)
    parser.add_argument("--min-stable", type=int, default=3,
                        help="Stable realizations (counting pooled duplicates) a composition needs (default 3)")
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = args.set_dir / "analysis"
    rng = np.random.default_rng(0)

    g = notebook_tables()
    dm, Tm = g["dm"], g["Tm"]
    dn, Tn, targets = realization_targets(args.set_dir, g)
    ens = ensemble_targets(dn, Tn, targets)
    prod = pd.concat([dm[["formula", "run_name", "series", "pair", "x2"]], Tm[targets]], axis=1)
    et = prod[["formula", "run_name", "series", "pair", "x2"]].merge(ens, on="formula", how="inner")
    for t in targets:
        et[f"{t}_single"] = et["formula"].map(prod.set_index("formula")[t])
    et.to_csv(out / "ensemble_targets.csv", index=False)
    print(f"{len(dn)} stable realizations, {len(ens)} compositions; {len(et)}/{len(dm)} production "
          f"compositions matched ({et['pair'].value_counts().to_dict()}); stable cells per composition: "
          f"{et['n_stable'].value_counts().sort_index().to_dict()}")

    base = et[et["n_stable"] >= args.min_stable].set_index("run_name")
    sources = prediction_sources(g, targets)
    rows = []
    for (src, t), pred in sources.items():
        for subset in ("all", "Cl/Br", "Br/I"):
            b = base if subset == "all" else base[base["pair"] == subset]
            d = pd.DataFrame({"y1": b[f"{t}_single"], "y2": b[f"{t}_mean"], "sem": b[f"{t}_sem"],
                              "s": b["series"], "p": pred.reindex(b.index)}).dropna(subset=["y1", "y2", "p"])
            d = d[d.groupby("s")["s"].transform("size") >= 2]
            if d["s"].nunique() < 3:
                continue
            y1, y2, p, s = (d[c].to_numpy() for c in ("y1", "y2", "p", "s"))
            r1, r2 = r2_series(y1, p, s), r2_series(y2, p, s)
            lo, hi = boot_dr2(y1, y2, p, s, args.n_boot, rng) if args.n_boot else (np.nan, np.nan)
            v1 = ((y1 - d.groupby("s")["y1"].transform("mean").to_numpy()) ** 2).mean()
            v2 = ((y2 - d.groupby("s")["y2"].transform("mean").to_numpy()) ** 2).mean()
            dev = y1 - y2  # the production cell's disorder deviation
            rows.append({"source": src, "target": t, "subset": subset, "n_rows": len(d), "n_series": d["s"].nunique(),
                         "R2_single": r1, "R2_ensemble": r2, "dR2": r2 - r1, "dR2_lo": lo, "dR2_hi": hi,
                         "single->ens": r2_series(y2, y1, s), "ceiling": 1 - np.nanmean(d["sem"] ** 2) / v2,
                         "mse_single": ((y1 - p) ** 2).mean(), "mse_ensemble": ((y2 - p) ** 2).mean(),
                         "var_series_single": v1, "var_series_ensemble": v2,
                         # slope of (prediction - ensemble mean) on the cell's deviation: 0 = the prediction
                         # ignores which SQS cell was computed, 1 = it follows that cell's noise fully
                         "tracks_cell": float(np.dot(p - y2, dev) / np.dot(dev, dev)) if np.dot(dev, dev) > 0 else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(out / "rescore.csv", index=False)
    summ = res[(res["subset"] == "all") & res["source"].isin(HEADLINE)].pivot(
        index="target", columns="source", values=["R2_single", "R2_ensemble"])
    ref = res[(res["subset"] == "all") & (res["source"] == "linear")].set_index("target")[["single->ens", "ceiling",
                                                                                           "n_rows", "n_series"]]
    summ.columns = [f"{s} | {k.replace('R2_', '')}" for k, s in summ.columns]
    summ = summ.join(ref).reindex(targets)
    summ.to_csv(out / "rescore_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = [c for s in HEADLINE for c in (f"{s} | single", f"{s} | ensemble") if c in summ]
        print("\nR2(series), single-SQS target -> ensemble-averaged target (same rows, all pairs):")
        print(summ[cols + ["single->ens", "ceiling", "n_rows", "n_series"]].round(2).to_string())
    print(f"-> {out}/ensemble_targets.csv, rescore.csv, rescore_summary.csv")
    if not args.no_figures:
        figures(res, base, sources, targets, out / "figures")


def figures(res: pd.DataFrame, base: pd.DataFrame, sources: dict, targets: list[str], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "axes.labelcolor": "#333", "xtick.color": "#555",
                         "ytick.color": "#555"})
    LO = -1.5
    show = [s for s in ["linear", "held-out pair GP", "base series GP",
                        "base series GP [CG, ensemble labels, ensemble geometry]",
                        "base series tree [C, ensemble labels, single geometry]"]
            if s in set(res["source"]) and res.loc[res["source"] == s, "R2_single"].notna().any()]
    a = res[res["subset"] == "all"]
    order = [t for t in targets if t in set(a["target"])]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show) + 1.2, 0.27 * len(order) + 1.4),
                             sharey=True, squeeze=False)
    for ax, src in zip(axes[0], show):
        d = a[a["source"] == src].set_index("target").reindex(order)
        y = np.arange(len(order))
        c1, c2 = d["R2_single"].clip(lower=LO), d["R2_ensemble"].clip(lower=LO)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.hlines(y, c1, c2, color=GREY, lw=1.5)
        ax.scatter(c1, y, s=26, color=BLUE, zorder=3, label="vs single SQS", edgecolor="white", linewidth=0.8)
        ax.scatter(c2, y, s=26, color=ORANGE, zorder=3, label="vs 5-SQS mean", edgecolor="white", linewidth=0.8)
        if "ceiling" in d:
            ax.scatter(d["ceiling"].clip(lower=LO), y, marker="|", s=60, color="#333", zorder=2,
                       label="noise ceiling after averaging")
        for yi, (v1, v2) in enumerate(zip(d["R2_single"], d["R2_ensemble"])):
            if np.isfinite(v1) and min(v1, v2) < LO:
                ax.annotate(f"{v1:.0f}→{v2:.0f}", (LO, yi), xytext=(2, 3), textcoords="offset points",
                            fontsize=7, color="#555")
        ax.set_xlim(LO - 0.05, 1.02)
        ax.set_title(src, fontsize=9)
        ax.set_xlabel("R²(series)")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    axes[0][0].set_yticks(np.arange(len(order)), order)
    axes[0][0].invert_yaxis()
    axes[0][-1].legend(loc="lower right", fontsize=7, frameon=False)
    n = a[a["source"] == "linear"]["n_rows"].max()
    fig.suptitle(f"Same predictions, scored against one SQS cell vs the 5-cell mean ({n} compositions, "
                 f"Cl/Br + Br/I); values < {LO} pinned at the left edge", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "rescore_r2.png", dpi=170)
    plt.close(fig)

    # example series: the target where averaging moved R2 most for the held-out-pair GP
    src = "held-out pair GP"
    d = a[a["source"] == src]
    if not len(d):
        return
    ex = [t for t in ["rel_width_X", "shape_pc1", "rel_centroid_B", "log_msd_A"] if t in set(d["target"])][:3]
    # the two longest series of each pair
    per = base.groupby("series").agg(n=("x2", "size"), pair=("pair", "first"))
    per = per[per["n"] >= 3].sort_values("n", ascending=False)
    pick = [s for pair in ("Br/I", "Cl/Br") for s in per.index[per["pair"] == pair][:2]]
    if not pick or not ex:
        print("no series with >= 3 averaged compositions: rescore_series.png skipped")
        return
    fig, axes = plt.subplots(len(ex), len(pick), figsize=(2.6 * len(pick), 2.1 * len(ex)), squeeze=False)
    for i, t in enumerate(ex):
        pred = sources[(src, t)]
        for j, s in enumerate(pick):
            ax = axes[i][j]
            b = base[base["series"] == s].sort_values("x2")
            ax.errorbar(b["x2"], b[f"{t}_mean"], yerr=b[f"{t}_sem"], fmt="o-", color=ORANGE, ms=4, lw=1.5,
                        capsize=2, label="5-SQS mean ± s.e.")
            ax.plot(b["x2"], b[f"{t}_single"], "o", color=BLUE, ms=4, label="single SQS (production)")
            ax.plot(b["x2"], pred.reindex(b.index), "s--", color=AQUA, ms=3.5, lw=1.2, label=src)
            if i == 0:
                ax.set_title(s, fontsize=8)
            if j == 0:
                ax.set_ylabel(t)
            if i == len(ex) - 1:
                ax.set_xlabel("x (second halide)")
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "rescore_series.png", dpi=170)
    plt.close(fig)
    print(f"-> {fig_dir}/rescore_r2.png, rescore_series.png")


if __name__ == "__main__":
    main()
