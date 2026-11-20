"""Figures for gp_disorder.py's result (gp_vs_tree/disorder/): why the X-site
disorder features and the bowing-only target don't fix R2(series).

    fig_gp_disorder.png
      (a, b) R2(series) per target: linear end-member line, GP on the base
             features, + disorder features, bowing-only target; held-out halide
             pair (a) and held-out series (b). Values below -3 pinned at the edge.
      (c)    the disorder feature mass_cv_X per halide pair: MP training rows
             are all 0, and each pair occupies its own interval, so holding a
             pair out asks the GP to extrapolate in exactly those features.
      (d)    one held-out Cl/I series of rel_width_X: production values, the
             end-member line, and the three GP predictions.

Usage: uv run python3 gp_vs_tree/plot_gp_disorder.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = ROOT / "gp_vs_tree" / "disorder"
GREY, BLUE, ORANGE, AQUA, INK = "#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#333333"
MODELS = [("linear", "end-member line", GREY), ("base GP", "GP, base features", BLUE),
          ("dis GP", "GP + disorder features", ORANGE), ("dis_delta GP", "GP, bowing-only target", AQUA)]
LO = -3.0


def dot_panel(ax, res: pd.DataFrame, scheme: str, title: str) -> None:
    order = res.sort_values(f"{scheme} base GP", ascending=False).index
    y = np.arange(len(order))
    ax.axvline(0, color=GREY, lw=0.8)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    for k, (col, label, c) in enumerate(MODELS):
        key = f"{scheme} linear" if col == "linear" else f"{scheme} {col}"
        v = res.loc[order, key]
        pinned = v < LO
        ax.scatter(v[~pinned], (y + (k - 1.5) * 0.17)[~pinned.to_numpy()], s=24, color=c, label=label, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(np.full(pinned.sum(), LO), (y + (k - 1.5) * 0.17)[pinned.to_numpy()], s=26, color=c,
                   marker="<", zorder=3, edgecolor="white", linewidth=0.6)
        for yi, val in enumerate(v):
            if val < LO:
                ax.annotate(f"{val:.0f}", (LO, yi + (k - 1.5) * 0.17), xytext=(5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6, color=INK)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(LO - 0.1, 1.02)
    ax.set_xlabel(f"R²(series)      (◀ = below {LO:.0f}, value printed beside it)")
    ax.set_title(title, fontsize=9, loc="left")


def main() -> None:
    g: dict = {}
    exec(open(ROOT / "gp_vs_tree" / "gp_setup.py").read(), g)
    dm, Tm, end_value = g["dm"], g["Tm"], g["end_value"]
    from src.analysis.composition_features import x_site_disorder

    res = pd.read_csv(D / "gp_disorder.csv", index_col=0)
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "xtick.color": "#555", "ytick.color": "#555"})
    fig = plt.figure(figsize=(11, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.38, wspace=0.42)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    dot_panel(a, res, "pair", "(a) held-out halide pair: transfer to a pair never seen")
    dot_panel(b, res, "series", "(b) held-out series, every pair seen")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.955))


    # (c) disorder-feature support per pair
    c = fig.add_subplot(gs[1, 0])
    dis = x_site_disorder(dm)
    pairs = ["Cl/Br", "Br/I", "Cl/I"]
    c.scatter(np.zeros(1), [3], s=40, color=INK, zorder=3)
    c.annotate(f"MP polymorphs (all {len(g['df'])}): 0", (0, 3), xytext=(6, 0), textcoords="offset points",
               va="center", fontsize=7.5, color=INK)
    for k, p in enumerate(pairs):
        v = dis.loc[dm["pair"] == p, "mass_cv_X"]
        c.scatter(v, np.full(len(v), 2 - k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)),
                  s=6, color=GREY, alpha=0.6, lw=0)
        c.hlines(2 - k, v.min(), v.max(), color=INK, lw=2)
    c.set_yticks([3, 2, 1, 0], ["MP", *pairs])
    c.set_xlabel("mass_cv_X (X-site mass coefficient of variation)")
    c.set_title("(c) each pair sits in its own range of the disorder features", fontsize=9, loc="left")
    c.grid(axis="x", color="#e6e5e0", lw=0.6)

    # (d) a held-out Cl/I series, rel_width_X
    d = fig.add_subplot(gs[1, 1])
    t = "rel_width_X"
    y = Tm[t]
    lin = ((1 - dm["x2"]) * end_value[t].reindex(dm["end_X1"]).to_numpy()
           + dm["x2"] * end_value[t].reindex(dm["end_X2"]).to_numpy())
    ok = y.notna() & lin.notna() & (dm["pair"] == "Cl/I")
    s = dm.loc[ok].groupby("series").size().sort_values(ascending=False).index[0]
    rows = dm.index[ok & (dm["series"] == s)]
    rows = rows[np.argsort(dm.loc[rows, "x2"].to_numpy())]
    x = dm.loc[rows, "x2"]
    e1, e2 = end_value[t].get(dm.loc[rows[0], "end_X1"]), end_value[t].get(dm.loc[rows[0], "end_X2"])
    d.plot([0, 1], [e1, e2], color=GREY, lw=1.5, label="end-member line")
    d.scatter([0, 1], [e1, e2], s=40, color=GREY, zorder=3, label="end members (MP)")
    d.scatter(x, y[rows], s=36, color=INK, zorder=4, label="mixed, single SQS")
    for col, label, cc in MODELS[1:]:
        v, sch = col.split()[0], "pair"
        pr = pd.read_csv(D / "parts" / t / f"{v}__{sch}__Cl-I.csv", index_col=0)["GP"]
        d.plot(x, pr.reindex(rows), "o--", color=cc, ms=4, lw=1.3, label=label)
    d.set_xlabel("x (I fraction)")
    d.set_ylabel(t)
    d.set_title(f"(d) Cl/I held out: {s}", fontsize=9, loc="left")
    d.legend(fontsize=6.5, frameon=False, loc="best")
    fig.savefig(ROOT / "gp_vs_tree" / "fig_gp_disorder.png", dpi=170, bbox_inches="tight")
    print(f"-> {ROOT / 'gp_vs_tree' / 'fig_gp_disorder.png'}")


if __name__ == "__main__":
    main()
