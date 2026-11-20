"""Figure for propose_candidates.py --retrospective (data/proposals/retrospective).

    fig_retrospective.png
      (a) hit rate of the model's top picks vs the pool's base rate, at each
          tolerance (a hit: every core target within tol x MLIP noise); the
          ratio is the enrichment the model buys.
      (b) per held-out chemical cluster, tol 3: base rate -> top-pick hit rate,
          with the cluster's size; clusters where the model adds nothing stand out.
      (c) per target draw: median Chebyshev miss of the top picks vs of the whole
          pool; below the diagonal the model's picks are closer than a random member.

Usage: uv run python3 scripts/analysis/plot_retrospective.py [--dir data/proposals/retrospective]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GREY, BLUE, INK = "#8a8984", "#2a78d6", "#333333"


def cluster_chemistry() -> dict:
    """cluster -> its X anions by share, e.g. 'Br 60% I 40%', from the selected polymorphs."""
    p = PROJECT_ROOT / "data" / "mlip_stable_polymorphs" / "stable_polymorphs.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, usecols=["cluster", "mp_x_site"]).dropna()
    out = {}
    for k, g in d.groupby(d["cluster"].astype(int)):
        sh = g["mp_x_site"].value_counts(normalize=True)
        out[k] = " ".join(f"{x} {v:.0%}" for x, v in sh.items() if v >= 0.1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "data" / "proposals" / "retrospective")
    parser.add_argument("--n-top", type=int, default=8, help="Picks per target the run used (label only)")
    args = parser.parse_args()
    res = pd.read_csv(args.dir / "retrospective.csv")
    summ = pd.read_csv(args.dir / "retrospective_summary.csv")

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GREY, "xtick.color": "#555", "ytick.color": "#555"})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})

    x = np.arange(len(summ))
    w = 0.36
    a.bar(x - w / 2 - 0.01, summ["base_rate"], w, color=GREY, label="random pool member")
    a.bar(x + w / 2 + 0.01, summ["hit_rate_top"], w, color=BLUE, label=f"model's top {args.n_top}")
    for xi, r in zip(x, summ.itertuples()):
        a.annotate(f"{r.enrichment:.1f}×", (xi + w / 2, r.hit_rate_top), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=8, color=INK)
    a.set_xticks(x, [f"±{t:g}" for t in summ["tol"]])
    a.set_xlabel("tolerance (MLIP-noise units, every target)")
    a.set_ylabel("fraction that hit the target")
    a.set_title(f"(a) hit rate, {int(summ['n'].iloc[0])} target draws", fontsize=9, loc="left")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")

    chem = cluster_chemistry()
    t3 = res[res["tol"] == 3.0]
    per = t3.groupby("cluster").agg(base=("base_rate", "mean"), top=("hit_rate_top", "mean"),
                                    n=("n_cluster", "first"), k=("target", "size")).astype({"n": int}).sort_values("top")
    y = np.arange(len(per))
    b.hlines(y, per["base"], per["top"], color=GREY, lw=1.5)
    b.scatter(per["base"], y, s=30, color=GREY, zorder=3, label="random pool member", edgecolor="white")
    b.scatter(per["top"], y, s=30, color=BLUE, zorder=3, label=f"model's top {args.n_top}", edgecolor="white")
    b.set_yticks(y, [f"{k}: {chem.get(k, '?')}  (n={n})" for k, n in zip(per.index, per["n"])])
    b.set_xlabel("hit rate at ±3")
    b.set_title(f"(b) per held-out chemical cluster ({int(per['k'].iloc[0])} targets each)", fontsize=9, loc="left")
    b.grid(axis="x", color="#e6e5e0", lw=0.6)

    m = t3.drop_duplicates(["cluster", "target"])
    lim = [0, float(np.ceil(max(m["median_miss_pool"].max(), m["median_miss_top"].max())))]
    c.plot(lim, lim, color=GREY, lw=1)
    c.scatter(m["median_miss_pool"], m["median_miss_top"], s=18, color=BLUE, alpha=0.8, edgecolor="white", lw=0.5)
    c.axhspan(0, 3, color="#eeede9", lw=0, zorder=0)
    c.text(lim[1] * 0.98, 1.5, "within ±3", ha="right", va="center", fontsize=7, color="#555")
    frac = (m["median_miss_top"] < m["median_miss_pool"]).mean()
    c.text(lim[1] * 0.02, lim[1] * 0.95, f"top picks closer in {frac:.0%} of draws", fontsize=7.5, color=INK, va="top")
    c.set_xlim(lim)
    c.set_ylim(lim)
    c.set_xlabel("median miss, whole held-out cluster")
    c.set_ylabel(f"median miss, top {args.n_top}")
    c.set_title("(c) per target draw (Chebyshev, noise units)", fontsize=9, loc="left")
    fig.tight_layout(w_pad=2.5)
    out = args.dir / "fig_retrospective.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
