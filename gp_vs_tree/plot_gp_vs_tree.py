"""Figures for the GP vs gradient-boosted-tree comparison (gp_try.py's gp_all.csv
and pred_<target>.csv).

    fig_r2_by_test.png     every target's R2 in each test, tree vs GP
    fig_r2_scatter.png     GP R2 against tree R2, one dot per target, per test
    fig_series.png         example Cl-Br-I series: MLIP values, held-out-pair
                           predictions of each model, linear end-member baseline

Usage (from the repo root):
    MPLBACKEND=Agg uv run python gp_vs_tree/plot_gp_vs_tree.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
exec(open(HERE / "gp_setup.py").read())  # df, T, dm, Tm, end_value, has_ends, picked, plt, np, pd

try:
    plt.style.use("my_style")
except OSError:
    pass

TREE, GP, BASE = "#eb6834", "#2a78d6", "#8a8984"
INK, MUTED = "#25282a", "#6f6e69"
R2_FLOOR = -2.0  # values below are drawn at the floor with an arrow
TESTS = [("MP R2w", "MP, within family\n(grouped CV)"),
         ("R2series MP->mixed", "(1) MP only\n$\\to$ mixed, series"),
         ("R2series held-out pair", "(2) held-out pair,\nseries"),
         ("held-out chloride +mixed", "(3) held-out chloride\n+ mixed"),
         ("held-out bromide +mixed", "(3) held-out Br/I\n+ mixed")]
label = lambda t: t.replace("_", " ")

res = pd.read_csv(HERE / "gp_all.csv", index_col=0)
halide = pd.read_csv(OUT_DIR / "descriptor_halide_interpolation.csv", index_col=0)
res["linear"] = halide["R2series linear (end rows)"]
order = res.sort_values("R2series held-out pair GP").index  # best at the top


def clipped(ax, x, y, color, marker, size, **kw):
    x = np.asarray(x, float)
    low = x < R2_FLOOR
    ax.scatter(np.where(low, R2_FLOOR, x)[~low], np.asarray(y)[~low], color=color, marker=marker, s=size, zorder=3, **kw)
    ax.scatter(np.full(low.sum(), R2_FLOOR), np.asarray(y)[low], color=color, marker="<", s=size, zorder=3)


# --- 1. R2 per target per test ---------------------------------------------
fig, axes = plt.subplots(1, len(TESTS), figsize=(2.3 * len(TESTS) + 1.6, 0.27 * len(order) + 1.6), sharey=True)
yy = np.arange(len(order))
for ax, (key, title) in zip(axes, TESTS):
    t, g = res.loc[order, f"{key} tree"], res.loc[order, f"{key} GP"]
    for y0, a, b in zip(yy, t.clip(lower=R2_FLOOR), g.clip(lower=R2_FLOOR)):
        ax.plot([a, b], [y0, y0], color="#d6d5d0", lw=2, zorder=1)
    if key == "R2series held-out pair":
        lin = res.loc[order, "linear"]
        clipped(ax, lin, yy, BASE, "|", 60, linewidths=2)
    clipped(ax, t, yy, TREE, "o", 30, edgecolors="white", linewidths=0.8)
    clipped(ax, g, yy, GP, "o", 30, edgecolors="white", linewidths=0.8)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(R2_FLOOR - 0.15, 1.05)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_title(title + f"\n{{\\footnotesize median: tree {t.median():.2f}, GP {g.median():.2f}}}", fontsize=10, color=INK)
    ax.set_xlabel("$R^2$", fontsize=11)
    ax.tick_params(labelsize=9)
axes[0].set_yticks(yy, [label(t) + (r"$^\ast$" if halide.loc[t, "shortlist"] else "") for t in order], fontsize=9)
axes[0].set_ylim(-0.7, len(order) - 0.3)
handles = [plt.Line2D([], [], color=TREE, marker="o", ls="", label="gradient-boosted trees"),
           plt.Line2D([], [], color=GP, marker="o", ls="", label="Gaussian process"),
           plt.Line2D([], [], color=BASE, marker="|", ls="", markersize=9, mew=2, label="linear end-member baseline"),
           plt.Line2D([], [], color=MUTED, marker="<", ls="", label=f"below {R2_FLOOR:.0f}")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.55, -0.01))
fig.text(0.01, 0.005, r"$^\ast$ shortlisted target", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(HERE / "fig_r2_by_test.png", dpi=200, bbox_inches="tight")

# --- 2. GP vs tree scatter ---------------------------------------------------
fig, axes = plt.subplots(1, len(TESTS), figsize=(2.6 * len(TESTS), 2.9))
for ax, (key, title) in zip(axes, TESTS):
    t, g = res[f"{key} tree"].clip(lower=R2_FLOOR), res[f"{key} GP"].clip(lower=R2_FLOOR)
    lo = min(t.min(), g.min()) - 0.1
    ax.fill_between([lo, 1.05], [lo, 1.05], 1.05, color=GP, alpha=0.06, lw=0)
    ax.plot([lo, 1.05], [lo, 1.05], color=MUTED, lw=0.8, ls="--")
    ax.scatter(t, g, color=GP, s=20, edgecolors="white", linewidths=0.6, zorder=3)
    for name in ["s_300K", "log_k_B"]:
        ax.annotate(label(name), (t[name], g[name]), xytext=(3, -8), textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_xlim(lo, 1.05); ax.set_ylim(lo, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("tree $R^2$", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.text(0.04, 0.96, f"GP better: {(res[f'{key} GP'] > res[f'{key} tree']).sum()}/{len(res)}",
            transform=ax.transAxes, fontsize=8, color=INK, va="top")
axes[0].set_ylabel("GP $R^2$", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_r2_scatter.png", dpi=200, bbox_inches="tight")

# --- 3. example series (held-out-pair predictions) ---------------------------
preds = {t: pd.read_csv(HERE / f"pred_{t}.csv").set_index("run_name") for t in res.index}
show_t = ["s_300K", "rel_centroid_B", "log_k_B", "rel_width_X"]
counts = dm[has_ends].groupby("series").size()
show_s = [s for s in ["Cs | Pb Cl/Br", "Cs | Pb Br/I", "Cs | Ag;Bi Cl/Br", "Cs | Na;In Cl/Br"] if s in counts.index]
show_s = (show_s + [s for s in counts.sort_values(ascending=False).index if s not in show_s])[:4]
fig, axes = plt.subplots(len(show_t), len(show_s), figsize=(3.0 * len(show_s), 2.3 * len(show_t)), squeeze=False)
for j, s in enumerate(show_s):
    d_s = dm[dm["series"] == s].sort_values("x2")
    e1, e2 = d_s["end_X1"].iloc[0], d_s["end_X2"].iloc[0]
    for i, tgt in enumerate(show_t):
        ax = axes[i, j]
        ends = [end_value.loc[e1, tgt], end_value.loc[e2, tgt]]
        p = preds[tgt].loc[d_s["run_name"]]
        ax.plot([0, 1], ends, ls="--", color=BASE, lw=1.5)
        ax.plot(d_s["x2"], p["pair tree"], color=TREE, lw=2, marker="o", ms=3.5)
        ax.plot(d_s["x2"], p["pair GP"], color=GP, lw=2, marker="o", ms=3.5)
        ax.scatter(np.r_[0, d_s["x2"], 1], np.r_[ends[0], Tm.loc[d_s.index, tgt], ends[1]], color=INK, s=16, zorder=4)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_title(s.replace("_", " ").replace(";", "/").replace(" | ", ", "), fontsize=9.5)
        if j == 0:
            ax.set_ylabel(label(tgt), fontsize=10)
        if i == len(show_t) - 1:
            ax.set_xlabel(f"x ({d_s['X2'].iloc[0]} share)", fontsize=9.5)
handles = [plt.Line2D([], [], color=INK, marker="o", ls="", label="MLIP (end members at x = 0, 1)"),
           plt.Line2D([], [], color=TREE, lw=2, label="trees"),
           plt.Line2D([], [], color=GP, lw=2, label="GP"),
           plt.Line2D([], [], color=BASE, lw=1.5, ls="--", label="linear end-member baseline")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
fig.suptitle("Held-out halide pair: models trained on MP + the other two pairs' mixtures", fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(HERE / "fig_series.png", dpi=200, bbox_inches="tight")
print("wrote", *sorted(p.name for p in HERE.glob("fig_*.png")))
