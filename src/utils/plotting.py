"""Shared figure conventions: the project matplotlib style, one palette, and
the panel layouts more than one script draws.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order: a model keeps its colour across every figure.
# QE, the reference, is drawn in neutral ink rather than taking a slot.
# All 8 hues are only distinguishable side by side (grouped bars); figures
# where every series would overlap every other (dispersions, parity) are
# small multiples -- one model per panel -- instead.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PANEL_COLS = 4
REFERENCE_COLOR = "#1f1f1e"
HIGHLIGHT = SERIES_COLORS[0]  # the one series in a single-series panel
CONTEXT = "#c8c7c2"  # everything-else points behind a highlighted group
ZERO_LINE = "#8a8a86"  # zero / parity lines
GUIDE_LINE = "#c9c9c4"  # special-point verticals


def use_project_style() -> None:
    """The user's "my_style" matplotlib style, when it is installed; the
    default style otherwise (e.g. on a cluster node)."""
    try:
        plt.style.use("my_style")
    except OSError:
        pass


def grid_shape(n: int, max_cols: int = PANEL_COLS) -> tuple[int, int]:
    """(nrows, ncols) for n panels, at most max_cols per row."""
    ncols = min(n, max_cols)
    return int(np.ceil(n / ncols)), ncols


def series_colors(labels: list[str]) -> dict[str, str]:
    """label -> colour in SERIES_COLORS order; ValueError past the palette's 8."""
    if len(labels) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} series per figure (one colour each), got {len(labels)}")
    return dict(zip(labels, SERIES_COLORS))


def plot_band_panels(x: np.ndarray, curves: np.ndarray, labels: np.ndarray, titles: dict, xlabel: str,
                     out_path: Path, logx: bool = False, sharey: bool = False, title_fontsize: float = 8) -> None:
    """Small multiples, one panel per distinct label: the median of that
    group's `curves` rows with a 10-90% band. sharey=True puts every panel on
    one y scale (and keeps its ticks); otherwise each is autoscaled, shape only."""
    groups = sorted(np.unique(labels))
    nrows, ncols = grid_shape(len(groups), 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True, sharey=sharey,
                             squeeze=False)
    for ax, c in zip(axes.flat, groups):
        member = curves[labels == c]
        lo, med, hi = np.percentile(member, [10, 50, 90], axis=0)
        ax.fill_between(x, lo, hi, color=HIGHLIGHT, alpha=0.2, linewidth=0)
        ax.plot(x, med, color=HIGHLIGHT, linewidth=1.5)
        ax.set_title(titles[c], fontsize=title_fontsize)
        if not sharey:
            ax.set_yticks([])
        if logx:
            ax.set_xscale("log")
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("vDOS (norm.)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
