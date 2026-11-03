"""Matplotlib figure helpers shared by all stages.

Colours follow the entity (model / method label) and are assigned in the fixed order in
which labels first appear, so "gp" has the same colour on every chart of a run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DASHES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (8, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
_COLOR: dict[str, int] = {}


def slot(label: str) -> int:
    if label not in _COLOR:
        _COLOR[label] = len(_COLOR) % len(PALETTE)
    return _COLOR[label]


def color(label):
    return PALETTE[slot(label)]


def dash(label):
    return DASHES[slot(label)]


def marker(label):
    return MARKERS[slot(label)]


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.axisbelow": True,
    })


style()


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def grid(n, ncol=3, w=4.2, h=3.4, **kw):
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False, **kw)
    for a in axes.flat[n:]:
        a.set_visible(False)
    return fig, axes.flat[:n]


def bars(ax, labels, values, err=None, colors=None, ref=None, ref_label=None, unit=""):
    x = np.arange(len(labels))
    cs = colors or [color(l) for l in labels]
    ax.bar(x, values, yerr=err, color=cs, width=0.62, capsize=3, edgecolor="white", linewidth=1)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:,.0f}" if abs(v) >= 10 else f"{v:.3g}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, labels, rotation=25 if max(map(len, labels)) > 8 else 0, ha="right" if max(map(len, labels)) > 8 else "center")
    if ref is not None:
        ax.axhline(ref, color=INK2, ls="--", lw=1.2)
        ax.annotate(ref_label or "", (len(labels) - 0.5, ref), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK2)
    ax.set_ylabel(unit)
    ax.grid(axis="x", visible=False)
