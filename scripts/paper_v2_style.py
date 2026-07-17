# -*- coding: utf-8 -*-
"""Shared publication style for the v2 figure set (top-conference aesthetics)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- colorblind-safe, muted palette ----
C = {
    "fno":     "#12507b",   # deep navy  -> Guided-DiffFNO (Full)
    "fno_ng":  "#6ba7cf",   # light blue -> FNO NoGuide
    "mlp":     "#e0812f",   # warm orange-> Diff-MLP
    "sac":     "#9aa0a6",   # grey       -> SAC baselines / reference
    "accent":  "#2a9d8f",   # teal accent
    "hi":      "#c1272d",   # red highlight
    "ink":     "#222222",
    "soft":    "#6b7076",   # muted label grey
    "grid":    "#e2e2e2",
}

# canonical labels reused across every figure so the legend never drifts
LAB = {"fno": "Guided-DiffFNO", "fno_ng": "DiffFNO (no guidance)", "mlp": "Diff-MLP"}

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": C["ink"],
        "axes.titleweight": "bold",
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.1,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,   # editable text in vector PDF
        "ps.fonttype": 42,
    })

def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:  ax.spines["left"].set_visible(False)
    if not bottom: ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)        # y-only gridlines everywhere
    ax.margins(x=0.04)

def barlabel(ax, xs, ys, texts, color=None, dy=0.0, fs=8.5, weight="bold", va="bottom"):
    """Direct value labels above bars; dy is an absolute y-offset in data units."""
    for x, y, t in zip(xs, ys, texts):
        ax.text(x, y + dy, t, ha="center", va=va, fontsize=fs,
                fontweight=weight, color=color if color else C["ink"])

def save(fig, outdir, name):
    import os
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"))
    print("  saved", name)
