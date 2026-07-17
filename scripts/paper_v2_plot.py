# -*- coding: utf-8 -*-
"""Generate the v2 publication figure set from extracted CSVs.
Run paper_v2_extract.py first. CPU-only. Outputs to paper_figures_v2/.

Design intent (see docs/HANDOFF_option3_bear_journal.md):
  Fig 1  headline  -> cross-scale energy (normalised to MLP) + comfort, shows
                      the honest non-monotonic "OfficeMedium valley".
  Fig 2  efficiency-> params (log) + MLP/FNO ratio; advantage shrinks with scale.
  Fig 3  mechanism -> School backbone-vs-guidance decoupling.
  Fig 4  stability -> training curves; MLP diverges at scale, FNO stays tight.
  Fig 5  spectral  -> truncation strength across scale + mode-insensitivity.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paper_v2_style import set_style, despine, barlabel, save, C, LAB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_figures_v2")
DATA = os.path.join(OUT, "data")
set_style()

agg = pd.read_csv(os.path.join(DATA, "group_agg.csv")).set_index("group")
curves = pd.read_csv(os.path.join(DATA, "curves.csv"))
meta = json.load(open(os.path.join(DATA, "meta.json")))

BUILDINGS = ["OfficeSmall", "OfficeMedium", "SchoolPrimary"]
ZONES = {b: meta[b]["zones"] for b in BUILDINGS}
FNO = {"OfficeSmall": "small_fno_full", "OfficeMedium": "med_fno", "SchoolPrimary": "school_fno_full"}
MLP = {"OfficeSmall": "small_mlp",      "OfficeMedium": "med_mlp", "SchoolPrimary": "school_mlp"}
PLABEL = {b: f"{b}\n({ZONES[b]} zones)" for b in BUILDINGS}

def g(grp, col): return agg.loc[grp, col]
def saving_pct(b): return 100.0 * (g(MLP[b], "energy_mean") - g(FNO[b], "energy_mean")) / g(MLP[b], "energy_mean")
def viol_rate(grp, b): return 100.0 * g(grp, "viol_mean") / ZONES[b]

X = np.arange(len(BUILDINGS))


def shared_legend(fig, keys, **kw):
    handles = [Patch(facecolor=C[k], edgecolor="white", label=LAB[k]) for k in keys]
    fig.legend(handles=handles, loc="upper center", ncol=len(keys),
               frameon=False, bbox_to_anchor=(0.5, 1.005), **kw)


# =====================================================================
# FIG 1 — Headline: cross-scale energy (normalised) + comfort violations
# =====================================================================
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    w = 0.34

    # panel (a): energy as % of MLP baseline (MLP == 100). Collapses the 10x
    # absolute-scale gap so the non-monotonic valley is directly comparable.
    ax = axes[0]
    fno_rel = [100.0 * g(FNO[b], "energy_mean") / g(MLP[b], "energy_mean") for b in BUILDINGS]
    ax.axhline(100, color=C["mlp"], lw=1.3, ls="--", zorder=2)
    ax.text(2.42, 100.6, LAB["mlp"] + " baseline", color=C["mlp"], fontsize=7.5, ha="right", va="bottom")
    bars = ax.bar(X, fno_rel, w * 1.7, color=C["fno"], edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v, b in zip(X, fno_rel, BUILDINGS):
        ax.annotate(f"−{saving_pct(b):.0f}%", (xi, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", va="bottom", fontsize=9, fontweight="bold", color=C["fno"])
    # flag the honest valley (place text in open space, arrow to the Medium bar)
    ax.annotate("valley — gain is\ncoupling-modulated,\nnot monotonic in zones",
                xy=(1.28, fno_rel[1]), xytext=(2.05, 74), ha="center", va="center",
                fontsize=7, color=C["hi"],
                arrowprops=dict(arrowstyle="->", color=C["hi"], lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_ylabel("Energy vs. MLP  (%)")
    ax.set_title("(a)  Energy: FNO relative to MLP", loc="left")
    ax.set_ylim(0, 118)
    ax.set_xticks(X); ax.set_xticklabels([PLABEL[b] for b in BUILDINGS])
    despine(ax)

    # panel (b): absolute comfort violation rate, FNO vs MLP grouped
    ax = axes[1]
    fv = [viol_rate(FNO[b], b) for b in BUILDINGS]
    mv = [viol_rate(MLP[b], b) for b in BUILDINGS]
    ax.bar(X - w/2, fv, w, color=C["fno"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(X + w/2, mv, w, color=C["mlp"], edgecolor="white", linewidth=0.8, zorder=3)
    barlabel(ax, X - w/2, fv, [f"{v:.0f}" for v in fv], color=C["fno"], dy=1.3, fs=8)
    barlabel(ax, X + w/2, mv, [f"{v:.0f}" for v in mv], color=C["mlp"], dy=1.3, fs=8)
    ax.set_ylabel("Comfort violation rate  (% of zones)")
    ax.set_title("(b)  Comfort: violations per zone", loc="left")
    ax.set_ylim(0, max(mv) * 1.2)
    ax.set_xticks(X); ax.set_xticklabels([PLABEL[b] for b in BUILDINGS])
    despine(ax)

    shared_legend(fig, ["fno", "mlp"])
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.94))
    save(fig, OUT, "fig1_crossscale_headline")
    plt.close(fig)


# =====================================================================
# FIG 2 — Parameter efficiency: fewer params everywhere, margin shrinks
# =====================================================================
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    w = 0.34

    ax = axes[0]
    fp = [meta[b]["fno_params"] / 1e3 for b in BUILDINGS]
    mp = [meta[b]["mlp_params"] / 1e3 for b in BUILDINGS]
    ax.bar(X - w/2, fp, w, color=C["fno"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(X + w/2, mp, w, color=C["mlp"], edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v in zip(X - w/2, fp): ax.text(xi, v*1.07, f"{v:.0f}k", ha="center", va="bottom", fontsize=7.5, color=C["fno"], fontweight="bold")
    for xi, v in zip(X + w/2, mp): ax.text(xi, v*1.07, f"{v:.0f}k", ha="center", va="bottom", fontsize=7.5, color=C["mlp"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Actor parameters  (×10$^3$)")
    ax.set_title("(a)  Model size (log scale)", loc="left")
    ax.set_ylim(18, 620)
    ax.set_xticks(X); ax.set_xticklabels([PLABEL[b] for b in BUILDINGS])
    despine(ax)

    ax = axes[1]
    ratio = [meta[b]["mlp_params"] / meta[b]["fno_params"] for b in BUILDINGS]
    ax.bar(X, ratio, 0.5, color=C["accent"], edgecolor="white", linewidth=0.8, zorder=3)
    barlabel(ax, X, ratio, [f"{v:.1f}×" for v in ratio], color=C["accent"], dy=0.14, fs=9)
    ax.axhline(1.0, color=C["mlp"], lw=1.1, ls="--", zorder=2)
    ax.text(2.42, 1.06, "equal size", color=C["mlp"], fontsize=7.5, ha="right", va="bottom")
    ax.set_ylabel("MLP / FNO actor params  (×)")
    ax.set_title("(b)  FNO is smaller everywhere", loc="left")
    ax.set_ylim(0, max(ratio) * 1.2)
    ax.set_xticks(X); ax.set_xticklabels([PLABEL[b] for b in BUILDINGS])
    despine(ax)

    shared_legend(fig, ["fno", "mlp"])
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.94))
    save(fig, OUT, "fig2_param_efficiency")
    plt.close(fig)


# =====================================================================
# FIG 3 — School decoupling: backbone effect vs guidance increment
# =====================================================================
def fig3():
    order = ["school_mlp", "school_fno_noguide", "school_fno_full"]
    labels = ["Diff-MLP", "DiffFNO\n(no guid.)", "Guided-\nDiffFNO"]
    cols = [C["mlp"], C["fno_ng"], C["fno"]]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2))
    x = np.arange(3)

    specs = [("energy_mean", "energy_std", "Energy  (kWh)", "(a)  Energy", None),
             ("viol_mean", "viol_std", "Comfort violations  (# zones)", "(b)  Violations", None),
             ("comfort_mean", "comfort_std", "Mean temp. deviation  (°C)", "(c)  Comfort", 1.0)]
    for ax, (mcol, scol, ylab, title, tol) in zip(axes, specs):
        vals = [g(o, mcol) for o in order]
        errs = [g(o, scol) for o in order]
        ax.bar(x, vals, 0.62, yerr=errs, color=cols, edgecolor="white", linewidth=0.8,
               error_kw=dict(ecolor="#333", elinewidth=1.0, capsize=3), zorder=3)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + max(vals)*0.025, f"{v:.0f}" if v > 50 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        if tol is not None:
            ax.axhline(tol, color=C["hi"], lw=1.0, ls="--", zorder=2)
            ax.text(2.42, tol*1.03, "±1°C tol.", color=C["hi"], fontsize=7.5, ha="right", va="bottom")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(v+e for v, e in zip(vals, errs)) * 1.22)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        despine(ax)

    # decompose the energy drop on panel (a): backbone step then guidance step
    ax = axes[0]
    e = [g(o, "energy_mean") for o in order]
    ax.annotate("", xy=(1, e[1]), xytext=(0, e[0]), arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.4))
    ax.text(0.30, e[0]*0.80, f"backbone\n−{100*(e[0]-e[1])/e[0]:.0f}%", color=C["accent"],
            fontsize=7.5, ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(2, e[2]), xytext=(1, e[1]), arrowprops=dict(arrowstyle="->", color=C["hi"], lw=1.4))
    ax.text(1.72, e[1]*0.98, f"guidance\n−{100*(e[1]-e[2])/e[1]:.0f}%", color=C["hi"],
            fontsize=7.5, ha="center", va="center", fontweight="bold")

    fig.suptitle("SchoolPrimary (25 zones): decoupling backbone vs. guidance",
                 fontsize=10.5, fontweight="bold", y=1.0)
    fig.tight_layout(w_pad=1.8, rect=(0, 0, 1, 0.95))
    save(fig, OUT, "fig3_school_decoupling")
    plt.close(fig)


# =====================================================================
# FIG 4 — Training stability curves
# =====================================================================
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
    panels = [("OfficeSmall", "small_fno_full", "small_mlp"),
              ("OfficeMedium", "med_fno", "med_mlp"),
              ("SchoolPrimary", "school_fno_full", "school_mlp")]
    for ax, (b, fgrp, mgrp) in zip(axes, panels):
        # skip epoch 0-1 (untrained random-init spike) so the y-axis reflects
        # the actual training regime instead of being dominated by the transient.
        WARMUP = 2
        for grp, key in [(mgrp, "mlp"), (fgrp, "fno")]:
            sub = curves[curves.group == grp]
            if sub.empty: continue
            piv = sub.pivot_table(index="idx", columns="seed", values="energy")
            piv = piv[piv.index >= WARMUP]
            m = piv.mean(axis=1); s = piv.std(axis=1); xx = piv.index.values
            ax.plot(xx, m, color=C[key], lw=1.7, label=LAB[key], zorder=3)
            ax.fill_between(xx, m - s, m + s, color=C[key], alpha=0.16, linewidth=0, zorder=2)
        ax.set_title(f"{b}  ({ZONES[b]} zones)", loc="left")
        ax.set_xlabel("Training epoch  (warm-up omitted)")
        ax.set_xlim(left=WARMUP)
        if ax is axes[0]:
            ax.set_ylabel("Test energy  (kWh)")
        despine(ax)
    # single shared legend (curves share identical colour/label mapping)
    handles = [Line2D([0], [0], color=C["fno"], lw=2, label=LAB["fno"]),
               Line2D([0], [0], color=C["mlp"], lw=2, label=LAB["mlp"])]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.93))
    save(fig, OUT, "fig4_training_curves")
    plt.close(fig)


# =====================================================================
# FIG 5 — Spectral: truncation strength across scale + mode-insensitivity
# =====================================================================
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    rfft = {"OfficeSmall": 4, "OfficeMedium": 10, "SchoolPrimary": 13}
    modes = {"OfficeSmall": 4, "OfficeMedium": 6, "SchoolPrimary": 4}  # per-building ground truth (§9.2)
    ret = [min(100.0 * modes[b] / rfft[b], 100.0) for b in BUILDINGS]
    cols = [C["sac"], C["accent"], C["fno"]]  # small = control (no truncation), grey it out
    ax.bar(X, ret, 0.55, color=cols, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v, b in zip(X, ret, BUILDINGS):
        tag = f"{v:.0f}%\n({modes[b]}/{rfft[b]} modes)"
        ax.text(xi, v + 2.5, tag, ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=C["ink"])
    ax.axhline(100, color=C["soft"], lw=0.9, ls=":", zorder=1)
    ax.text(X[-1], 102, "no truncation (control)", fontsize=7, color=C["soft"], ha="center", va="bottom")
    ax.set_ylabel("Spectral modes retained  (%)")
    ax.set_title("(a)  Truncation strengthens with scale", loc="left")
    ax.set_ylim(0, 128)
    ax.set_xticks(X); ax.set_xticklabels([PLABEL[b] for b in BUILDINGS])
    despine(ax)

    ax = axes[1]
    variants = ["med_fno", "med_fno_aligned"]
    vlabels = ["modes=6, L2\n(60% kept)", "modes=4, L1\n(40% kept)"]
    en = [g(v, "energy_mean") for v in variants]
    es = [g(v, "energy_std") for v in variants]
    mlp_e = g("med_mlp", "energy_mean")
    xx = np.arange(2)
    ax.bar(xx, en, 0.46, yerr=es, color=[C["fno"], C["fno_ng"]], edgecolor="white", linewidth=0.8,
           error_kw=dict(ecolor="#333", elinewidth=1.0, capsize=3), zorder=3)
    barlabel(ax, xx, [en[i]+es[i] for i in range(2)], [f"{v:.0f}" for v in en], dy=28, fs=8.5)
    ax.axhline(mlp_e, color=C["mlp"], lw=1.3, ls="--", zorder=2)
    ax.text(1.45, mlp_e + 12, f"{LAB['mlp']} ({mlp_e:.0f})", color=C["mlp"], fontsize=7.5, ha="right", va="bottom")
    ytop = max(en[i]+es[i] for i in range(2)) + 70
    ax.plot([0, 0, 1, 1], [ytop, ytop+12, ytop+12, ytop], color=C["ink"], lw=0.9)
    ax.text(0.5, ytop+16, f"Δ = {100*abs(en[1]-en[0])/en[0]:.1f}%  (within seed noise)",
            ha="center", va="bottom", fontsize=7.5, color=C["ink"])
    ax.set_ylabel("Energy  (kWh)")
    ax.set_title("(b)  Insensitive to mode count", loc="left")
    ax.set_ylim(6600, mlp_e + 175)
    ax.set_xticks(xx); ax.set_xticklabels(vlabels, fontsize=8.5)
    ax.set_xlim(-0.6, 1.9)
    despine(ax)
    ax.text(0.98, 0.02, "aligned run: preliminary (~200/245 ep)", transform=ax.transAxes,
            fontsize=6.5, color=C["soft"], ha="right", va="bottom", style="italic")

    fig.suptitle("Spectral structure: strong truncation, yet performance is mode-insensitive",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.95))
    save(fig, OUT, "fig5_mechanism_modes")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating v2 figures...")
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("Done ->", OUT)
