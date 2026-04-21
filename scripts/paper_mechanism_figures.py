#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-ready mechanism figures for the Guided-DiffFNO paper.

Outputs:
1. critic_q_mc_return.(png|pdf)
2. multizone_action_coordination.(png|pdf)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dropt_utils.paper_building_profiles import (
    default_out_dir,
    pick_run_dirs_by_seed_preference,
    resolve_bcfixclean_officemedium_run_dirs,
    resolve_bcfixclean_officemedium_run_dir_groups,
    resolve_bcfixclean_smalloffice_run_dirs,
    resolve_bcfixclean_smalloffice_run_dir_groups,
)

LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure")
PROFILE = "legacy"
AGGREGATE_SEEDS = False
RUN_GROUPS: Dict[str, List[str]] = {}
WINDOW_START = 48
WINDOW_END = 120
REP_EPISODE = 0
QMC_OUT_BASENAME = "critic_q_mc_return"
BASE_FONT_SIZE = 11
AXIS_FONT_SIZE = 12.5
TITLE_FONT_SIZE = 12.5
TICK_FONT_SIZE = 10.8
LEGEND_FONT_SIZE = 10.2
ANNOTATION_FONT_SIZE = 10.0
PANEL_TITLE_FONT_SIZE = 11.2


@dataclass(frozen=True)
class RunSpec:
    label: str
    matcher: str
    color: str


RUNS: Sequence[RunSpec] = (
    RunSpec("Guided-DiffFNO", "fno_guided_full", "#1f77b4"),
    RunSpec("DiffFNO w/o Guidance", "fno_guided_noguide_align", "#ff7f0e"),
    RunSpec("Diffusion-MLP", "MLP", "#17becf"),
)


ZONE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.linewidth": 0.8,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig, out_basename: str) -> List[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUT_DIR, f"{out_basename}.{ext}")
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out_paths.append(out_path)
    return out_paths


def _resolve_legacy_run_dirs(log_root: str) -> Dict[str, str]:
    all_dirs = [name for name in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, name))]
    resolved: Dict[str, str] = {}
    for name in all_dirs:
        if name.startswith("fno_guided_noguide_align"):
            resolved["fno_guided_noguide_align"] = name
        elif (
            name.startswith("fno_guided")
            and "noguide" not in name
            and "nores" not in name
            and "_m2_" not in name
            and "_m8_" not in name
            and "OfficeSmall" not in name
            and "bcfix_clean" not in name
        ):
            resolved["fno_guided_full"] = name
        elif name.startswith("MLP"):
            resolved["MLP"] = name
    missing = [spec.matcher for spec in RUNS if spec.matcher not in resolved]
    if missing:
        raise FileNotFoundError(f"Missing run directories for: {missing}")
    return resolved


def _resolve_run_dirs(log_root: str) -> Dict[str, str]:
    required = [spec.matcher for spec in RUNS]
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dirs(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dirs(log_root, required, allow_partial=True)
    return _resolve_legacy_run_dirs(log_root)


def _resolve_run_groups(log_root: str) -> Dict[str, List[str]]:
    required = [spec.matcher for spec in RUNS]
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dir_groups(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dir_groups(log_root, required, allow_partial=True)
    return {matcher: [name] for matcher, name in _resolve_legacy_run_dirs(log_root).items()}


def _load_npz(run_name: str, filename: str) -> np.lib.npyio.NpzFile:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _quantile_bin_stats(x: np.ndarray, y: np.ndarray, bins: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    x_centers: List[float] = []
    y_means: List[float] = []
    y_errs: List[float] = []
    for idx in range(bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if not np.any(mask):
            continue
        x_sel = x[mask]
        y_sel = y[mask]
        x_centers.append(float(np.mean(x_sel)))
        y_means.append(float(np.mean(y_sel)))
        y_errs.append(float(np.std(y_sel, ddof=0)))
    return (
        np.asarray(x_centers, dtype=np.float64),
        np.asarray(y_means, dtype=np.float64),
        np.asarray(y_errs, dtype=np.float64),
    )


def plot_q_vs_mc(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    spec = next(item for item in RUNS if item.label == "Guided-DiffFNO")
    run_names = RUN_GROUPS.get(spec.matcher, []) if AGGREGATE_SEEDS else []
    if not run_names and spec.matcher in run_map:
        run_names = [run_map[spec.matcher]]
    if not run_names:
        return []
    q_parts: List[np.ndarray] = []
    mc_parts: List[np.ndarray] = []
    for run_name in run_names:
        data = _load_npz(run_name, "critic_q_vs_return.npz")
        q_parts.append(np.asarray(data["q_values"], dtype=np.float64))
        mc_parts.append(np.asarray(data["mc_returns"], dtype=np.float64))
    q = np.concatenate(q_parts, axis=0)
    mc = np.concatenate(mc_parts, axis=0)

    pearson = _pearson(q, mc)
    spearman = _spearman(q, mc)
    coeff = np.polyfit(q, mc, deg=1)
    x_line = np.linspace(float(np.min(q)), float(np.max(q)), 100)
    y_line = coeff[0] * x_line + coeff[1]
    q_bin, mc_bin, mc_err = _quantile_bin_stats(q, mc, bins=7)

    fig, ax = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    hb = ax.hexbin(
        q,
        mc,
        gridsize=28,
        mincnt=1,
        cmap="Blues",
        linewidths=0.0,
        alpha=0.9,
        zorder=1,
    )
    ax.plot(x_line, y_line, color="#111827", linewidth=1.6, linestyle="--", label="Linear fit", zorder=2)
    if q_bin.size > 0:
        ax.errorbar(
            q_bin,
            mc_bin,
            yerr=mc_err,
            fmt="o-",
            color="#7c2d12",
            ecolor="#9a3412",
            elinewidth=1.0,
            capsize=2.8,
            markersize=4.8,
            linewidth=1.6,
            label="Binned mean ± std",
            zorder=3,
        )
    ax.set_xlabel(r"Critic estimate $Q(s,a)$")
    ax.set_ylabel("Monte Carlo return")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(hb, ax=ax, shrink=0.88, pad=0.02, label="Sample count")
    ax.text(
        0.03,
        0.97,
        (
            f"{'3-seed aggregate' if AGGREGATE_SEEDS else 'Representative seed'}\n"
            f"Pearson $r={pearson:.3f}$\n"
            f"Spearman $\\rho={spearman:.3f}$\n"
            f"$N={q.size}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.95},
    )
    ax.legend(loc="lower right", frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, QMC_OUT_BASENAME)
    plt.close(fig)
    return paths


def _window_actions(run_name: str) -> np.ndarray:
    traj = _load_npz(run_name, "trajectories.npz")
    return np.asarray(traj["actions"][REP_EPISODE, WINDOW_START:WINDOW_END], dtype=np.float64)


def _pairwise_corr_mean(actions: np.ndarray) -> float:
    vals: List[float] = []
    for i in range(actions.shape[1]):
        for j in range(i + 1, actions.shape[1]):
            vals.append(float(np.corrcoef(actions[:, i], actions[:, j])[0, 1]))
    return float(np.nanmean(vals))


def _cross_zone_std_mean(actions: np.ndarray) -> float:
    return float(np.mean(np.std(actions, axis=1)))


def plot_multizone_coordination(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    selected = [
        next(item for item in RUNS if item.label == "Guided-DiffFNO"),
        next(item for item in RUNS if item.label == "DiffFNO w/o Guidance"),
        next(item for item in RUNS if item.label == "Diffusion-MLP"),
    ]
    selected = [spec for spec in selected if spec.matcher in run_map]
    if not selected:
        return []

    fig, axes = plt.subplots(len(selected), 1, figsize=(7.4, 2.55 * len(selected)), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(WINDOW_END - WINDOW_START)

    for ax, spec in zip(axes, selected):
        run_name = run_map[spec.matcher]
        actions = _window_actions(run_name)
        mean_pair_corr = _pairwise_corr_mean(actions)
        mean_cross_zone_std = _cross_zone_std_mean(actions)

        for zone_idx in range(actions.shape[1]):
            ax.plot(
                x,
                actions[:, zone_idx],
                linewidth=1.8,
                color=ZONE_COLORS[zone_idx % len(ZONE_COLORS)],
                label=f"Zone {zone_idx + 1}",
            )

        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"{spec.label}: mean pairwise corr. = {mean_pair_corr:.3f}, mean cross-zone std. = {mean_cross_zone_std:.3f}",
            fontsize=PANEL_TITLE_FONT_SIZE,
        )

    axes[-1].set_xlabel("Hour")
    for ax in axes:
        ax.set_ylabel("Action")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "multizone_action_coordination")
    plt.close(fig)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready mechanism figures.")
    parser.add_argument(
        "--profile",
        choices=["legacy", "bcfixclean_smalloffice", "bcfixclean_officemedium_partial"],
        default="legacy",
        help="Run-selection profile used to resolve log_building directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for the generated figures.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate seed-dependent statistical figures while keeping coordination plots representative.",
    )
    parser.add_argument(
        "--qmc-basename",
        type=str,
        default="critic_q_mc_return",
        help="Output basename for the critic-Q-vs-return figure.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, PROFILE, AGGREGATE_SEEDS, RUN_GROUPS, QMC_OUT_BASENAME
    args = _parse_args()
    PROFILE = args.profile
    AGGREGATE_SEEDS = bool(args.aggregate_seeds)
    QMC_OUT_BASENAME = str(args.qmc_basename)
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, PROFILE)
    if AGGREGATE_SEEDS:
        RUN_GROUPS = _resolve_run_groups(LOG_ROOT)
        run_map = pick_run_dirs_by_seed_preference(LOG_ROOT, RUN_GROUPS)
    else:
        RUN_GROUPS = {}
        run_map = _resolve_run_dirs(LOG_ROOT)
    outputs: List[str] = []
    outputs.extend(plot_q_vs_mc(run_map))
    outputs.extend(plot_multizone_coordination(run_map))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
