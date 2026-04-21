#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-ready comparison figures for building energy and comfort
violations using a fixed set of paper method names.

Each figure summarizes the last 5 test checkpoints from the selected runs and
exports both PNG and PDF into paperfigure/.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tensorboard.backend.event_processing import event_accumulator
from dropt_utils.paper_building_profiles import (
    default_out_dir,
    resolve_bcfixclean_officemedium_run_dirs,
    resolve_bcfixclean_officemedium_run_dir_groups,
    resolve_bcfixclean_smalloffice_run_dirs,
    resolve_bcfixclean_smalloffice_run_dir_groups,
)


LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure")
PROFILE = "legacy"
AGGREGATE_SEEDS = False
OUT_BASENAME_ENERGY = "compare_energy"
OUT_BASENAME_VIOLATIONS = "compare_violations"
OUT_BASENAME_PARETO = "compare_energy_violations"
SUMMARY_K = 5
BASE_FONT_SIZE = 13.5
AXIS_FONT_SIZE = 15.5
TITLE_FONT_SIZE = 15.5
TICK_FONT_SIZE = 13.2
LEGEND_FONT_SIZE = 12.8
ANNOTATION_FONT_SIZE = 12.6
SMALL_ANNOTATION_FONT_SIZE = 11.8


@dataclass(frozen=True)
class RunSpec:
    label: str
    matcher: str
    color: str
    kind: str


RUN_SPECS: Sequence[RunSpec] = (
    RunSpec("Guided-DiffFNO", "fno_guided_full", "#1f4e79", "diffusion"),
    RunSpec("DiffFNO w/o Guidance", "fno_guided_noguide_align", "#4e79a7", "diffusion"),
    RunSpec("DiffFNO w/o Residual & Guidance", "fno_guided_nores_noguide_align", "#7aa6d1", "diffusion"),
    RunSpec("DiffFNO w/o Residual", "fno_guided_nores", "#2f6b8a", "diffusion"),
    RunSpec("Diffusion Policy (MLP backbone)", "MLP", "#9ecae1", "diffusion"),
    RunSpec("MPC", "default_mpc_latest", "#6b7280", "baseline"),
    RunSpec("SAC+MPC", "sac_baseline_mpc", "#8c564b", "baseline"),
    RunSpec("SAC", "sac_baseline", "#c44e52", "baseline"),
)


def _event_file(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        if name.startswith("events.out.tfevents"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")


def _load_series(event_path: str, tag: str) -> List[Tuple[int, float]]:
    acc = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    acc.Reload()
    values = acc.Scalars(tag)
    return sorted((item.step, float(item.value)) for item in values)


def _mean_last_k(values: Iterable[Tuple[int, float]], k: int = SUMMARY_K) -> float:
    series = [value for _, value in values]
    if not series:
        return float("nan")
    kk = max(1, min(k, len(series)))
    return float(np.mean(series[-kk:]))


def _resolve_legacy_run_dirs(log_root: str) -> Dict[str, str]:
    all_dirs = [name for name in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, name))]
    resolved: Dict[str, str] = {}

    default_mpc_runs = sorted(name for name in all_dirs if name.startswith("default_mpc"))
    if not default_mpc_runs:
        raise FileNotFoundError("Could not find any default_mpc run under log_building/")
    resolved["default_mpc_latest"] = default_mpc_runs[-1]

    for name in all_dirs:
        if name.startswith("fno_guided_noguide_align"):
            resolved["fno_guided_noguide_align"] = name
        elif name.startswith("fno_guided_nores_noguide_align"):
            resolved["fno_guided_nores_noguide_align"] = name
        elif name.startswith("fno_guided_nores") and "noguide" not in name:
            resolved["fno_guided_nores"] = name
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
        elif name.startswith("sac_baseline_mpc") and "bcfix" not in name:
            resolved["sac_baseline_mpc"] = name
        elif name.startswith("sac_baseline") and "mpc" not in name and "bcfix" not in name:
            resolved["sac_baseline"] = name

    missing = [spec.matcher for spec in RUN_SPECS if spec.matcher not in resolved]
    if missing:
        raise FileNotFoundError(f"Missing run directories for matchers: {missing}")
    return resolved


def _resolve_run_dirs(log_root: str) -> Dict[str, str]:
    required = [spec.matcher for spec in RUN_SPECS]
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dirs(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dirs(log_root, required, allow_partial=True)
    return _resolve_legacy_run_dirs(log_root)


def _resolve_run_groups(log_root: str) -> Dict[str, List[str]]:
    required = [spec.matcher for spec in RUN_SPECS]
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dir_groups(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dir_groups(log_root, required, allow_partial=True)
    return {matcher: [name] for matcher, name in _resolve_legacy_run_dirs(log_root).items()}


def _build_records(log_root: str) -> List[Dict[str, object]]:
    if AGGREGATE_SEEDS:
        resolved_groups = _resolve_run_groups(log_root)
    else:
        resolved_single = _resolve_run_dirs(log_root)
        resolved_groups = {matcher: [name] for matcher, name in resolved_single.items()}
    records: List[Dict[str, object]] = []
    for spec in RUN_SPECS:
        run_names = resolved_groups.get(spec.matcher, [])
        if not run_names:
            continue
        energies: List[float] = []
        violations_list: List[float] = []
        for run_name in run_names:
            run_dir = os.path.join(log_root, run_name)
            event_path = _event_file(run_dir)
            energies.append(_mean_last_k(_load_series(event_path, "test/avg_energy")))
            violations_list.append(_mean_last_k(_load_series(event_path, "test/avg_violations")))
        energy_arr = np.asarray(energies, dtype=np.float64)
        violations_arr = np.asarray(violations_list, dtype=np.float64)
        records.append(
            {
                "label": spec.label,
                "run_name": " | ".join(run_names),
                "run_names": run_names,
                "energy": float(np.nanmean(energy_arr)),
                "energy_std": float(np.nanstd(energy_arr, ddof=0)),
                "violations": float(np.nanmean(violations_arr)),
                "violations_std": float(np.nanstd(violations_arr, ddof=0)),
                "n_runs": len(run_names),
                "color": spec.color,
                "kind": spec.kind,
            }
        )
    return records


def _save_mapping_csv(records: Sequence[Dict[str, object]], out_dir: str) -> None:
    lines = [
        "Paper Name,Source Log Directories,Num Runs,Energy Mean kWh,Energy Std,Comfort Violations Mean,Comfort Violations Std"
    ]
    for row in records:
        lines.append(
            ",".join(
                [
                    str(row["label"]),
                    '"' + str(row["run_name"]).replace('"', '""') + '"',
                    str(int(row.get("n_runs", 1))),
                    f"{float(row['energy']):.6f}",
                    f"{float(row.get('energy_std', 0.0)):.6f}",
                    f"{float(row['violations']):.6f}",
                    f"{float(row.get('violations_std', 0.0)):.6f}",
                ]
            )
        )
    with open(os.path.join(out_dir, "compare_energy_violations_mapping.csv"), "w", encoding="utf-8-sig") as fh:
        fh.write("\n".join(lines) + "\n")


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


def _render_single_metric(
    records: Sequence[Dict[str, object]],
    metric_key: str,
    xlabel: str,
    title: str,
    value_fmt: str,
    tick_fmt: str,
    out_dir: str,
    out_basename: str,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import StrMethodFormatter
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required: {exc}")

    os.makedirs(out_dir, exist_ok=True)
    _setup_matplotlib()

    ordered_records = list(records)[::-1]
    labels = [str(row["label"]) for row in ordered_records]
    colors = [str(row["color"]) for row in ordered_records]
    y = np.arange(len(labels))
    values = np.asarray([float(row[metric_key]) for row in ordered_records], dtype=np.float64)
    std_key = f"{metric_key}_std"
    errors = np.asarray([float(row.get(std_key, 0.0)) for row in ordered_records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    bar_kw = dict(height=0.68, edgecolor="black", linewidth=0.5, alpha=0.96)
    ax.barh(y, values, color=colors, **bar_kw)
    if np.any(errors > 0):
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#374151", elinewidth=1.0, capsize=3.0, zorder=3)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(StrMethodFormatter(tick_fmt))

    if metric_key == "energy":
        value_pad = max(60.0, (values + errors).max() * 0.12)
        x_extra = value_pad * 1.55
        text_shift = value_pad * 0.16
    else:
        value_pad = max(0.12, (values + errors).max() * 0.18)
        x_extra = value_pad * 3.1
        text_shift = value_pad * 0.34

    ax.set_xlim(0, float((values + errors).max()) + x_extra)

    for idx, value in enumerate(values):
        err = errors[idx] if idx < len(errors) else 0.0
        ax.text(
            value + err + text_shift,
            idx,
            format(value, value_fmt),
            va="center",
            ha="left",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#222222",
        )

    fig.patch.set_facecolor("white")

    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"{out_basename}.{ext}")
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out_paths.append(out_path)

    plt.close(fig)
    return out_paths


def _pareto_mask(records: Sequence[Dict[str, object]]) -> np.ndarray:
    pts = np.asarray([[float(r["energy"]), float(r["violations"])] for r in records], dtype=np.float64)
    keep = np.ones((pts.shape[0],), dtype=bool)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[0]):
            if i == j:
                continue
            # both objectives are minimized
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    return keep


def _short_label(label: str) -> str:
    mapping = {
        "Guided-DiffFNO": "Guided",
        "DiffFNO w/o Guidance": "NoGuide",
        "DiffFNO w/o Residual & Guidance": "NoRes+NoGuide",
        "DiffFNO w/o Residual": "NoRes",
        "Diffusion Policy (MLP backbone)": "MLP",
        "MPC": "MPC",
        "SAC+MPC": "SAC+MPC",
        "SAC": "SAC",
    }
    return mapping.get(label, label)


def _render_pareto(records: Sequence[Dict[str, object]], out_dir: str, out_basename: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required: {exc}")

    os.makedirs(out_dir, exist_ok=True)
    _setup_matplotlib()
    fig = plt.figure(figsize=(9.6, 6.35), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[6.1, 1.45, 0.98], height_ratios=[1.15, 4.15], wspace=0.035, hspace=0.045)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tm = fig.add_subplot(gs[0, 1], sharey=ax_tl)
    ax_tr = fig.add_subplot(gs[0, 2], sharey=ax_tl)
    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_tl)
    ax_bm = fig.add_subplot(gs[1, 1], sharex=ax_tm, sharey=ax_bl)
    ax_br = fig.add_subplot(gs[1, 2], sharex=ax_tr, sharey=ax_bl)

    top_axes = [ax_tl, ax_tm, ax_tr]
    bottom_axes = [ax_bl, ax_bm, ax_br]
    all_axes = top_axes + bottom_axes

    x_segments = [
        (844.0, 1068.0),
        (1776.0, 1948.0),
        (5948.0, 6010.0),
    ]
    y_bottom = (0.35, 2.05)
    y_top = (3.1, 5.75)

    frontier_mask = _pareto_mask(records)
    frontier_records = [row for row, keep in zip(records, frontier_mask.tolist()) if keep]
    frontier_records = sorted(frontier_records, key=lambda r: float(r["energy"]))
    marker_by_kind = {"diffusion": "o", "baseline": "s"}

    def _axes_for_point(energy: float, violations: float):
        x_idx = None
        for idx, (xmin, xmax) in enumerate(x_segments):
            if xmin <= energy <= xmax:
                x_idx = idx
                break
        if x_idx is None:
            return None
        if y_bottom[0] <= violations <= y_bottom[1]:
            return bottom_axes[x_idx]
        if y_top[0] <= violations <= y_top[1]:
            return top_axes[x_idx]
        return None

    def _draw_points() -> None:
        offsets = {
            "Guided-DiffFNO": (31.0, -0.11),
            "DiffFNO w/o Guidance": (22.0, 0.03),
            "DiffFNO w/o Residual & Guidance": (16.0, 0.18),
            "DiffFNO w/o Residual": (18.0, 0.08),
            "Diffusion Policy (MLP backbone)": (12.0, 0.08),
            "MPC": (8.0, 0.11),
            "SAC+MPC": (16.0, 0.22),
            "SAC": (-16.0, 0.20),
        }
        arrow_labels = {"Guided-DiffFNO", "DiffFNO w/o Residual", "SAC+MPC"}

        for row, is_frontier in zip(records, frontier_mask.tolist()):
            energy = float(row["energy"])
            violations = float(row["violations"])
            energy_std = float(row.get("energy_std", 0.0))
            violations_std = float(row.get("violations_std", 0.0))
            label = str(row["label"])
            color = str(row["color"])
            kind = str(row["kind"])
            marker = marker_by_kind.get(kind, "o")
            # Slightly larger markers help the sparse broken-axis layout feel
            # less empty without making the local cluster overcrowded.
            size = 134 if label == "Guided-DiffFNO" else 108
            edge = "#111111" if is_frontier else "white"
            lw = 1.3 if is_frontier else 0.9
            alpha = 0.98 if is_frontier else 0.93

            ax_obj = _axes_for_point(energy, violations)
            if ax_obj is None:
                continue

            ax_obj.scatter(
                energy,
                violations,
                s=size,
                c=color,
                marker=marker,
                edgecolors=edge,
                linewidths=lw,
                alpha=alpha,
                zorder=3,
            )
            if energy_std > 0 or violations_std > 0:
                ax_obj.errorbar(
                    energy,
                    violations,
                    xerr=energy_std if energy_std > 0 else None,
                    yerr=violations_std if violations_std > 0 else None,
                    fmt="none",
                    ecolor="#374151",
                    elinewidth=1.15,
                    capsize=3.2,
                    zorder=2,
                    alpha=0.9,
                )

            dx, dy = offsets.get(label, (12.0, 0.03))
            text_x = energy + dx
            text_y = violations + dy
            if label in arrow_labels:
                ax_obj.annotate(
                    _short_label(label),
                    xy=(energy, violations),
                    xytext=(text_x, text_y),
                    textcoords="data",
                    fontsize=SMALL_ANNOTATION_FONT_SIZE,
                    color="#1f2937",
                    ha="left" if dx >= 0 else "right",
                    va="center",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": "#6b7280"},
                )
            else:
                ax_obj.text(
                    text_x,
                    text_y,
                    _short_label(label),
                    fontsize=SMALL_ANNOTATION_FONT_SIZE,
                    color="#1f2937",
                    ha="left" if dx >= 0 else "right",
                    va="center",
                )

    def _draw_frontier() -> None:
        local = [row for row in frontier_records if x_segments[0][0] <= float(row["energy"]) <= x_segments[0][1]]
        if len(local) < 2:
            return
        xs = [float(row["energy"]) for row in local]
        ys = [float(row["violations"]) for row in local]
        ax_bl.plot(xs, ys, color="#111111", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)

    def _add_break_marks() -> None:
        # Follow the official broken-axis recipe: place slash markers directly in
        # axes coordinates so they stay attached to the spine endpoints.
        d = 0.55
        marker_kwargs = dict(
            marker=[(-1, -d), (1, d)],
            markersize=11,
            linestyle="none",
            color="#111111",
            mec="#111111",
            mew=1.0,
            clip_on=False,
            zorder=10,
        )

        # x-axis breaks: only on the visible bottom spines.
        ax_bl.plot([1], [0], transform=ax_bl.transAxes, **marker_kwargs)
        ax_bm.plot([0, 1], [0, 0], transform=ax_bm.transAxes, **marker_kwargs)
        ax_br.plot([0], [0], transform=ax_br.transAxes, **marker_kwargs)

        # y-axis break: only on the visible left spines.
        ax_tl.plot([0], [0], transform=ax_tl.transAxes, **marker_kwargs)
        ax_bl.plot([0], [1], transform=ax_bl.transAxes, **marker_kwargs)

    _draw_points()
    _draw_frontier()

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="white", markersize=9.6, label="Diffusion-based"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#6b7280", markeredgecolor="white", markersize=9.4, label="Baseline"),
        Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.4, label="Pareto frontier"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.00),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#d1d5db",
        facecolor="white",
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.92)

    for ax_obj, (xmin, xmax) in zip(top_axes, x_segments):
        ax_obj.set_xlim(xmin, xmax)
        ax_obj.set_ylim(*y_top)
        ax_obj.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax_obj.spines["top"].set_visible(False)
        ax_obj.spines["bottom"].set_visible(False)
        ax_obj.tick_params(bottom=False, labelbottom=False)

    for ax_obj, (xmin, xmax) in zip(bottom_axes, x_segments):
        ax_obj.set_xlim(xmin, xmax)
        ax_obj.set_ylim(*y_bottom)
        ax_obj.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax_obj.spines["top"].set_visible(False)

    for ax_obj in [ax_tm, ax_tr, ax_bm, ax_br]:
        ax_obj.spines["left"].set_visible(False)
        ax_obj.tick_params(left=False, labelleft=False)

    ax_tl.spines["right"].set_visible(False)
    ax_bl.spines["right"].set_visible(False)
    ax_tm.spines["right"].set_visible(False)
    ax_bm.spines["right"].set_visible(False)

    ax_tl.set_yticks([3.5, 4.5, 5.5])
    ax_bl.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax_bl.set_xticks([850, 900, 1000])
    ax_bm.set_xticks([1800, 1900])
    ax_br.set_xticks([5950, 6000])

    _add_break_marks()

    fig.supxlabel("Energy consumption (kWh)")
    fig.supylabel("Comfort violations")

    fig.patch.set_facecolor("white")
    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"{out_basename}.{ext}")
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out_paths.append(out_path)

    plt.close(fig)
    return out_paths


def render(records: Sequence[Dict[str, object]], out_dir: str) -> List[str]:
    if not records:
        return []
    out_paths: List[str] = []
    out_paths.extend(_render_pareto(records=records, out_dir=out_dir, out_basename=OUT_BASENAME_PARETO))
    out_paths.extend(
        _render_single_metric(
            records=records,
            metric_key="energy",
            xlabel="Energy consumption (kWh)",
            title="Energy",
            value_fmt=",.1f",
            tick_fmt="{x:,.0f}",
            out_dir=out_dir,
            out_basename=OUT_BASENAME_ENERGY,
        )
    )
    out_paths.extend(
        _render_single_metric(
            records=records,
            metric_key="violations",
            xlabel="Comfort violations",
            title="Violations",
            value_fmt=".2f",
            tick_fmt="{x:.1f}",
            out_dir=out_dir,
            out_basename=OUT_BASENAME_VIOLATIONS,
        )
    )
    _save_mapping_csv(records, out_dir)
    return out_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready energy/violation comparison figures.")
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
        help="Aggregate all matched runs for each method and display mean plus standard deviation.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, PROFILE, AGGREGATE_SEEDS
    args = _parse_args()
    PROFILE = args.profile
    AGGREGATE_SEEDS = bool(args.aggregate_seeds)
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, PROFILE)
    records = _build_records(LOG_ROOT)
    out_paths = render(records, OUT_DIR)
    for path in out_paths:
        print(path)


if __name__ == "__main__":
    main()
