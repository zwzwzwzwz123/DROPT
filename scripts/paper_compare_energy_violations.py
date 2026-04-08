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
    resolve_bcfixclean_smalloffice_run_dirs,
)


LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure")
PROFILE = "legacy"
OUT_BASENAME_ENERGY = "compare_energy"
OUT_BASENAME_VIOLATIONS = "compare_violations"
OUT_BASENAME_PARETO = "compare_energy_violations"
SUMMARY_K = 5


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


def _build_records(log_root: str) -> List[Dict[str, object]]:
    resolved = _resolve_run_dirs(log_root)
    records: List[Dict[str, object]] = []
    for spec in RUN_SPECS:
        if spec.matcher not in resolved:
            continue
        run_name = resolved[spec.matcher]
        run_dir = os.path.join(log_root, run_name)
        event_path = _event_file(run_dir)
        energy = _mean_last_k(_load_series(event_path, "test/avg_energy"))
        violations = _mean_last_k(_load_series(event_path, "test/avg_violations"))
        records.append(
            {
                "label": spec.label,
                "run_name": run_name,
                "energy": energy,
                "violations": violations,
                "color": spec.color,
                "kind": spec.kind,
            }
        )
    return records


def _save_mapping_csv(records: Sequence[Dict[str, object]], out_dir: str) -> None:
    lines = ["Paper Name,Source Log Directory,Energy_kWh,Comfort Violations"]
    for row in records:
        lines.append(
            f"{row['label']},{row['run_name']},{float(row['energy']):.6f},{float(row['violations']):.6f}"
        )
    with open(os.path.join(out_dir, "compare_energy_violations_mapping.csv"), "w", encoding="utf-8-sig") as fh:
        fh.write("\n".join(lines) + "\n")


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
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

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    bar_kw = dict(height=0.68, edgecolor="black", linewidth=0.5, alpha=0.96)
    ax.barh(y, values, color=colors, **bar_kw)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(StrMethodFormatter(tick_fmt))

    if metric_key == "energy":
        value_pad = max(60.0, values.max() * 0.12)
        x_extra = value_pad * 1.55
        text_shift = value_pad * 0.12
    else:
        value_pad = max(0.12, values.max() * 0.18)
        x_extra = value_pad * 2.6
        text_shift = value_pad * 0.22

    ax.set_xlim(0, values.max() + x_extra)

    for idx, value in enumerate(values):
        ax.text(
            value + text_shift,
            idx,
            format(value, value_fmt),
            va="center",
            ha="left",
            fontsize=9,
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

    excluded_labels = {"SAC", "SAC+MPC"}
    plot_records = [row for row in records if str(row["label"]) not in excluded_labels]
    sac_record = next((row for row in records if str(row["label"]) == "SAC"), None)
    sac_mpc_record = next((row for row in records if str(row["label"]) == "SAC+MPC"), None)

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)

    frontier_mask = _pareto_mask(plot_records)
    frontier_records = [row for row, keep in zip(plot_records, frontier_mask.tolist()) if keep]
    frontier_records = sorted(frontier_records, key=lambda r: float(r["energy"]))

    marker_by_kind = {"diffusion": "o", "baseline": "s"}

    def _draw_points(ax_obj, annotate: bool) -> None:
        for row, is_frontier in zip(plot_records, frontier_mask.tolist()):
            energy = float(row["energy"])
            violations = float(row["violations"])
            label = str(row["label"])
            color = str(row["color"])
            kind = str(row["kind"])
            marker = marker_by_kind.get(kind, "o")
            size = 92 if label == "Guided-DiffFNO" else 76
            edge = "#111111" if is_frontier else "white"
            lw = 1.2 if is_frontier else 0.8
            alpha = 0.98 if is_frontier else 0.93

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

        if frontier_records:
            xs = [float(row["energy"]) for row in frontier_records]
            ys = [float(row["violations"]) for row in frontier_records]
            ax_obj.plot(xs, ys, color="#111111", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)

        if not annotate:
            return

        offsets = {
            "Guided-DiffFNO": (18.0, -0.065),
            "DiffFNO w/o Guidance": (14.0, 0.025),
            "DiffFNO w/o Residual & Guidance": (14.0, 0.04),
            "DiffFNO w/o Residual": (16.0, 0.055),
            "Diffusion Policy (MLP backbone)": (12.0, 0.04),
            "MPC": (10.0, 0.055),
        }
        for row in plot_records:
            energy = float(row["energy"])
            violations = float(row["violations"])
            label = str(row["label"])
            dx, dy = offsets.get(label, (12.0, 0.03))
            text_x = energy + dx
            text_y = violations + dy
            if label in {"Guided-DiffFNO", "DiffFNO w/o Residual"}:
                ax_obj.annotate(
                    _short_label(label),
                    xy=(energy, violations),
                    xytext=(text_x, text_y),
                    textcoords="data",
                    fontsize=8.8,
                    color="#1f2937",
                    ha="left",
                    va="center",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.8, "color": "#6b7280"},
                )
            else:
                ax_obj.text(
                    text_x,
                    text_y,
                    _short_label(label),
                    fontsize=8.8,
                    color="#1f2937",
                    ha="left" if dx >= 0 else "right",
                    va="center",
                )

    _draw_points(ax, annotate=True)

    ax.annotate(
        "Better",
        xy=(0.08, 0.12),
        xytext=(0.22, 0.26),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#374151"},
        fontsize=9.5,
        color="#374151",
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="white", markersize=8, label="Diffusion-based"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#6b7280", markeredgecolor="white", markersize=8, label="Baseline"),
        Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.4, label="Pareto frontier"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#d1d5db",
        facecolor="white",
        columnspacing=1.4,
        handletextpad=0.5,
    )

    ax.set_xlabel("Energy consumption (kWh)")
    ax.set_ylabel("Comfort violations")
    ax.set_title("Energy-comfort Pareto comparison")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(830.0, 1200.0)
    ax.set_ylim(0.50, 2.00)
    ax.set_xticks([850, 900, 1000, 1100, 1200])
    ax.set_yticks([0.6, 1.0, 1.4, 1.8])

    note_lines: List[str] = []
    if sac_record is not None:
        note_lines.append(
            f"SAC ({float(sac_record['energy']):.0f} kWh, {float(sac_record['violations']):.2f} violations)"
        )
    if sac_mpc_record is not None:
        note_lines.append(
            f"SAC+MPC ({float(sac_mpc_record['energy']):.0f} kWh, {float(sac_mpc_record['violations']):.2f} violations)"
        )
    if note_lines:
        ax.text(
            0.985,
            0.04,
            "\n".join(note_lines) + "\nexcluded from the plot for scale clarity.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.94},
        )

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
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, PROFILE
    args = _parse_args()
    PROFILE = args.profile
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, PROFILE)
    records = _build_records(LOG_ROOT)
    out_paths = render(records, OUT_DIR)
    for path in out_paths:
        print(path)


if __name__ == "__main__":
    main()
