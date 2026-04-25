#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a conference-oriented OfficeSmall multiseed figure set where the
no-guidance residual DiffFNO model is treated as the main method.

Outputs:
1. compare_energy_violations_noguide_main.(png|pdf)
2. ablation_summary_heatmap_noguide_main.(png|pdf)
3. smalloffice_physical_psd_compare_noguide_main.(png|pdf)

The figures are exported into a fresh output folder together with mapping and
summary files so the conference narrative is kept separate from the journal
version that emphasizes guidance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from tensorboard.backend.event_processing import event_accumulator

from dropt_utils.paper_building_profiles import (
    pick_run_dirs_by_seed_preference,
    resolve_bcfixclean_smalloffice_run_dir_groups,
)
from main_building_fno_guided_bcfix_clean import make_building_env_bcfix_clean


LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "paperfigure_bcfixclean_smalloffice_multiseed_noguide_main")
LOW_FREQ_CPD = 2.0
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
    kind: str = "diffusion"


PARETO_METHODS: Sequence[RunSpec] = (
    RunSpec("DiffFNO", "fno_guided_noguide_align", "#1f4e79", "diffusion"),
    RunSpec("DiffFNO w/o Residual", "fno_guided_nores_noguide_align", "#4e79a7", "diffusion"),
    RunSpec("Diffusion-MLP", "MLP", "#9ecae1", "diffusion"),
    RunSpec("MPC", "default_mpc_latest", "#6b7280", "baseline"),
    RunSpec("SAC+MPC", "sac_baseline_mpc", "#8c564b", "baseline"),
    RunSpec("SAC", "sac_baseline", "#c44e52", "baseline"),
)

HEATMAP_METHODS: Sequence[RunSpec] = (
    RunSpec("DiffFNO", "fno_guided_noguide_align", "#1f4e79"),
    RunSpec("DiffFNO w/o Residual", "fno_guided_nores_noguide_align", "#4e79a7"),
    RunSpec("Diffusion-MLP", "MLP", "#9ecae1"),
)

PSD_METHODS: Sequence[RunSpec] = (
    RunSpec("DiffFNO", "fno_guided_noguide_align", "#1f4e79"),
    RunSpec("Diffusion-MLP", "MLP", "#17becf"),
)


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


def _save_figure(fig, out_dir: str, out_basename: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"{out_basename}.{ext}")
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out_paths.append(out_path)
    return out_paths


def _find_event_file(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        if name.startswith("events.out.tfevents"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")


def _load_scalar_series(event_path: str, tag: str) -> List[Tuple[int, float]]:
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
    if tag not in acc.Tags().get("scalars", []):
        return []
    return sorted((item.step, float(item.value)) for item in acc.Scalars(tag))


def _mean_last_k(values: Iterable[Tuple[int, float]], k: int = SUMMARY_K) -> float:
    series = [value for _, value in values]
    if not series:
        return float("nan")
    kk = max(1, min(k, len(series)))
    return float(np.mean(series[-kk:]))


def _load_metadata(run_name: str) -> Dict[str, object]:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "paper_metadata.pkl")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    args = obj.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Invalid metadata args in {path}")
    return args


def _load_npz(run_name: str, filename: str = "trajectories.npz") -> np.lib.npyio.NpzFile:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _run_path(run_name: str) -> str:
    return os.path.join(LOG_ROOT, run_name)


def _comfort_rate(run_name: str) -> float:
    traj = _load_npz(run_name)
    violations = np.asarray(traj["comfort_violations"], dtype=np.float64)
    lengths = np.asarray(traj["lengths"], dtype=np.int32)
    action_dim = int(traj["actions"].shape[-1])
    total_violations = 0.0
    total_slots = 0
    for ep_idx, length in enumerate(lengths.tolist()):
        total_violations += float(np.sum(violations[ep_idx, : int(length)]))
        total_slots += int(length) * action_dim
    if total_slots <= 0:
        return float("nan")
    return float(1.0 - total_violations / total_slots)


def _action_mse(run_name: str) -> float:
    traj = _load_npz(run_name)
    actions = np.asarray(traj["actions"], dtype=np.float64)
    lengths = np.asarray(traj["lengths"], dtype=np.int32)
    deltas: List[np.ndarray] = []
    for ep_idx, length in enumerate(lengths.tolist()):
        if int(length) <= 1:
            continue
        ep_actions = actions[ep_idx, : int(length)]
        diff = np.diff(ep_actions, axis=0)
        deltas.append(np.mean(np.square(diff), axis=1))
    if not deltas:
        return float("nan")
    return float(np.mean(np.concatenate(deltas, axis=0)))


def _convergence_auc(run_name: str) -> float:
    event_path = _find_event_file(_run_path(run_name))
    series = _load_scalar_series(event_path, "test/reward")
    if not series:
        return float("nan")
    steps = np.asarray([step for step, _ in series], dtype=np.float64)
    rewards = np.asarray([value for _, value in series], dtype=np.float64)
    return float(np.trapz(rewards, steps))


def _minmax(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not higher_is_better:
        arr = -arr
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isclose(vmax, vmin):
        return np.ones_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _resolve_run_groups(required_matchers: Iterable[str]) -> Dict[str, List[str]]:
    return resolve_bcfixclean_smalloffice_run_dir_groups(LOG_ROOT, required_matchers)


def _load_or_compute_action_psd(run_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "actions_welch_psd.npz")
    if os.path.exists(path):
        data = np.load(path)
        return (
            np.asarray(data["frequency_hz"], dtype=np.float64),
            np.asarray(data["psd_mean"], dtype=np.float64),
        )

    traj = _load_npz(run_name)
    actions = np.asarray(traj["actions"], dtype=np.float64)
    lengths = np.asarray(traj["lengths"], dtype=np.int32)
    valid_lengths = [int(length) for length in lengths.tolist() if int(length) >= 8]
    if not valid_lengths:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    from scipy.signal import welch  # type: ignore

    fs_hz = 1.0 / 3600.0
    base_nperseg = min(64, min(valid_lengths))
    psd_values: List[np.ndarray] = []
    freq_hz: np.ndarray | None = None

    for ep_idx, length in enumerate(lengths.tolist()):
        length = int(length)
        if length < 8:
            continue
        ep_actions = actions[ep_idx, :length]
        for action_idx in range(ep_actions.shape[1]):
            nperseg = min(base_nperseg, length)
            noverlap = min(max(0, nperseg // 2), max(0, length - 1))
            freq_local, psd_local = welch(
                ep_actions[:, action_idx],
                fs=fs_hz,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            if freq_hz is None:
                freq_hz = np.asarray(freq_local, dtype=np.float64)
            psd_values.append(np.asarray(psd_local, dtype=np.float64))

    if freq_hz is None or not psd_values:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return freq_hz, np.mean(np.stack(psd_values, axis=0), axis=0)


def _aggregate_action_psd(run_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    freq_ref: np.ndarray | None = None
    psd_list: List[np.ndarray] = []
    for run_name in run_names:
        freq_hz, psd = _load_or_compute_action_psd(run_name)
        if freq_hz.size == 0 or psd.size == 0:
            continue
        if freq_ref is None:
            freq_ref = freq_hz
        elif not np.array_equal(freq_ref, freq_hz):
            raise RuntimeError(f"PSD frequency bins do not match across runs for {run_names}")
        psd_list.append(psd)
    if freq_ref is None or not psd_list:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    stack = np.stack(psd_list, axis=0)
    return freq_ref, np.mean(stack, axis=0)


def _trajectory_hvac_power_kw(actions: np.ndarray, lengths: np.ndarray, ac_map: np.ndarray, max_power: float) -> List[np.ndarray]:
    series_list: List[np.ndarray] = []
    scale = float(max_power) / 1000.0
    for ep_idx, length in enumerate(lengths.tolist()):
        length = int(length)
        if length <= 1:
            continue
        ep_actions = np.asarray(actions[ep_idx, :length], dtype=np.float64)
        ep_power = np.sum(np.abs(ep_actions) * ac_map[None, :], axis=1) * scale
        series_list.append(ep_power.astype(np.float64))
    return series_list


def _rollout_physical_mpc(metadata: Dict[str, object], episodes: int, seed: int) -> List[np.ndarray]:
    env, _, _ = make_building_env_bcfix_clean(
        building_type=str(metadata["building_type"]),
        weather_type=str(metadata["weather_type"]),
        location=str(metadata["location"]),
        training_num=1,
        test_num=1,
        vector_env_type="dummy",
        target_temp=float(metadata["target_temp"]),
        temp_tolerance=float(metadata["temp_tolerance"]),
        max_power=int(metadata["max_power"]),
        time_resolution=int(metadata["time_resolution"]),
        energy_weight=float(metadata["energy_weight"]),
        temp_weight=float(metadata["temp_weight"]),
        episode_length=int(metadata["episode_length"]),
        add_violation_penalty=bool(metadata["add_violation_penalty"]),
        violation_penalty=float(metadata["violation_penalty"]),
        reward_scale=float(metadata["reward_scale"]),
        expert_type="mpc",
        expert_kwargs={"planning_steps": 3},
    )
    try:
        outputs: List[np.ndarray] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            power_series: List[float] = []
            while not (done or truncated):
                action = np.asarray(env.expert_controller.get_action(obs), dtype=np.float32)
                obs, _, done, truncated, info = env.step(action)
                power_series.append(float(info["hvac_power_kw"]))
            outputs.append(np.asarray(power_series, dtype=np.float64))
        return outputs
    finally:
        if hasattr(env, "close"):
            env.close()


def _average_welch(series_list: Sequence[np.ndarray], fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch  # type: ignore

    valid = [series for series in series_list if series.size >= 8]
    if not valid:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    base_nperseg = min(64, min(series.size for series in valid))
    all_psd: List[np.ndarray] = []
    freq_hz: np.ndarray | None = None

    for series in valid:
        nperseg = min(base_nperseg, series.size)
        noverlap = min(max(0, nperseg // 2), max(0, series.size - 1))
        freq_local, psd_local = welch(
            series,
            fs=fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        if freq_hz is None:
            freq_hz = np.asarray(freq_local, dtype=np.float64)
        all_psd.append(np.asarray(psd_local, dtype=np.float64))

    if freq_hz is None or not all_psd:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return freq_hz, np.mean(np.stack(all_psd, axis=0), axis=0)


def _low_freq_ratio(freq_cpd: np.ndarray, psd: np.ndarray, cutoff_cpd: float) -> float:
    if freq_cpd.size == 0 or psd.size == 0 or float(np.sum(psd)) <= 0.0:
        return float("nan")
    return float(np.sum(psd[freq_cpd <= cutoff_cpd]) / np.sum(psd))


def _pareto_mask(records: Sequence[Dict[str, object]]) -> np.ndarray:
    pts = np.asarray([[float(r["energy"]), float(r["violations"])] for r in records], dtype=np.float64)
    keep = np.ones((pts.shape[0],), dtype=bool)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[0]):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                keep[i] = False
                break
    return keep


def _short_label(label: str) -> str:
    mapping = {
        "DiffFNO": "DiffFNO",
        "DiffFNO w/o Residual": "NoRes",
        "Diffusion-MLP": "MLP",
        "MPC": "MPC",
        "SAC+MPC": "SAC+MPC",
        "SAC": "SAC",
    }
    return mapping.get(label, label)


def _build_pareto_records(run_groups: Dict[str, List[str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in PARETO_METHODS:
        run_names = list(run_groups.get(spec.matcher, []))
        if not run_names:
            continue
        energies: List[float] = []
        violations_list: List[float] = []
        for run_name in run_names:
            run_dir = _run_path(run_name)
            event_path = _find_event_file(run_dir)
            energies.append(_mean_last_k(_load_scalar_series(event_path, "test/avg_energy")))
            violations_list.append(_mean_last_k(_load_scalar_series(event_path, "test/avg_violations")))
        energy_arr = np.asarray(energies, dtype=np.float64)
        violations_arr = np.asarray(violations_list, dtype=np.float64)
        records.append(
            {
                "label": spec.label,
                "matcher": spec.matcher,
                "run_names": run_names,
                "energy": float(np.nanmean(energy_arr)),
                "energy_std": float(np.nanstd(energy_arr, ddof=0)),
                "violations": float(np.nanmean(violations_arr)),
                "violations_std": float(np.nanstd(violations_arr, ddof=0)),
                "color": spec.color,
                "kind": spec.kind,
                "n_runs": len(run_names),
            }
        )
    return records


def render_pareto(records: Sequence[Dict[str, object]], out_dir: str) -> List[str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    x_segments = [
        (844.0, 1068.0),
        (1776.0, 1948.0),
        (5948.0, 6010.0),
    ]
    y_bottom = (0.75, 1.95)
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

    offsets = {
        "DiffFNO": (26.0, 0.03),
        "DiffFNO w/o Residual": (12.0, 0.11),
        "Diffusion-MLP": (12.0, 0.08),
        "MPC": (8.0, 0.12),
        "SAC+MPC": (16.0, 0.20),
        "SAC": (-14.0, 0.16),
    }
    arrow_labels = {"DiffFNO", "SAC+MPC"}

    for row, is_frontier in zip(records, frontier_mask.tolist()):
        energy = float(row["energy"])
        violations = float(row["violations"])
        energy_std = float(row.get("energy_std", 0.0))
        violations_std = float(row.get("violations_std", 0.0))
        label = str(row["label"])
        color = str(row["color"])
        kind = str(row["kind"])
        marker = marker_by_kind.get(kind, "o")
        size = 134 if label == "DiffFNO" else 108
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

    local = [row for row in frontier_records if x_segments[0][0] <= float(row["energy"]) <= x_segments[0][1]]
    if len(local) >= 2:
        xs = [float(row["energy"]) for row in local]
        ys = [float(row["violations"]) for row in local]
        ax_bl.plot(xs, ys, color="#111111", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)

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
    ax_bl.set_yticks([0.8, 1.1, 1.4, 1.7])
    ax_bl.set_xticks([850, 900, 1000])
    ax_bm.set_xticks([1800, 1900])
    ax_br.set_xticks([5950, 6000])

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
    ax_bl.plot([1], [0], transform=ax_bl.transAxes, **marker_kwargs)
    ax_bm.plot([0, 1], [0, 0], transform=ax_bm.transAxes, **marker_kwargs)
    ax_br.plot([0], [0], transform=ax_br.transAxes, **marker_kwargs)
    ax_tl.plot([0], [0], transform=ax_tl.transAxes, **marker_kwargs)
    ax_bl.plot([0], [1], transform=ax_bl.transAxes, **marker_kwargs)

    fig.supxlabel("Energy consumption (kWh)")
    fig.supylabel("Comfort violations")
    fig.patch.set_facecolor("white")
    return _save_figure(fig, out_dir, "compare_energy_violations_noguide_main")


def render_heatmap(run_groups: Dict[str, List[str]], out_dir: str) -> Tuple[List[str], List[Dict[str, object]]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _setup_matplotlib()
    labels = [spec.label for spec in HEATMAP_METHODS]

    energies = []
    comfort = []
    smoothness = []
    convergence = []
    rows: List[Dict[str, object]] = []

    for spec in HEATMAP_METHODS:
        run_names = list(run_groups.get(spec.matcher, []))
        if not run_names:
            raise FileNotFoundError(f"Missing run group for {spec.matcher}")
        energy_vals = []
        comfort_vals = []
        smoothness_vals = []
        convergence_vals = []
        for run_name in run_names:
            event_path = _find_event_file(_run_path(run_name))
            energy_vals.append(_mean_last_k(_load_scalar_series(event_path, "test/avg_energy")))
            comfort_vals.append(_comfort_rate(run_name))
            smoothness_vals.append(_action_mse(run_name))
            convergence_vals.append(_convergence_auc(run_name))
        energy_mean = float(np.nanmean(np.asarray(energy_vals, dtype=np.float64)))
        comfort_mean = float(np.nanmean(np.asarray(comfort_vals, dtype=np.float64)))
        smoothness_mean = float(np.nanmean(np.asarray(smoothness_vals, dtype=np.float64)))
        convergence_mean = float(np.nanmean(np.asarray(convergence_vals, dtype=np.float64)))
        energies.append(energy_mean)
        comfort.append(comfort_mean)
        smoothness.append(smoothness_mean)
        convergence.append(convergence_mean)
        rows.append(
            {
                "label": spec.label,
                "matcher": spec.matcher,
                "run_names": run_names,
                "energy": energy_mean,
                "comfort": comfort_mean,
                "smoothness_mse": smoothness_mean,
                "convergence_auc": convergence_mean,
            }
        )

    matrix = np.vstack(
        [
            _minmax(energies, higher_is_better=False),
            _minmax(comfort, higher_is_better=True),
            _minmax(smoothness, higher_is_better=False),
            _minmax(convergence, higher_is_better=True),
        ]
    ).T

    cmap = LinearSegmentedColormap.from_list(
        "academic_teal_blue",
        ["#f8fbff", "#d8eaf4", "#94c7cf", "#3d8ca3", "#173b67"],
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(4), ["Energy", "Comfort", "Smoothness", "Convergence"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=6)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            r, g, b, _ = im.cmap(val)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if luminance < 0.52 else "#0f172a",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Normalized score")
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, out_dir, "ablation_summary_heatmap_noguide_main")
    plt.close(fig)
    return paths, rows


def render_physical_psd(run_groups: Dict[str, List[str]], representative_map: Dict[str, str], out_dir: str) -> Tuple[List[str], Dict[str, object]]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    metadata = _load_metadata(representative_map["fno_guided_noguide_align"])
    env_for_meta, _, _ = make_building_env_bcfix_clean(
        building_type=str(metadata["building_type"]),
        weather_type=str(metadata["weather_type"]),
        location=str(metadata["location"]),
        training_num=1,
        test_num=1,
        vector_env_type="dummy",
        target_temp=float(metadata["target_temp"]),
        temp_tolerance=float(metadata["temp_tolerance"]),
        max_power=int(metadata["max_power"]),
        time_resolution=int(metadata["time_resolution"]),
        energy_weight=float(metadata["energy_weight"]),
        temp_weight=float(metadata["temp_weight"]),
        episode_length=int(metadata["episode_length"]),
        add_violation_penalty=bool(metadata["add_violation_penalty"]),
        violation_penalty=float(metadata["violation_penalty"]),
        reward_scale=float(metadata["reward_scale"]),
    )
    try:
        ac_map = np.asarray(env_for_meta.ac_map, dtype=np.float64)
    finally:
        if hasattr(env_for_meta, "close"):
            env_for_meta.close()

    diff_power: List[np.ndarray] = []
    mlp_power: List[np.ndarray] = []
    mpc_power: List[np.ndarray] = []

    diff_run_names = list(run_groups["fno_guided_noguide_align"])
    mlp_run_names = list(run_groups["MLP"])

    for run_name in diff_run_names:
        traj = _load_npz(run_name)
        diff_power.extend(_trajectory_hvac_power_kw(traj["actions"], traj["lengths"], ac_map, float(metadata["max_power"])))
        rollout_metadata = _load_metadata(run_name)
        mpc_episodes = int(np.sum(np.asarray(traj["lengths"]) > 1))
        mpc_power.extend(_rollout_physical_mpc(rollout_metadata, episodes=mpc_episodes, seed=int(rollout_metadata["seed"])))

    for run_name in mlp_run_names:
        traj = _load_npz(run_name)
        mlp_power.extend(_trajectory_hvac_power_kw(traj["actions"], traj["lengths"], ac_map, float(metadata["max_power"])))

    fs_hz = 1.0 / float(metadata["time_resolution"])
    freq_hz, psd_mpc = _average_welch(mpc_power, fs_hz)
    freq_hz_diff, psd_diff = _average_welch(diff_power, fs_hz)
    freq_hz_mlp, psd_mlp = _average_welch(mlp_power, fs_hz)
    if not (np.array_equal(freq_hz, freq_hz_diff) and np.array_equal(freq_hz, freq_hz_mlp)):
        raise RuntimeError("Frequency bins do not match across PSD computations.")

    freq_cpd = freq_hz * 86400.0
    low_freq_ratio = {
        "physical_mpc": _low_freq_ratio(freq_cpd, psd_mpc, LOW_FREQ_CPD),
        "difffno": _low_freq_ratio(freq_cpd, psd_diff, LOW_FREQ_CPD),
        "diffusion_mlp": _low_freq_ratio(freq_cpd, psd_mlp, LOW_FREQ_CPD),
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.plot(
        freq_cpd,
        psd_mpc,
        color="#111111",
        linewidth=2.4,
        label=f"Physical MPC | low<= {LOW_FREQ_CPD:g} cpd: {100.0 * low_freq_ratio['physical_mpc']:.1f}%",
    )
    ax.plot(
        freq_cpd,
        psd_diff,
        color="#1f4e79",
        linewidth=2.2,
        label=f"DiffFNO | low<= {LOW_FREQ_CPD:g} cpd: {100.0 * low_freq_ratio['difffno']:.1f}%",
    )
    ax.plot(
        freq_cpd,
        psd_mlp,
        color="#17becf",
        linewidth=2.2,
        label=f"Diffusion-MLP | low<= {LOW_FREQ_CPD:g} cpd: {100.0 * low_freq_ratio['diffusion_mlp']:.1f}%",
    )
    ax.axvline(LOW_FREQ_CPD, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.text(
        LOW_FREQ_CPD + 0.04,
        0.96,
        "2 cpd",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#4b5563",
    )
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel(r"PSD of HVAC Power (kW$^2$/Hz)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.55, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, out_dir, "smalloffice_physical_psd_compare_noguide_main")
    plt.close(fig)

    summary = {
        "building_type": metadata["building_type"],
        "weather_type": metadata["weather_type"],
        "location": metadata["location"],
        "episode_length": int(metadata["episode_length"]),
        "time_resolution_s": int(metadata["time_resolution"]),
        "low_freq_cpd": LOW_FREQ_CPD,
        "low_freq_ratio": low_freq_ratio,
        "main_run_dir": representative_map["fno_guided_noguide_align"],
        "mlp_run_dir": representative_map["MLP"],
        "main_run_dirs": diff_run_names,
        "mlp_run_dirs": mlp_run_names,
        "aggregate_seeds": True,
        "notes": "Conference-oriented PSD comparison with no-guidance DiffFNO as the main method.",
    }
    return paths, summary


def _write_mapping(records: Sequence[Dict[str, object]], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "compare_energy_violations_mapping_noguide_main.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "Paper Name",
                "Source Log Directories",
                "Num Runs",
                "Energy Mean kWh",
                "Energy Std",
                "Comfort Violations Mean",
                "Comfort Violations Std",
            ]
        )
        for row in records:
            writer.writerow(
                [
                    row["label"],
                    " | ".join(row["run_names"]),
                    int(row["n_runs"]),
                    f"{float(row['energy']):.6f}",
                    f"{float(row['energy_std']):.6f}",
                    f"{float(row['violations']):.6f}",
                    f"{float(row['violations_std']):.6f}",
                ]
            )
    return out_path


def _write_heatmap_csv(rows: Sequence[Dict[str, object]], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "ablation_summary_noguide_main.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Label", "Run Directories", "Energy", "Comfort", "SmoothnessMSE", "ConvergenceAUC"])
        for row in rows:
            writer.writerow(
                [
                    row["label"],
                    " | ".join(row["run_names"]),
                    f"{float(row['energy']):.6f}",
                    f"{float(row['comfort']):.6f}",
                    f"{float(row['smoothness_mse']):.6f}",
                    f"{float(row['convergence_auc']):.6f}",
                ]
            )
    return out_path


def _write_psd_summary(summary: Dict[str, object], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "smalloffice_physical_psd_summary_noguide_main.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out_path


def _write_readme(out_dir: str) -> str:
    text = """# OfficeSmall bcfix-clean multiseed figures (no-guidance main)

This directory contains the conference-oriented multiseed figures where the
no-guidance residual DiffFNO model is treated as the main method.

Generated figures:
- compare_energy_violations_noguide_main.pdf/png
- ablation_summary_heatmap_noguide_main.pdf/png
- smalloffice_physical_psd_compare_noguide_main.pdf/png

Method selection logic:
- Pareto figure: DiffFNO, DiffFNO w/o Residual, Diffusion-MLP, MPC, SAC, SAC+MPC
- Heatmap: DiffFNO, DiffFNO w/o Residual, Diffusion-MLP
- Physical PSD: Physical MPC, DiffFNO, Diffusion-MLP

Rationale:
- Guidance-based variants are intentionally excluded from the conference
  figure set so the narrative stays centered on the unguided DiffFNO method.
- The heatmap isolates the residual-branch trade-off inside the unguided
  family, which aligns with the conference claim.
"""
    out_path = os.path.join(out_dir, "README_noguide_main.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OfficeSmall multiseed figures with the no-guidance model as the main method.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    required = {spec.matcher for spec in PARETO_METHODS} | {spec.matcher for spec in HEATMAP_METHODS} | {spec.matcher for spec in PSD_METHODS}
    run_groups = _resolve_run_groups(required)
    representative_map = pick_run_dirs_by_seed_preference(LOG_ROOT, run_groups)

    pareto_records = _build_pareto_records(run_groups)
    outputs: List[str] = []
    outputs.extend(render_pareto(pareto_records, out_dir))
    outputs.append(_write_mapping(pareto_records, out_dir))

    heatmap_paths, heatmap_rows = render_heatmap(run_groups, out_dir)
    outputs.extend(heatmap_paths)
    outputs.append(_write_heatmap_csv(heatmap_rows, out_dir))

    psd_paths, psd_summary = render_physical_psd(run_groups, representative_map, out_dir)
    outputs.extend(psd_paths)
    outputs.append(_write_psd_summary(psd_summary, out_dir))
    outputs.append(_write_readme(out_dir))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
