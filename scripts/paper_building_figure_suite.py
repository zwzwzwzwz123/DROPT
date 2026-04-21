#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a publication-ready figure suite for the key building-control figures.

Figures:
1. Reward curves
2. Action smoothness comparison
3. Representative temperature trajectories
4. Representative control sequences
5. Welch PSD comparison
6. Ablation summary heatmap
"""

from __future__ import annotations

import argparse
import csv
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
REWARD_SMOOTH = 7
REWARD_LINEWIDTH = 1.15
ACTION_PSD_LINEWIDTH = 1.3
SUMMARY_K = 5
ABLATION_CMAP_NAME = "academic_teal_blue"
ABLATION_OUT_BASENAME = "ablation_summary_heatmap"
BASE_FONT_SIZE = 13.5
AXIS_FONT_SIZE = 15.5
TITLE_FONT_SIZE = 15.5
TICK_FONT_SIZE = 13.2
LEGEND_FONT_SIZE = 12.8
PANEL_TITLE_FONT_SIZE = 13.8
SUPTITLE_FONT_SIZE = 17.2
ANNOTATION_FONT_SIZE = 12.6
ROOM_INDEX = 3
TEMP_ROOM_INDEX = ROOM_INDEX
CONTROL_ROOM_INDEX = 4
DEFAULT_TEMP_ROOM_INDEX = TEMP_ROOM_INDEX
DEFAULT_CONTROL_ROOM_INDEX = CONTROL_ROOM_INDEX
WINDOW_START = 48
WINDOW_END = 120
REP_EPISODE = 0


@dataclass(frozen=True)
class RunSpec:
    label: str
    matcher: str
    color: str


ALL_METHODS: Sequence[RunSpec] = (
    RunSpec("Guided-DiffFNO", "fno_guided_full", "#1f77b4"),
    RunSpec("DiffFNO w/o Guidance", "fno_guided_noguide_align", "#ff7f0e"),
    RunSpec("DiffFNO w/o Residual & Guidance", "fno_guided_nores_noguide_align", "#9467bd"),
    RunSpec("DiffFNO w/o Residual", "fno_guided_nores", "#2ca02c"),
    RunSpec("Diffusion Policy (MLP backbone)", "MLP", "#17becf"),
    RunSpec("MPC", "default_mpc_latest", "#7f7f7f"),
    RunSpec("SAC", "sac_baseline", "#d62728"),
)

LEARNED_METHODS: Sequence[RunSpec] = tuple(spec for spec in ALL_METHODS if spec.label != "MPC")

REWARD_ONLY_METHODS: Sequence[RunSpec] = LEARNED_METHODS + (
    RunSpec("SAC+MPC", "sac_baseline_mpc", "#8c564b"),
)

TRAJECTORY_METHODS: Sequence[RunSpec] = REWARD_ONLY_METHODS

ABLATION_METHODS: Sequence[RunSpec] = (
    RunSpec("Guided-DiffFNO", "fno_guided_full", "#1f77b4"),
    RunSpec("DiffFNO w/o Guidance", "fno_guided_noguide_align", "#ff7f0e"),
    RunSpec("DiffFNO w/o Residual & Guidance", "fno_guided_nores_noguide_align", "#9467bd"),
    RunSpec("DiffFNO w/o Residual", "fno_guided_nores", "#2ca02c"),
    RunSpec("Diffusion Policy (MLP backbone)", "MLP", "#17becf"),
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


def _save_figure(fig, out_basename: str) -> List[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_paths: List[str] = []
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUT_DIR, f"{out_basename}.{ext}")
        fig.savefig(out_path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out_paths.append(out_path)
    return out_paths


def _find_event_file(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        if name.startswith("events.out.tfevents"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No event file found in {run_dir}")


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


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _resolve_legacy_run_dirs(log_root: str) -> Dict[str, str]:
    all_dirs = [name for name in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, name))]
    resolved: Dict[str, str] = {}

    default_mpc_runs = sorted(name for name in all_dirs if name.startswith("default_mpc"))
    if default_mpc_runs:
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

    required = {spec.matcher for spec in ALL_METHODS} | {spec.matcher for spec in REWARD_ONLY_METHODS}
    missing = sorted(matcher for matcher in required if matcher not in resolved)
    if missing:
        raise FileNotFoundError(f"Missing runs for: {missing}")
    return resolved


def _resolve_run_dirs(log_root: str) -> Dict[str, str]:
    required = {spec.matcher for spec in ALL_METHODS} | {spec.matcher for spec in REWARD_ONLY_METHODS}
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dirs(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dirs(log_root, required, allow_partial=True)
    return _resolve_legacy_run_dirs(log_root)


def _resolve_run_groups(log_root: str) -> Dict[str, List[str]]:
    required = {spec.matcher for spec in ALL_METHODS} | {spec.matcher for spec in REWARD_ONLY_METHODS}
    if PROFILE == "bcfixclean_smalloffice":
        return resolve_bcfixclean_smalloffice_run_dir_groups(log_root, required)
    if PROFILE == "bcfixclean_officemedium_partial":
        return resolve_bcfixclean_officemedium_run_dir_groups(log_root, required, allow_partial=True)
    return {matcher: [name] for matcher, name in _resolve_legacy_run_dirs(log_root).items()}


def _run_dir_map() -> Dict[str, str]:
    return _resolve_run_dirs(LOG_ROOT)


def _run_path(run_name: str) -> str:
    return os.path.join(LOG_ROOT, run_name)


def _load_npz(run_name: str, filename: str) -> np.lib.npyio.NpzFile:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing paper data: {path}")
    return np.load(path)


def _load_or_compute_action_psd(run_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "actions_welch_psd.npz")
    if os.path.exists(path):
        data = np.load(path)
        return (
            np.asarray(data["frequency_hz"], dtype=np.float64),
            np.asarray(data["psd_mean"], dtype=np.float64),
        )

    traj = _load_npz(run_name, "trajectories.npz")
    actions = np.asarray(traj["actions"], dtype=np.float64)
    lengths = np.asarray(traj["lengths"], dtype=np.int32)
    valid_lengths = [int(length) for length in lengths.tolist() if int(length) >= 8]
    if not valid_lengths:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    try:
        from scipy.signal import welch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to compute PSD from trajectories.npz") from exc

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
            freq_hz_local, psd_local = welch(
                ep_actions[:, action_idx],
                fs=fs_hz,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
            )
            if freq_hz is None:
                freq_hz = np.asarray(freq_hz_local, dtype=np.float64)
            psd_values.append(np.asarray(psd_local, dtype=np.float64))

    if freq_hz is None or not psd_values:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return freq_hz, np.mean(np.stack(psd_values, axis=0), axis=0)


def _run_names_for(matcher: str, run_map: Dict[str, str]) -> List[str]:
    if AGGREGATE_SEEDS and matcher in RUN_GROUPS:
        return list(RUN_GROUPS[matcher])
    if matcher in run_map:
        return [run_map[matcher]]
    return []


def _aggregate_scalar_curves(run_names: Sequence[str], tag: str, smooth_window: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    series_list: List[List[Tuple[int, float]]] = []
    for run_name in run_names:
        event_path = _find_event_file(_run_path(run_name))
        series = _load_scalar_series(event_path, tag)
        if not series:
            continue
        steps = np.asarray([step for step, _ in series], dtype=np.float64)
        values = np.asarray([value for _, value in series], dtype=np.float64)
        values = _smooth(values, smooth_window)
        series_list.append(list(zip(steps.tolist(), values.tolist())))
    if not series_list:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    all_steps = sorted({int(step) for series in series_list for step, _ in series})
    matrix = np.full((len(series_list), len(all_steps)), np.nan, dtype=np.float64)
    step_to_idx = {step: idx for idx, step in enumerate(all_steps)}
    for row_idx, series in enumerate(series_list):
        for step, value in series:
            matrix[row_idx, step_to_idx[int(step)]] = float(value)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=0)
    return np.asarray(all_steps, dtype=np.float64), mean, std


def _aggregate_action_psd(run_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    stack = np.stack(psd_list, axis=0)
    return freq_ref, np.mean(stack, axis=0), np.std(stack, axis=0, ddof=0)


def _method_mapping_rows(run_map: Dict[str, str]) -> List[Tuple[str, str]]:
    return [(spec.label, run_map[spec.matcher]) for spec in ALL_METHODS if spec.matcher in run_map]


def export_mapping_csv(run_map: Dict[str, str]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "paper_method_mapping.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        if AGGREGATE_SEEDS:
            writer.writerow(["Paper Name", "Representative Log Directory", "Aggregated Log Directories"])
            for spec in ALL_METHODS:
                if spec.matcher not in run_map and spec.matcher not in RUN_GROUPS:
                    continue
                writer.writerow(
                    [
                        spec.label,
                        run_map.get(spec.matcher, ""),
                        " | ".join(RUN_GROUPS.get(spec.matcher, [])),
                    ]
                )
        else:
            writer.writerow(["Paper Name", "Source Log Directory"])
            writer.writerows(_method_mapping_rows(run_map))
    return out_path


def plot_reward_curves(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    x_max = 0.0
    for spec in REWARD_ONLY_METHODS:
        run_names = _run_names_for(spec.matcher, run_map)
        if not run_names:
            continue
        steps_raw, mean_values, std_values = _aggregate_scalar_curves(run_names, "test/reward", smooth_window=REWARD_SMOOTH)
        if steps_raw.size == 0:
            continue
        steps = steps_raw / 1e6
        ax.plot(steps, mean_values, color=spec.color, linewidth=REWARD_LINEWIDTH, label=spec.label)
        if AGGREGATE_SEEDS and len(run_names) > 1:
            ax.fill_between(steps, mean_values - std_values, mean_values + std_values, color=spec.color, alpha=0.14, linewidth=0.0)
        x_max = max(x_max, float(np.max(steps)))

    ax.set_xlabel("Training steps (×10⁶)")
    ax.set_ylabel("Test reward")
    if x_max > 0:
        ax.set_xlim(0.0, x_max * 1.02)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "compare_reward_curves")
    plt.close(fig)
    return paths


def _action_delta_distribution(actions: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values: List[np.ndarray] = []
    for ep_idx, length in enumerate(lengths.tolist()):
        if int(length) <= 1:
            continue
        ep_actions = actions[ep_idx, : int(length)]
        deltas = np.abs(np.diff(ep_actions, axis=0)).mean(axis=1)
        values.append(deltas.astype(np.float64))
    if not values:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(values, axis=0)


def plot_action_smoothness(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    labels: List[str] = []
    data: List[np.ndarray] = []
    colors: List[str] = []
    means: List[float] = []

    for spec in LEARNED_METHODS:
        run_names = _run_names_for(spec.matcher, run_map)
        if not run_names:
            continue
        dists: List[np.ndarray] = []
        for run_name in run_names:
            traj = _load_npz(run_name, "trajectories.npz")
            dist = _action_delta_distribution(traj["actions"], traj["lengths"])
            if dist.size > 0:
                dists.append(dist)
        if not dists:
            continue
        dist = np.concatenate(dists, axis=0)
        if dist.size == 0:
            continue
        labels.append(spec.label)
        data.append(dist)
        colors.append(spec.color)
        means.append(float(np.mean(dist)))

    if not data:
        plt.close(fig)
        return []

    positions = np.arange(1, len(labels) + 1)
    box = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    ax.scatter(means, positions, color="black", s=18, zorder=3, label="Mean")
    ax.set_yticks(positions, labels)
    ax.set_xlabel(r"Mean absolute action change, $|\Delta a|$")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "compare_action_smoothness")
    plt.close(fig)
    return paths


def _get_temperature_band(run_name: str) -> Tuple[float, float]:
    traj = _load_npz(run_name, "trajectories.npz")
    del traj
    import pickle

    meta_path = os.path.join(LOG_ROOT, run_name, "paper_data", "paper_metadata.pkl")
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    args = meta.get("args", {})
    target = float(args["target_temp"])
    tolerance = float(args["temp_tolerance"])
    return target - tolerance, target + tolerance


def _windowed_room_series(run_name: str, key: str, room_idx: int, episode_idx: int) -> np.ndarray:
    traj = _load_npz(run_name, "trajectories.npz")
    arr = traj[key]
    if key == "states":
        series = arr[episode_idx, WINDOW_START:WINDOW_END, room_idx]
    elif key == "actions":
        series = arr[episode_idx, WINDOW_START:WINDOW_END, room_idx]
    else:
        raise KeyError(key)
    return np.asarray(series, dtype=np.float64)


def plot_temperature_trajectories(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)

    selected = [
        next(spec for spec in ALL_METHODS if spec.label == "Guided-DiffFNO"),
        next(spec for spec in ALL_METHODS if spec.label == "DiffFNO w/o Guidance"),
    ]
    x = np.arange(WINDOW_END - WINDOW_START)
    linestyles = {
        "Guided-DiffFNO": "-",
        "DiffFNO w/o Guidance": "--",
    }
    for spec in selected:
        if spec.matcher not in run_map:
            continue
        run_name = run_map[spec.matcher]
        temp = _windowed_room_series(run_name, "states", TEMP_ROOM_INDEX, REP_EPISODE)
        ax.plot(
            x,
            temp,
            linewidth=2.0,
            color=spec.color,
            linestyle=linestyles.get(spec.label, "-"),
            label=spec.label,
        )

    if not ax.lines or "fno_guided_full" not in run_map:
        plt.close(fig)
        return []
    lower, upper = _get_temperature_band(run_map["fno_guided_full"])
    ax.axhline(lower, color="#7f7f7f", linestyle="--", linewidth=1.0, label="Comfort band")
    ax.axhline(upper, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hour")
    ax.set_ylabel(f"Temperature, room {TEMP_ROOM_INDEX} (°C)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "temperature_trajectories_paper")
    plt.close(fig)
    return paths


def plot_control_sequences(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)

    selected_labels = [
        "Guided-DiffFNO",
        "Diffusion Policy (MLP backbone)",
        "SAC",
    ]
    x = np.arange(WINDOW_END - WINDOW_START)
    for label in selected_labels:
        spec = next(item for item in ALL_METHODS if item.label == label)
        if spec.matcher not in run_map:
            continue
        run_name = run_map[spec.matcher]
        action = _windowed_room_series(run_name, "actions", CONTROL_ROOM_INDEX, REP_EPISODE)
        ax.plot(x, action, linewidth=2.0, color=spec.color, label=spec.label)

    if not ax.lines:
        plt.close(fig)
        return []

    ax.set_xlabel("Hour")
    ax.set_ylabel(f"Action, room {CONTROL_ROOM_INDEX}")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "control_sequence_paper")
    plt.close(fig)
    return paths


def _panel_axes_count(n: int) -> Tuple[int, int]:
    cols = 2
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_temperature_trajectories_all(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    methods = [spec for spec in TRAJECTORY_METHODS if spec.matcher in run_map]
    if not methods or "fno_guided_full" not in run_map:
        return []
    rows, cols = _panel_axes_count(len(methods))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 2.6 * rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    x = np.arange(WINDOW_END - WINDOW_START)
    lower, upper = _get_temperature_band(run_map["fno_guided_full"])
    y_min = lower - 2.0
    y_max = upper + 5.0

    for idx, spec in enumerate(methods):
        ax = axes[idx // cols, idx % cols]
        run_name = run_map[spec.matcher]
        temp = _windowed_room_series(run_name, "states", TEMP_ROOM_INDEX, REP_EPISODE)
        ax.axhspan(lower, upper, color="#d9d9d9", alpha=0.22, zorder=0)
        ax.plot(x, temp, linewidth=2.0, color=spec.color)
        ax.axhline(lower, color="#8a8a8a", linestyle="--", linewidth=0.9)
        ax.axhline(upper, color="#8a8a8a", linestyle="--", linewidth=0.9)
        ax.set_title(spec.label, fontsize=PANEL_TITLE_FONT_SIZE)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(y_min, y_max)

    for idx in range(len(methods), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Hour")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(f"Temp., room {TEMP_ROOM_INDEX} (°C)")

    paths = _save_figure(fig, "temperature_trajectories_all_models")
    plt.close(fig)
    return paths


def plot_control_sequences_all(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    methods = [spec for spec in TRAJECTORY_METHODS if spec.matcher in run_map]
    if not methods:
        return []
    rows, cols = _panel_axes_count(len(methods))
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 2.6 * rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    x = np.arange(WINDOW_END - WINDOW_START)

    for idx, spec in enumerate(methods):
        ax = axes[idx // cols, idx % cols]
        run_name = run_map[spec.matcher]
        action = _windowed_room_series(run_name, "actions", CONTROL_ROOM_INDEX, REP_EPISODE)
        ax.plot(x, action, linewidth=2.0, color=spec.color)
        ax.set_title(spec.label, fontsize=PANEL_TITLE_FONT_SIZE)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-1.05, 1.05)

    for idx in range(len(methods), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Hour")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel(f"Action, room {CONTROL_ROOM_INDEX}")

    paths = _save_figure(fig, "control_sequence_all_models")
    plt.close(fig)
    return paths


def plot_action_psd(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    for spec in LEARNED_METHODS:
        run_names = _run_names_for(spec.matcher, run_map)
        if not run_names:
            continue
        freq_hz, psd, psd_std = _aggregate_action_psd(run_names)
        freq_cpd = freq_hz * 86400.0
        if freq_cpd.size == 0 or psd.size == 0:
            continue
        ax.plot(freq_cpd, psd, linewidth=ACTION_PSD_LINEWIDTH, color=spec.color, label=spec.label)
        if AGGREGATE_SEEDS and len(run_names) > 1:
            lower = np.clip(psd - psd_std, a_min=np.finfo(np.float64).tiny, a_max=None)
            upper = np.clip(psd + psd_std, a_min=np.finfo(np.float64).tiny, a_max=None)
            ax.fill_between(freq_cpd, lower, upper, color=spec.color, alpha=0.14, linewidth=0.0)

    if not ax.lines:
        plt.close(fig)
        return []

    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Welch PSD")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.55, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "action_psd_compare")
    plt.close(fig)
    return paths


def _mean_last_k(values: Iterable[Tuple[int, float]], k: int = SUMMARY_K) -> float:
    series = [value for _, value in values]
    if not series:
        return float("nan")
    kk = max(1, min(k, len(series)))
    return float(np.mean(series[-kk:]))


def _comfort_rate(run_name: str) -> float:
    traj = _load_npz(run_name, "trajectories.npz")
    violations = traj["comfort_violations"]
    lengths = traj["lengths"]
    action_dim = int(traj["actions"].shape[-1])
    total_violations = 0.0
    total_slots = 0
    for ep_idx, length in enumerate(lengths.tolist()):
        total_violations += float(np.sum(violations[ep_idx, : int(length)]))
        total_slots += int(length) * action_dim
    if total_slots == 0:
        return float("nan")
    return float(1.0 - total_violations / total_slots)


def _action_mse(run_name: str) -> float:
    traj = _load_npz(run_name, "trajectories.npz")
    actions = traj["actions"]
    lengths = traj["lengths"]
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


def _resolve_ablation_cmap(name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    key = str(name).strip().lower()
    if key == "academic_teal_blue":
        return LinearSegmentedColormap.from_list(
            "academic_teal_blue",
            ["#f8fbff", "#d8eaf4", "#94c7cf", "#3d8ca3", "#173b67"],
        )
    if key == "dopamine":
        return LinearSegmentedColormap.from_list(
            "dopamine",
            ["#fff7db", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"],
        )
    if key == "ylgnbu":
        return plt.get_cmap("YlGnBu")
    if key == "viridis":
        return plt.get_cmap("viridis")
    if key == "turbo":
        return plt.get_cmap("turbo")
    return plt.get_cmap(name)


def plot_ablation_summary(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    if any(spec.matcher not in run_map for spec in ABLATION_METHODS):
        return []
    labels = [spec.label for spec in ABLATION_METHODS]

    energies = []
    comfort = []
    smoothness = []
    convergence = []
    for spec in ABLATION_METHODS:
        run_names = _run_names_for(spec.matcher, run_map)
        if not run_names:
            return []
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
        energies.append(float(np.nanmean(np.asarray(energy_vals, dtype=np.float64))))
        comfort.append(float(np.nanmean(np.asarray(comfort_vals, dtype=np.float64))))
        smoothness.append(float(np.nanmean(np.asarray(smoothness_vals, dtype=np.float64))))
        convergence.append(float(np.nanmean(np.asarray(convergence_vals, dtype=np.float64))))

    matrix = np.vstack(
        [
            _minmax(energies, higher_is_better=False),
            _minmax(comfort, higher_is_better=True),
            _minmax(smoothness, higher_is_better=False),
            _minmax(convergence, higher_is_better=True),
        ]
    ).T

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap=_resolve_ablation_cmap(ABLATION_CMAP_NAME), vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(4), ["Energy", "Comfort", "Smoothness", "Convergence"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_box_aspect(1.0)
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

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Normalized score")
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, ABLATION_OUT_BASENAME)
    plt.close(fig)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready building figure suite.")
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
        "--temp-room-index",
        type=int,
        default=DEFAULT_TEMP_ROOM_INDEX,
        help="Room index used by temperature trajectory figures.",
    )
    parser.add_argument(
        "--control-room-index",
        type=int,
        default=DEFAULT_CONTROL_ROOM_INDEX,
        help="Room index used by control-sequence figures.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate matched runs across seeds for statistical figures while keeping representative trajectories on a canonical seed.",
    )
    parser.add_argument(
        "--ablation-cmap",
        type=str,
        default="academic_teal_blue",
        help="Colormap used by the ablation heatmap. Examples: viridis, YlGnBu, dopamine, turbo.",
    )
    parser.add_argument(
        "--ablation-out-basename",
        type=str,
        default="ablation_summary_heatmap",
        help="Output basename for the ablation heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, PROFILE, TEMP_ROOM_INDEX, CONTROL_ROOM_INDEX, AGGREGATE_SEEDS, RUN_GROUPS, ABLATION_CMAP_NAME, ABLATION_OUT_BASENAME
    args = _parse_args()
    PROFILE = args.profile
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, PROFILE)
    TEMP_ROOM_INDEX = args.temp_room_index
    CONTROL_ROOM_INDEX = args.control_room_index
    AGGREGATE_SEEDS = bool(args.aggregate_seeds)
    ABLATION_CMAP_NAME = str(args.ablation_cmap)
    ABLATION_OUT_BASENAME = str(args.ablation_out_basename)
    if AGGREGATE_SEEDS:
        RUN_GROUPS = _resolve_run_groups(LOG_ROOT)
        run_map = pick_run_dirs_by_seed_preference(LOG_ROOT, RUN_GROUPS)
    else:
        RUN_GROUPS = {}
        run_map = _run_dir_map()
    outputs: List[str] = []
    outputs.extend(plot_reward_curves(run_map))
    outputs.extend(plot_action_smoothness(run_map))
    outputs.extend(plot_temperature_trajectories(run_map))
    outputs.extend(plot_temperature_trajectories_all(run_map))
    outputs.extend(plot_control_sequences(run_map))
    outputs.extend(plot_control_sequences_all(run_map))
    outputs.extend(plot_action_psd(run_map))
    outputs.extend(plot_ablation_summary(run_map))
    outputs.append(export_mapping_csv(run_map))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
