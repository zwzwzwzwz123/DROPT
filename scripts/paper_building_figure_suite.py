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
    resolve_bcfixclean_officemedium_run_dirs,
    resolve_bcfixclean_smalloffice_run_dirs,
)


LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure")
PROFILE = "legacy"
REWARD_SMOOTH = 7
SUMMARY_K = 5
ROOM_INDEX = 3
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
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
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


def _method_mapping_rows(run_map: Dict[str, str]) -> List[Tuple[str, str]]:
    return [(spec.label, run_map[spec.matcher]) for spec in ALL_METHODS if spec.matcher in run_map]


def export_mapping_csv(run_map: Dict[str, str]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "paper_method_mapping.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Paper Name", "Source Log Directory"])
        writer.writerows(_method_mapping_rows(run_map))
    return out_path


def plot_reward_curves(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    x_max = 0.0
    for spec in REWARD_ONLY_METHODS:
        if spec.matcher not in run_map:
            continue
        run_name = run_map[spec.matcher]
        event_path = _find_event_file(_run_path(run_name))
        series = _load_scalar_series(event_path, "test/reward")
        if not series:
            continue
        steps = np.asarray([step for step, _ in series], dtype=np.float64) / 1e6
        values = np.asarray([value for _, value in series], dtype=np.float64)
        values = _smooth(values, REWARD_SMOOTH)
        ax.plot(steps, values, color=spec.color, linewidth=2.0, label=spec.label)
        x_max = max(x_max, float(np.max(steps)))

    ax.set_xlabel("Training steps ($\\times 10^6$)")
    ax.set_ylabel("Test reward")
    ax.set_title("Reward curves")
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
        if spec.matcher not in run_map:
            continue
        run_name = run_map[spec.matcher]
        traj = _load_npz(run_name, "trajectories.npz")
        dist = _action_delta_distribution(traj["actions"], traj["lengths"])
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
    ax.set_title("Action smoothness")
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
        temp = _windowed_room_series(run_name, "states", ROOM_INDEX, REP_EPISODE)
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
    ax.set_ylabel(f"Temperature, room {ROOM_INDEX} (°C)")
    ax.set_title("Representative temperature trajectory")
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
        action = _windowed_room_series(run_name, "actions", ROOM_INDEX, REP_EPISODE)
        ax.plot(x, action, linewidth=2.0, color=spec.color, label=spec.label)

    if not ax.lines:
        plt.close(fig)
        return []

    ax.set_xlabel("Hour")
    ax.set_ylabel(f"Action, room {ROOM_INDEX}")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Representative control sequence")
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
        temp = _windowed_room_series(run_name, "states", ROOM_INDEX, REP_EPISODE)
        ax.axhspan(lower, upper, color="#d9d9d9", alpha=0.22, zorder=0)
        ax.plot(x, temp, linewidth=2.0, color=spec.color)
        ax.axhline(lower, color="#8a8a8a", linestyle="--", linewidth=0.9)
        ax.axhline(upper, color="#8a8a8a", linestyle="--", linewidth=0.9)
        ax.set_title(spec.label, fontsize=10)
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
            ax.set_ylabel(f"Temp., room {ROOM_INDEX} (°C)")

    fig.suptitle("Representative temperature trajectories across methods", fontsize=13)
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
        action = _windowed_room_series(run_name, "actions", ROOM_INDEX, REP_EPISODE)
        ax.plot(x, action, linewidth=2.0, color=spec.color)
        ax.set_title(spec.label, fontsize=10)
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
            ax.set_ylabel(f"Action, room {ROOM_INDEX}")

    fig.suptitle("Representative control sequences across methods", fontsize=13)
    paths = _save_figure(fig, "control_sequence_all_models")
    plt.close(fig)
    return paths


def plot_action_psd(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    for spec in LEARNED_METHODS:
        if spec.matcher not in run_map:
            continue
        run_name = run_map[spec.matcher]
        freq_hz, psd = _load_or_compute_action_psd(run_name)
        freq_cpd = freq_hz * 86400.0
        if freq_cpd.size == 0 or psd.size == 0:
            continue
        ax.plot(freq_cpd, psd, linewidth=2.0, color=spec.color, label=spec.label)

    if not ax.lines:
        plt.close(fig)
        return []

    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Welch PSD")
    ax.set_yscale("log")
    ax.set_title("Action frequency spectrum")
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


def plot_ablation_summary(run_map: Dict[str, str]) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    if any(spec.matcher not in run_map for spec in ABLATION_METHODS):
        return []
    labels = [spec.label for spec in ABLATION_METHODS]
    run_names = [run_map[spec.matcher] for spec in ABLATION_METHODS]

    energies = []
    comfort = []
    smoothness = []
    convergence = []
    for run_name in run_names:
        event_path = _find_event_file(_run_path(run_name))
        energy = _mean_last_k(_load_scalar_series(event_path, "test/avg_energy"))
        energies.append(energy)
        comfort.append(_comfort_rate(run_name))
        smoothness.append(_action_mse(run_name))
        convergence.append(_convergence_auc(run_name))

    matrix = np.vstack(
        [
            _minmax(energies, higher_is_better=False),
            _minmax(comfort, higher_is_better=True),
            _minmax(smoothness, higher_is_better=False),
            _minmax(convergence, higher_is_better=True),
        ]
    ).T

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(4), ["Energy", "Comfort", "Smoothness", "Convergence"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Ablation summary (normalized, higher is better)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.55 else "#1f2937",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Normalized score")
    fig.patch.set_facecolor("white")
    paths = _save_figure(fig, "ablation_summary_heatmap")
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
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, PROFILE
    args = _parse_args()
    PROFILE = args.profile
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, PROFILE)
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
