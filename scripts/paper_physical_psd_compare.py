#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a publication-ready physical-control PSD comparison figure for OfficeSmall.

The figure compares HVAC power spectra from:
1. Physical MPC controller rolled out in the current bcfix-clean environment
2. Guided-DiffFNO (current paper run)
3. Diffusion-MLP (current paper run)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dropt_utils.paper_building_profiles import (
    default_out_dir,
    resolve_bcfixclean_smalloffice_run_dir_groups,
    resolve_bcfixclean_smalloffice_run_dirs,
)
from main_building_fno_guided_bcfix_clean import make_building_env_bcfix_clean


LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure_bcfixclean_smalloffice")
AGGREGATE_SEEDS = False
LOW_FREQ_CPD = 2.0
BASE_FONT_SIZE = 11
AXIS_FONT_SIZE = 12.5
TITLE_FONT_SIZE = 12.5
TICK_FONT_SIZE = 10.8
LEGEND_FONT_SIZE = 10.2
ANNOTATION_FONT_SIZE = 10.0

GUIDED_COLOR = "#1f77b4"
MLP_COLOR = "#17becf"
MPC_COLOR = "#111111"


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


def _resolve_run_dirs() -> Dict[str, str]:
    required = ["fno_guided_full", "MLP"]
    return resolve_bcfixclean_smalloffice_run_dirs(LOG_ROOT, required)


def _resolve_run_groups() -> Dict[str, List[str]]:
    required = ["fno_guided_full", "MLP"]
    return resolve_bcfixclean_smalloffice_run_dir_groups(LOG_ROOT, required)


def _load_metadata(run_name: str) -> Dict[str, object]:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "paper_metadata.pkl")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    args = obj.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Invalid metadata args in {path}")
    return args


def _load_npz(run_name: str) -> np.lib.npyio.NpzFile:
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "trajectories.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


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


def _write_curve_csv(
    out_path: str,
    freq_hz: np.ndarray,
    freq_cpd: np.ndarray,
    psd_mpc: np.ndarray,
    psd_guided: np.ndarray,
    psd_mlp: np.ndarray,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "freq_hz",
                "freq_cycles_per_day",
                "psd_physical_mpc",
                "psd_guided_difffno",
                "psd_diffusion_mlp",
            ]
        )
        for idx in range(freq_hz.size):
            writer.writerow(
                [
                    float(freq_hz[idx]),
                    float(freq_cpd[idx]),
                    float(psd_mpc[idx]),
                    float(psd_guided[idx]),
                    float(psd_mlp[idx]),
                ]
            )


def _write_summary_json(
    out_path: str,
    metadata: Dict[str, object],
    low_freq_ratio: Dict[str, float],
    run_map: Dict[str, str],
    run_groups: Dict[str, List[str]] | None = None,
) -> None:
    payload = {
        "building_type": metadata["building_type"],
        "weather_type": metadata["weather_type"],
        "location": metadata["location"],
        "episode_length": int(metadata["episode_length"]),
        "time_resolution_s": int(metadata["time_resolution"]),
        "low_freq_cpd": LOW_FREQ_CPD,
        "low_freq_ratio": low_freq_ratio,
        "guided_run_dir": run_map["fno_guided_full"],
        "mlp_run_dir": run_map["MLP"],
        "guided_run_dirs": list(run_groups.get("fno_guided_full", [])) if run_groups else [run_map["fno_guided_full"]],
        "mlp_run_dirs": list(run_groups.get("MLP", [])) if run_groups else [run_map["MLP"]],
        "aggregate_seeds": bool(run_groups),
        "notes": "HVAC power PSD comparison under the current bcfix-clean OfficeSmall setting.",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def plot_physical_psd_compare(
    freq_cpd: np.ndarray,
    psd_mpc: np.ndarray,
    psd_guided: np.ndarray,
    psd_mlp: np.ndarray,
    low_freq_ratio: Dict[str, float],
) -> List[str]:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)

    ax.plot(
        freq_cpd,
        psd_mpc,
        color=MPC_COLOR,
        linewidth=2.4,
        label=f"Physical MPC | low<= {LOW_FREQ_CPD:g} cpd: {100.0 * low_freq_ratio['physical_mpc']:.1f}%",
    )
    ax.plot(
        freq_cpd,
        psd_guided,
        color=GUIDED_COLOR,
        linewidth=2.2,
        label=f"Guided-DiffFNO | low<= {LOW_FREQ_CPD:g} cpd: {100.0 * low_freq_ratio['guided_difffno']:.1f}%",
    )
    ax.plot(
        freq_cpd,
        psd_mlp,
        color=MLP_COLOR,
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
    return _save_figure(fig, "smalloffice_physical_psd_compare")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate physical-control PSD comparison for OfficeSmall.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for the generated figure and summary files.",
    )
    parser.add_argument(
        "--mpc-episodes",
        type=int,
        default=None,
        help="Number of MPC evaluation episodes to roll out. Defaults to the number of logged paper trajectories.",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate Guided-DiffFNO and Diffusion-MLP spectra across all matched seeds.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR, AGGREGATE_SEEDS
    args = _parse_args()
    AGGREGATE_SEEDS = bool(args.aggregate_seeds)
    OUT_DIR = args.out_dir or default_out_dir(ROOT_DIR, "bcfixclean_smalloffice")
    os.makedirs(OUT_DIR, exist_ok=True)

    run_groups = _resolve_run_groups() if AGGREGATE_SEEDS else {}
    if AGGREGATE_SEEDS:
        run_map = {
            matcher: sorted(names)[0]
            for matcher, names in run_groups.items()
            if names
        }
    else:
        run_map = _resolve_run_dirs()
    metadata = _load_metadata(run_map["fno_guided_full"])
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

    guided_power: List[np.ndarray] = []
    mlp_power: List[np.ndarray] = []
    mpc_power: List[np.ndarray] = []

    guided_run_names = run_groups.get("fno_guided_full", [run_map["fno_guided_full"]]) if AGGREGATE_SEEDS else [run_map["fno_guided_full"]]
    mlp_run_names = run_groups.get("MLP", [run_map["MLP"]]) if AGGREGATE_SEEDS else [run_map["MLP"]]

    for run_name in guided_run_names:
        traj = _load_npz(run_name)
        guided_power.extend(
            _trajectory_hvac_power_kw(traj["actions"], traj["lengths"], ac_map, float(metadata["max_power"]))
        )
        rollout_metadata = _load_metadata(run_name)
        mpc_episodes = int(args.mpc_episodes) if args.mpc_episodes is not None else int(np.sum(traj["lengths"] > 1))
        mpc_power.extend(
            _rollout_physical_mpc(rollout_metadata, episodes=mpc_episodes, seed=int(rollout_metadata["seed"]))
        )

    for run_name in mlp_run_names:
        traj = _load_npz(run_name)
        mlp_power.extend(
            _trajectory_hvac_power_kw(traj["actions"], traj["lengths"], ac_map, float(metadata["max_power"]))
        )

    fs_hz = 1.0 / float(metadata["time_resolution"])
    freq_hz, psd_mpc = _average_welch(mpc_power, fs_hz)
    freq_hz_guided, psd_guided = _average_welch(guided_power, fs_hz)
    freq_hz_mlp, psd_mlp = _average_welch(mlp_power, fs_hz)

    if not (np.array_equal(freq_hz, freq_hz_guided) and np.array_equal(freq_hz, freq_hz_mlp)):
        raise RuntimeError("Frequency bins do not match across PSD computations.")

    freq_cpd = freq_hz * 86400.0
    low_freq_ratio = {
        "physical_mpc": _low_freq_ratio(freq_cpd, psd_mpc, LOW_FREQ_CPD),
        "guided_difffno": _low_freq_ratio(freq_cpd, psd_guided, LOW_FREQ_CPD),
        "diffusion_mlp": _low_freq_ratio(freq_cpd, psd_mlp, LOW_FREQ_CPD),
    }

    out_paths = plot_physical_psd_compare(freq_cpd, psd_mpc, psd_guided, psd_mlp, low_freq_ratio)
    curve_csv = os.path.join(OUT_DIR, "smalloffice_physical_psd_curves.csv")
    summary_json = os.path.join(OUT_DIR, "smalloffice_physical_psd_summary.json")
    _write_curve_csv(curve_csv, freq_hz, freq_cpd, psd_mpc, psd_guided, psd_mlp)
    _write_summary_json(summary_json, metadata, low_freq_ratio, run_map, run_groups if AGGREGATE_SEEDS else None)

    for path in out_paths + [curve_csv, summary_json]:
        print(path)


if __name__ == "__main__":
    main()
