#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script A — spatial-axis (zone-axis) action spectrum.

The conference version claims the FNO applies a low-pass prior along the ZONE
axis, but every smoothness figure in the paper (mean|delta a|, Welch PSD) is
measured on the TIME axis. That leaves the causal chain unclosed. This script
supplies the missing evidence: it takes the per-timestep action vector across
zones, does an rfft along the ZONE axis, and averages |FFT|^2 over timesteps and
episodes. If the FNO prior works as claimed, FNO actions carry a larger fraction
of their cross-zone energy in low spatial modes than MLP.

Honesty caveat (verified in source): zones are indexed by geometry-file room
order; connectmap coupling is a general graph, not a 1D chain. So the "spatial
frequency" is a smoothness prior over the INDEX order, not over true physical
space. Reported alongside the figure, not hidden.

Pure offline: reads paper_data/trajectories.npz. No env, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dropt_utils.paper_building_profiles import (
    resolve_bcfixclean_building_run_dir_groups,
    pick_run_dirs_by_seed_preference,
)

LOG_ROOT = os.path.join(ROOT_DIR, "log_building")
OUT_DIR = os.path.join(ROOT_DIR, "paperfigure_spatial_spectrum")
BUILDINGS = ["OfficeSmall", "OfficeMedium"]  # School added after 1M run
MATCHERS = ["fno_guided_full", "MLP"]
LOW_MODE_CUTOFF = 4  # modes the FNO keeps; "low spatial band" = modes < this

def _resolve_run(building_type: str, matcher: str) -> str:
    groups = resolve_bcfixclean_building_run_dir_groups(
        log_root=LOG_ROOT,
        building_type=building_type,
        weather_type="Hot_Dry",
        required_matchers=MATCHERS,
        allow_partial=True,
    )
    chosen = pick_run_dirs_by_seed_preference(LOG_ROOT, groups)
    if matcher not in chosen:
        raise FileNotFoundError(f"No {matcher} run for {building_type}")
    return chosen[matcher]


def _spatial_spectrum(run_name: str) -> tuple[np.ndarray, int]:
    """Mean |rfft(action_vector over zones)|^2, averaged over all valid steps.

    trajectories.npz['actions'] is [episode, timestep, zone]. For each timestep we
    take the zone-axis vector, remove its mean (the DC/mode-0 term = overall power
    level, not a spatial pattern), rfft along zones, and average power per mode.
    """
    path = os.path.join(LOG_ROOT, run_name, "paper_data", "trajectories.npz")
    traj = np.load(path)
    actions = np.asarray(traj["actions"], dtype=np.float64)  # [E, T, Z]
    lengths = np.asarray(traj["lengths"]).astype(int)
    n_zone = actions.shape[2]
    rfft_len = n_zone // 2 + 1
    acc = np.zeros(rfft_len, dtype=np.float64)
    count = 0
    for ep_idx, length in enumerate(lengths.tolist()):
        if length <= 1:
            continue
        vecs = actions[ep_idx, :length]              # [t, Z]
        vecs = vecs - vecs.mean(axis=1, keepdims=True)  # drop DC / overall level
        ft = np.fft.rfft(vecs, axis=1)               # [t, rfft_len]
        acc += np.mean(np.abs(ft) ** 2, axis=0)
        count += 1
    if count == 0:
        return np.zeros(rfft_len), n_zone
    return acc / count, n_zone


def _low_mode_ratio(psd: np.ndarray, cutoff: int) -> float:
    """Fraction of spatial energy (excl. mode 0) in modes 1..cutoff-1."""
    total = float(psd[1:].sum())  # exclude DC (removed anyway, but be safe)
    if total <= 0:
        return float("nan")
    hi_end = min(cutoff, psd.size)
    return float(psd[1:hi_end].sum() / total)


def _plot(building: str, fno_psd: np.ndarray, mlp_psd: np.ndarray, n_zone: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    modes = np.arange(fno_psd.size)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(modes, fno_psd, "o-", color="#1f77b4", linewidth=2.0, label="Guided-DiffFNO")
    ax.plot(modes, mlp_psd, "s--", color="#17becf", linewidth=2.0, label="Diffusion-MLP")
    ax.axvline(LOW_MODE_CUTOFF - 0.5, color="#6b7280", linestyle=":", linewidth=1.1)
    ax.text(LOW_MODE_CUTOFF - 0.45, 0.96, f"FNO keeps modes < {LOW_MODE_CUTOFF}",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=10, color="#4b5563")
    ax.set_xlabel(f"Spatial mode along zone index (n_zone={n_zone})")
    ax.set_ylabel(r"Mean spatial PSD  $\langle|\mathrm{FFT}_\mathrm{zone}(a)|^2\rangle$")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{building}_spatial_spectrum.{ext}")
        fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    OUT_DIR = args.out_dir or OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_rows = []
    for building in BUILDINGS:
        fno_run = _resolve_run(building, "fno_guided_full")
        mlp_run = _resolve_run(building, "MLP")
        fno_psd, n_zone = _spatial_spectrum(fno_run)
        mlp_psd, _ = _spatial_spectrum(mlp_run)
        fno_low = _low_mode_ratio(fno_psd, LOW_MODE_CUTOFF)
        mlp_low = _low_mode_ratio(mlp_psd, LOW_MODE_CUTOFF)
        # Absolute cross-zone energy (sum over non-DC modes): the clearer signal.
        fno_energy = float(fno_psd[1:].sum())
        mlp_energy = float(mlp_psd[1:].sum())
        # Nyquist (highest spatial mode) share: checkerboard/type-alternation content
        # that the FNO residual bypass can reinject past the low-pass spectral conv.
        fno_nyq = float(fno_psd[-1] / fno_psd[1:].sum()) if fno_psd[1:].sum() > 0 else float("nan")
        mlp_nyq = float(mlp_psd[-1] / mlp_psd[1:].sum()) if mlp_psd[1:].sum() > 0 else float("nan")
        _plot(building, fno_psd, mlp_psd, n_zone)
        csv_rows.append({
            "building": building, "n_zone": n_zone, "rfft_len": fno_psd.size,
            "fno_crosszone_energy": round(fno_energy, 5), "mlp_crosszone_energy": round(mlp_energy, 5),
            "fno_over_mlp_energy": round(fno_energy / mlp_energy, 4) if mlp_energy > 0 else None,
            "fno_low_mode_ratio": round(fno_low, 4), "mlp_low_mode_ratio": round(mlp_low, 4),
            "fno_nyquist_share": round(fno_nyq, 4), "mlp_nyquist_share": round(mlp_nyq, 4),
            "fno_run": fno_run, "mlp_run": mlp_run,
        })
        print(f"{building}: zones={n_zone} | abs energy FNO/MLP={fno_energy/mlp_energy:.3f} "
              f"| low-mode shape FNO={fno_low:.3f} MLP={mlp_low:.3f} "
              f"| Nyquist share FNO={fno_nyq:.3f} MLP={mlp_nyq:.3f}")

    csv_path = os.path.join(OUT_DIR, "spatial_spectrum_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    json_path = os.path.join(OUT_DIR, "spatial_spectrum_notes.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "what": "Zone-axis (spatial) PSD of actions, DC removed. Closes the conference-version axis mismatch (prior FFT is on zone axis; prior smoothness evidence was on time axis).",
            "low_mode_ratio": f"fraction of cross-zone energy in modes 1..{LOW_MODE_CUTOFF-1}; higher = smoother across zones = FNO prior working",
            "caveat": "Zone index = geometry-file room order, not guaranteed physical adjacency (verified in utils_building.py). Spatial frequency is over index order.",
            "rows": csv_rows,
        }, fh, indent=2)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

