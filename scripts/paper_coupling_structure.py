#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script B — per-building thermal coupling structure (OfficeMedium "valley" hypothesis).

The FNO spectral-truncation prior filters along the *zone index* axis. It aligns
with the true thermal coupling ONLY when strongly-coupled zones are also adjacent
in index order (a "banded" coupling matrix). Where the coupling graph is scattered
relative to the index order, the 1D low-pass prior is misaligned and its benefit
should shrink.

Candidate physical explanation for the OfficeMedium valley (truncation 4/10 but
only 3% energy saving) vs SchoolPrimary. This script computes, per building, the
structural quantities the hypothesis needs. SchoolPrimary's advantage number is
filled in tomorrow from the 1M run; the structure here is fixed and comparable.

Pure offline: instantiates each env once (no training, no GPU), reads A_d/connectmap.
"""
from __future__ import annotations

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

from env.building_env_wrapper import BearEnvWrapper

OUT_DIR = os.path.join(ROOT_DIR, "paperfigure_coupling")
BUILDINGS = ["OfficeSmall", "OfficeMedium", "SchoolPrimary"]
MODES = 4  # unified truncation used by the School 1M run and OfficeSmall main table
COMMON = dict(
    weather_type="Hot_Dry",
    location="Tucson",
    target_temp=26.0,
    temp_tolerance=1.0,
    max_power=8000,
    time_resolution=3600,
)

def _coupling_matrix(building_type: str):
    """Instantiate env once, return (A_d off-diagonal coupling, connectmap adjacency)."""
    env = BearEnvWrapper(building_type=building_type, **COMMON)
    bear = env.bear_env
    A_d = np.asarray(bear.A_d, dtype=np.float64)          # discrete-time state matrix, [n,n]
    # connectmap is [n, n+1]; last column is ground. Keep zone-zone block only.
    connect = np.asarray(bear.connectmap, dtype=np.float64)[:, : A_d.shape[0]]
    return A_d, connect


def _offdiag_coupling_ratio(A_d: np.ndarray) -> float:
    """Total off-diagonal magnitude / total diagonal magnitude of A_d.

    A_d = expm(A*dt); off-diagonal entries are inter-zone heat transfer over one
    step. Higher ratio = stronger cross-zone coupling = more room for a spatial
    (cross-zone) prior to help at all.
    """
    n = A_d.shape[0]
    diag = np.abs(np.diag(A_d)).sum()
    off = np.abs(A_d).sum() - diag
    return float(off / diag) if diag > 0 else float("nan")


def _bandedness(M: np.ndarray) -> float:
    """Coupling-weighted mean |i-j| distance, normalized to [0,1] against the
    uniform-scatter baseline. 0 = all coupling on the diagonal (perfectly banded,
    FNO index-FFT prior aligned); ->1 = coupling spread across the index order
    (scattered, prior misaligned). Size-fair: normalized by n so buildings with
    different zone counts are comparable.
    """
    n = M.shape[0]
    if n < 2:
        return 0.0
    W = np.abs(M).copy()
    np.fill_diagonal(W, 0.0)
    total = W.sum()
    if total <= 0:
        return float("nan")
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float64)
    weighted_mean_dist = (W * dist).sum() / total
    # Uniform-scatter baseline: mean |i-j| over all off-diagonal pairs = (n+1)/3.
    baseline = (n + 1.0) / 3.0
    return float(weighted_mean_dist / baseline)


def _spectral_time_constants(A_d: np.ndarray, dt: float) -> dict:
    """Eigenvalues of A_d -> continuous-time decay rates -> thermal time constants (h).
    Slow modes (long tau) => low-frequency-dominated spatial response => favorable
    to a low-pass spatial prior.
    """
    eig = np.linalg.eigvals(A_d)
    mag = np.clip(np.abs(eig), 1e-12, 0.999999)
    tau_hours = -dt / np.log(mag) / 3600.0
    return {
        "tau_min_h": float(np.min(tau_hours)),
        "tau_max_h": float(np.max(tau_hours)),
        "tau_median_h": float(np.median(tau_hours)),
        "spectral_radius": float(np.max(np.abs(eig))),
    }
def _spatial_smoothness_of_optimal(A_d: np.ndarray, connect: np.ndarray) -> float:
    """Diagnostic: how index-smooth is the coupling row-structure itself.
    Mean |first difference| of each zone's coupling profile along the index axis,
    normalized by its own magnitude. Low => neighbors-in-index share coupling
    profiles => an index-FFT low-pass prior fits. Complements _bandedness.
    """
    W = np.abs(A_d).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    if n < 3:
        return float("nan")
    diffs = np.abs(np.diff(W, axis=1))
    denom = W[:, :-1] + W[:, 1:] + 1e-12
    return float(np.mean(diffs / denom))


# SchoolPrimary advantage is filled tomorrow from the 1M/guidance0.5/3-seed run.
# OfficeSmall/OfficeMedium energy-saving % are from the current main table (§2.3).
ENERGY_SAVING_PCT = {
    "OfficeSmall": 21.0,
    "OfficeMedium": 3.0,
    "SchoolPrimary": None,  # TODO: fill from school_guided_1m_s{42,0,1} tail means
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for b in BUILDINGS:
        A_d, connect = _coupling_matrix(b)
        n = A_d.shape[0]
        rfft_len = n // 2 + 1
        retained = min(MODES, rfft_len)
        row = {
            "building": b,
            "zones": n,
            "rfft_len": rfft_len,
            "modes": MODES,
            "modes_retained": retained,
            "truncation_frac": round(1.0 - retained / rfft_len, 3),
            "offdiag_coupling_ratio": round(_offdiag_coupling_ratio(A_d), 4),
            "bandedness_Ad": round(_bandedness(A_d), 4),
            "bandedness_connectmap": round(_bandedness(connect), 4),
            "index_smoothness": round(_spatial_smoothness_of_optimal(A_d, connect), 4),
            "energy_saving_pct": ENERGY_SAVING_PCT[b],
        }
        row.update({f"eig_{k}": round(v, 4) for k, v in _spectral_time_constants(A_d, COMMON["time_resolution"]).items()})
        rows.append(row)
        print(f"{b}: zones={n} trunc={row['truncation_frac']:.0%} "
              f"coupling={row['offdiag_coupling_ratio']:.3f} "
              f"banded(A_d)={row['bandedness_Ad']:.3f} banded(connect)={row['bandedness_connectmap']:.3f} "
              f"save={row['energy_saving_pct']}")

    csv_path = os.path.join(OUT_DIR, "coupling_structure.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "hypothesis": (
            "FNO's 1D low-pass prior acts along the zone-INDEX axis. It helps only "
            "when strong coupling is banded (concentrated near the diagonal) in that "
            "index order. If OfficeMedium is more scattered (higher bandedness) than "
            "SchoolPrimary, the index-FFT prior is misaligned there -> explains the "
            "valley (trunc 4/10 but only 3% saving)."
        ),
        "prediction_to_test_tomorrow": (
            "Rank buildings by bandedness_Ad (low=aligned). If FNO energy-saving % "
            "tracks alignment (OfficeSmall & SchoolPrimary aligned+high-save, "
            "OfficeMedium scattered+low-save), the valley is explained by index/topology "
            "misalignment, not by non-convergence."
        ),
        "caveats": [
            "SchoolPrimary energy_saving_pct is null until the 1M run completes; the "
            "old 164k/guidance0 number (44%) is NOT comparable and must not be used.",
            "bandedness measures alignment of coupling with the zone INDEX order, which "
            "is the geometry-file room order, not guaranteed physical adjacency. This is "
            "exactly the honesty caveat for the spatial-spectrum figure (script A).",
        ],
        "rows": rows,
    }
    json_path = os.path.join(OUT_DIR, "coupling_hypothesis.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

