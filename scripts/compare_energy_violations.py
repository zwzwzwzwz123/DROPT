#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare test avg_energy and avg_violations across multiple log dirs.

Default: scan log_building/ for runs with TensorBoard events.
Outputs a single figure with two subplots (energy + violations).
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path when running from scripts/
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tensorboard.backend.event_processing import event_accumulator


def find_event_file(log_dir: str) -> str:
    for name in os.listdir(log_dir):
        if name.startswith("events.out.tfevents"):
            return os.path.join(log_dir, name)
    return ""


def load_scalars(event_path: str, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
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
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        if tag in acc.Tags().get("scalars", []):
            vals = acc.Scalars(tag)
            out[tag] = [(v.step, float(v.value)) for v in vals]
    return out


def summarize(values: List[Tuple[int, float]], mode: str, k: int) -> float:
    if not values:
        return float("nan")
    values = sorted(values, key=lambda x: x[0])
    vals = [v for _, v in values]
    if mode == "last":
        return vals[-1]
    if mode == "best":
        return float(np.min(vals))
    # mean_last_k
    kk = max(1, min(k, len(vals)))
    return float(np.mean(vals[-kk:]))


def resolve_logs(log_root: str, logs: List[str]) -> List[str]:
    if logs:
        resolved = []
        for item in logs:
            if os.path.isdir(item):
                resolved.append(item)
                continue
            candidate = os.path.join(log_root, item)
            if os.path.isdir(candidate):
                resolved.append(candidate)
        return resolved

    # auto-scan
    found = []
    if not os.path.isdir(log_root):
        return found
    filter_key = "100万步"
    for name in os.listdir(log_root):
        path = os.path.join(log_root, name)
        if not os.path.isdir(path):
            continue
        if filter_key not in name:
            continue
        if find_event_file(path):
            found.append(path)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare avg_energy and avg_violations across runs.")
    parser.add_argument("--log-root", type=str, default="log_building")
    parser.add_argument(
        "--logs",
        type=str,
        default="",
        help="Comma-separated log dir names or paths (default: auto-scan log-root).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mean_last_k",
        choices=["last", "mean_last_k", "best"],
        help="How to summarize scalar series per run.",
    )
    parser.add_argument("--k", type=int, default=5, help="K for mean_last_k.")
    parser.add_argument("--out", type=str, default="", help="Output image path.")
    args = parser.parse_args()

    logs = [s.strip() for s in args.logs.split(",") if s.strip()]
    log_dirs = resolve_logs(args.log_root, logs)
    if not log_dirs:
        raise SystemExit("No log directories found.")

    tags = ["test/avg_energy", "test/avg_violations"]
    names = []
    energy_vals = []
    viol_vals = []

    for log_dir in log_dirs:
        event_path = find_event_file(log_dir)
        if not event_path:
            continue
        scalars = load_scalars(event_path, tags)
        energy = summarize(scalars.get(tags[0], []), args.mode, args.k)
        viol = summarize(scalars.get(tags[1], []), args.mode, args.k)
        if np.isnan(energy) and np.isnan(viol):
            continue
        names.append(os.path.basename(log_dir))
        energy_vals.append(energy)
        viol_vals.append(viol)

    if not names:
        raise SystemExit("No runs with target metrics found.")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib required: {exc}")

    # Ensure Chinese text renders correctly on Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(names))
    width = 0.6

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.6), 8), sharex=True)
    ax1, ax2 = axes

    ax1.bar(x, energy_vals, width=width, color="#1f77b4")
    ax1.set_ylabel("test/avg_energy")
    ax1.set_title(f"Energy ({args.mode}, k={args.k})")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, viol_vals, width=width, color="#d62728")
    ax2.set_ylabel("test/avg_violations")
    ax2.set_title(f"Violations ({args.mode}, k={args.k})")
    ax2.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")

    fig.suptitle("Energy and Violations Comparison", y=0.98)
    fig.tight_layout()

    out_path = args.out or os.path.join(args.log_root, "compare_energy_violations.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
