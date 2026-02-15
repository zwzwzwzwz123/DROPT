#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot reward curves from log_building runs whose folder name contains '100万步'.
Default tag: test/reward
"""

import argparse
import os
from typing import List, Tuple

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


def load_scalar(event_path: str, tag: str) -> List[Tuple[int, float]]:
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
    vals = acc.Scalars(tag)
    return [(v.step, float(v.value)) for v in vals]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def plot_curves(runs, tag: str, smooth_window: int, out_path: str) -> None:
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

    plt.figure(figsize=(10, 6))
    plotted = 0
    for name, event_path in sorted(runs):
        series = load_scalar(event_path, tag)
        if not series:
            continue
        series = sorted(series, key=lambda x: x[0])
        steps = np.array([s for s, _ in series], dtype=np.int64)
        vals = np.array([v for _, v in series], dtype=np.float32)
        if smooth_window and smooth_window > 1:
            vals_sm = smooth(vals, smooth_window)
            # mode="same" keeps the same length as input
            plt.plot(steps, vals_sm, label=name)
        else:
            plt.plot(steps, vals, label=name)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No runs contain tag: {tag}")

    plt.xlabel("Step")
    plt.ylabel(tag)
    title = f"Reward Curves ({tag}) - 100万步"
    if smooth_window and smooth_window > 1:
        title += f" (smooth={smooth_window})"
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reward curves across 100万步 runs.")
    parser.add_argument("--log-root", type=str, default="log_building")
    parser.add_argument("--tag", type=str, default="test/reward", help="Scalar tag to plot.")
    parser.add_argument("--smooth", type=int, default=0, help="Moving average window (0=off).")
    parser.add_argument("--out", type=str, default="", help="Output image path.")
    args = parser.parse_args()

    if not os.path.isdir(args.log_root):
        raise SystemExit(f"log_root not found: {args.log_root}")

    runs = []
    for name in os.listdir(args.log_root):
        if "100万步" not in name:
            continue
        path = os.path.join(args.log_root, name)
        if not os.path.isdir(path):
            continue
        event_path = find_event_file(path)
        if not event_path:
            continue
        runs.append((name, event_path))

    if not runs:
        raise SystemExit("No runs with '100万步' found.")

    out_base = args.out or os.path.join(args.log_root, "compare_reward_curves_100万步.png")
    base_root, base_ext = os.path.splitext(out_base)
    raw_path = f"{base_root}_raw{base_ext}"
    smooth_path = f"{base_root}_smooth{base_ext}"

    # Always generate raw curve
    plot_curves(runs, args.tag, 0, raw_path)

    # Generate smoothed curve if requested
    if args.smooth and args.smooth > 1:
        plot_curves(runs, args.tag, args.smooth, smooth_path)


if __name__ == "__main__":
    main()
