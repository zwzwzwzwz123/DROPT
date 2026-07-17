#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SchoolPrimary 1M guided runs: 末段窗均值 ± std (per §1.4 protocol).

抽三 seed(42/0/1) 的末段 N 点窗口均值/标准差，输出论文口径指标，
用于替换 HANDOFF §2.3 主表里 SchoolPrimary 那行的旧 164k 数据。
禁用单点 best/min（§1.4 血泪教训）。
"""
import os, sys, glob, argparse
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
RUNS = {
    "s42": "school_guided_1m_s42_SchoolPrimary_Hot_Dry_20260708_160354",
    "s0":  "school_guided_1m_s0_SchoolPrimary_Hot_Dry_20260708_160618",
    "s1":  "school_guided_1m_s1_SchoolPrimary_Hot_Dry_20260708_160630",
}
# zone 数 (§2.2) 用于每区违规率归一化
N_ZONE = 25
TAGS = {
    "reward":       "test/reward",
    "energy":       "test/avg_energy",
    "violations":   "test/avg_violations",
    "comfort_mean": "test/avg_comfort_mean",
}

def load(run_dir):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    if not ev:
        raise FileNotFoundError(run_dir)
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    acc.Reload()
    avail = acc.Tags().get("scalars", [])
    out = {}
    for name, tag in TAGS.items():
        if tag in avail:
            out[name] = np.array([s.value for s in acc.Scalars(tag)])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8, help="末段窗口点数")
    args = ap.parse_args()
    W = args.window

    data = {k: load(v) for k, v in RUNS.items()}

    print(f"=== 各 seed 末段 {W} 点窗均值 (test evals) ===")
    per_seed = {}   # metric -> list of per-seed窗均值
    for metric in TAGS:
        per_seed[metric] = []
    for seed, d in data.items():
        n = len(d.get("reward", []))
        line = [f"{seed} (n={n})"]
        for metric in TAGS:
            series = d.get(metric)
            if series is None or len(series) == 0:
                line.append(f"{metric}=NA")
                per_seed[metric].append(np.nan)
                continue
            win = series[-W:]
            m = float(np.mean(win))
            per_seed[metric].append(m)
            line.append(f"{metric}={m:.3f}")
        print("  " + "  ".join(line))

    print(f"\n=== 3-seed 聚合 (窗={W}): seed 窗均值的 mean ± std ===")
    for metric in TAGS:
        vals = np.array(per_seed[metric], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            print(f"  {metric}: NA")
            continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        extra = ""
        if metric == "violations":
            rate = m / N_ZONE
            extra = f"   → 每区违规率 = {rate:.3f} ({rate*100:.1f}%)"
        print(f"  {metric:12s}: {m:.3f} ± {s:.3f}{extra}")

if __name__ == "__main__":
    main()
