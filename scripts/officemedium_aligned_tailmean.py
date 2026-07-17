#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OfficeMedium FNO 对齐版(L1/modes4/w64) 1M 3-seed: 末段窗均值 ± std.

抽三 seed(42/0/1) 末段 N 点窗均值/std，论文口径(§1.4)，
用于把 HANDOFF §1 主表 OfficeMedium 行从旧 L2/modes6(7016) 替换成对齐版正式值。
禁用单点 best/min。zone=18 (§2.2)。
"""
import os, sys, glob, argparse
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
RUNS = {
    "s42": "officemedium_fno_aligned_1m_s42_OfficeMedium_Hot_Dry_20260714_215641",
    "s0":  "officemedium_fno_aligned_1m_s0_OfficeMedium_Hot_Dry_20260714_215752",
    "s1":  "officemedium_fno_aligned_1m_s1_OfficeMedium_Hot_Dry_20260714_215759",
}
N_ZONE = 18
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
    ap.add_argument("--window", type=int, default=8)
    args = ap.parse_args()
    W = args.window
    data = {k: load(v) for k, v in RUNS.items()}
    print(f"=== 各 seed 末段 {W} 点窗均值 (test evals) ===")
    per_seed = {m: [] for m in TAGS}
    for seed, d in data.items():
        n = len(d.get("reward", []))
        line = [f"{seed} (n={n})"]
        for metric in TAGS:
            series = d.get(metric)
            if series is None or len(series) == 0:
                line.append(f"{metric}=NA"); per_seed[metric].append(np.nan); continue
            m = float(np.mean(series[-W:]))
            per_seed[metric].append(m)
            line.append(f"{metric}={m:.3f}")
        print("  " + "  ".join(line))
    print(f"\n=== 3-seed 聚合 (窗={W}): seed 窗均值的 mean ± std ===")
    for metric in TAGS:
        vals = np.array(per_seed[metric], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            print(f"  {metric}: NA"); continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        extra = ""
        if metric == "violations":
            rate = m / N_ZONE
            extra = f"   → 每区违规率 = {rate:.3f} ({rate*100:.1f}%)"
        print(f"  {metric:12s}: {m:.3f} ± {s:.3f}{extra}")

if __name__ == "__main__":
    main()
