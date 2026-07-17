#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时诊断：抽 SAC OfficeSmall 三 seed 训练全程曲线，判断没收敛 vs 收敛到烂解。"""
import os, sys, glob
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
RUNS = {
    "sac_s42": "sac_baseline_bcfixclean_OfficeSmall_Hot_Dry_20260407_083330",
    "sac_s0":  "sac_baseline_bcfixclean_OfficeSmall_Hot_Dry_20260416_003623__sac_seed0",
    "sac_s1":  "sac_baseline_bcfixclean_OfficeSmall_Hot_Dry_20260420_170912__sac_seed1",
}
TAGS = {"reward":"test/reward","energy":"test/avg_energy",
        "violations":"test/avg_violations","comfort":"test/avg_comfort_mean"}

def load(run_dir):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    acc = EventAccumulator(ev[-1], size_guidance={"scalars":0}); acc.Reload()
    avail = acc.Tags().get("scalars", [])
    return {n:np.array([s.value for s in acc.Scalars(t)]) for n,t in TAGS.items() if t in avail}, avail

for seed, rd in RUNS.items():
    d, avail = load(rd)
    print(f"\n=== {seed} ===")
    e = d.get("energy")
    if e is None:
        print("  no energy tag. avail:", [a for a in avail if 'test' in a]); continue
    n = len(e)
    # 分五段看趋势
    segs = np.array_split(np.arange(n), 5)
    for metric in ["energy","violations","reward","comfort"]:
        s = d.get(metric)
        if s is None: continue
        seg_means = [f"{np.mean(s[seg]):.1f}" for seg in segs]
        print(f"  {metric:11s} n={len(s):3d} 五段均值: {' -> '.join(seg_means)}   末8窗={np.mean(s[-8:]):.2f}")
