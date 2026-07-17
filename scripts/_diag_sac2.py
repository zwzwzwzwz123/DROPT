#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断2：SAC+MPC 曲线 + SAC 的 alpha/熵/rew 关系（判断是否熵主导）。"""
import os, sys, glob
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
RUNS = {
    "sacmpc_s42": "sac_baseline_mpc_bcfixclean_OfficeSmall_Hot_Dry_20260407_185554",
    "sac_s42":    "sac_baseline_bcfixclean_OfficeSmall_Hot_Dry_20260407_083330",
}

def load(run_dir):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    acc = EventAccumulator(ev[-1], size_guidance={"scalars":0}); acc.Reload()
    avail = acc.Tags().get("scalars", [])
    return acc, avail

for name, rd in RUNS.items():
    acc, avail = load(rd)
    print(f"\n=== {name} ===")
    print("  all scalar tags:", avail)
    for tag in ["test/avg_energy","test/avg_violations","test/avg_comfort_mean",
                "train/alpha","train/entropy","train/loss/alpha","train/reward",
                "train/loss/actor","train/loss/critic1"]:
        if tag in avail:
            v = np.array([s.value for s in acc.Scalars(tag)])
            if len(v):
                print(f"  {tag:26s}: first={v[0]:.4f} last={v[-1]:.4f} min={v.min():.4f} max={v.max():.4f}")
