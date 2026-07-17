#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Peek at the in-progress SchoolPrimary 1M runs (NOT converged; trend only)."""
import os, sys, glob
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
TAGS = ["test/reward", "test/avg_energy", "test/avg_violations", "test/avg_comfort_mean"]

def load(run_dir):
    ev = glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*"))[0]
    acc = EventAccumulator(ev, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for t in TAGS:
        if t in acc.Tags().get("scalars", []):
            out[t] = [(s.step, s.value) for s in acc.Scalars(t)]
    return out

data = {k: load(v) for k, v in RUNS.items()}

# how many test evals logged, and the epoch/step axis
for seed, d in data.items():
    r = d.get("test/reward", [])
    print(f"{seed}: {len(r)} test evals, last step={r[-1][0] if r else 'NA'}")

print("\n=== 关键指标：最新 5 个 test 评估点（step, value）===")
for tag in TAGS:
    print(f"\n--- {tag} ---")
    for seed, d in data.items():
        series = d.get(tag, [])
        tail = series[-5:]
        vals = " ".join(f"{v:.2f}" for _, v in tail)
        print(f"  {seed}: {vals}")

print("\n=== 起点 vs 当前（首个 / 最新 test 点）===")
for tag in TAGS:
    print(f"\n--- {tag} ---")
    for seed, d in data.items():
        series = d.get(tag, [])
        if series:
            print(f"  {seed}: first={series[0][1]:.2f}  latest={series[-1][1]:.2f}")
