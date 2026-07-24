"""临时核查: OfficeMedium 默认档探针 vs 手调档对齐版, 各自末W窗均值。
每个 run 独立新建 EventAccumulator, 避免复用缓存导致数字串味。"""
import os, glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
W = 8
ZONES = 18

RUNS = {
    "probe_default_s42": "officemedium_fno_default_probe_s42_OfficeMedium_Hot_Dry_20260716_103411",
    "aligned_handtuned_s42": "officemedium_fno_aligned_1m_s42_OfficeMedium_Hot_Dry_20260714_215641",
    "aligned_handtuned_s0":  "officemedium_fno_aligned_1m_s0_OfficeMedium_Hot_Dry_20260714_215752",
    "aligned_handtuned_s1":  "officemedium_fno_aligned_1m_s1_OfficeMedium_Hot_Dry_20260714_215759",
}

def last_w_mean(run_dir, tag, w=W):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    if not ev:
        return None, 0
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None, -1
    vals = [s.value for s in acc.Scalars(tag)]
    if not vals:
        return None, 0
    tail = vals[-w:]
    return sum(tail) / len(tail), len(vals)

# 先探测可用 tag 名
sample_dir = RUNS["probe_default_s42"]
ev = sorted(glob.glob(os.path.join(ROOT, sample_dir, "events.out.tfevents.*")))
acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
tags = acc.Tags().get("scalars", [])
print("可用 scalar tags:", tags)
print()

# 猜测能耗/违规/comfort 的 tag
def pick(cands):
    for c in cands:
        if c in tags: return c
    return None
t_energy = pick(["test/avg_energy", "test/energy", "eval/energy"])
t_viol   = pick(["test/avg_violations", "test/violations", "eval/violations"])
t_comf   = pick(["test/avg_comfort_mean", "test/comfort_mean"])
print(f"用 tag: energy={t_energy}  viol={t_viol}  comfort={t_comf}\n")

print(f"{'run':26s} {'n':>5s} {'energy':>9s} {'viol':>7s} {'每区率':>7s} {'comfort':>8s}")
for name, d in RUNS.items():
    e, n = last_w_mean(d, t_energy)
    v, _ = last_w_mean(d, t_viol)
    c, _ = last_w_mean(d, t_comf) if t_comf else (None, 0)
    rate = (v / ZONES) if v is not None else None
    es = f"{e:9.1f}" if e is not None else "   n/a"
    vs = f"{v:7.3f}" if v is not None else "   n/a"
    rs = f"{rate*100:6.1f}%" if rate is not None else "   n/a"
    cs = f"{c:8.3f}" if c is not None else "     n/a"
    print(f"{name:26s} {n:5d} {es} {vs} {rs} {cs}")
