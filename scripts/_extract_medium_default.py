"""抽 OfficeMedium 默认档 3-seed 真实末8窗均值 (FNO + MLP)。
每 run 独立 EventAccumulator, 防缓存串味。选带 245 evals 的真身目录。"""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "log_building"
W = 8
ZONES = 18
T_E, T_V, T_C = "test/avg_energy", "test/avg_violations", "test/avg_comfort_mean"

# 每 seed -> 目录前缀 (选真身: n_evals=245 的)
GROUPS = {
    "FNO_default": {
        "s42": "officemedium_fno_default_probe_s42_OfficeMedium_Hot_Dry_20260716_103411",
        "s0":  "officemedium_fno_default_1m_s0_OfficeMedium_Hot_Dry_20260717_194958",
        "s1":  "officemedium_fno_default_1m_s1_OfficeMedium_Hot_Dry_20260717_195025",
    },
    "MLP_default": {
        "s42": "officemedium_mlp_default_1m_s42_OfficeMedium_Hot_Dry_20260717_195822",
        "s0":  "officemedium_mlp_default_1m_s0_OfficeMedium_Hot_Dry_20260717_235506",
        "s1":  "officemedium_mlp_default_1m_s1_OfficeMedium_Hot_Dry_20260718_040544",
    },
    "NoGuide_default": {  # guidance_scale=0.0, 默认档; s1 真身=195310 (095710 是0-eval stale)
        "s42": "officemedium_fno_noguide_default_1m_s42_OfficeMedium_Hot_Dry_20260720_111245",
        "s0":  "officemedium_fno_noguide_default_1m_s0_OfficeMedium_Hot_Dry_20260720_224820",
        "s1":  "officemedium_fno_noguide_default_1m_s1_OfficeMedium_Hot_Dry_20260721_195310",
    },
}

def tail_mean(run_dir, tag, w=W):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    if not ev: return None
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags().get("scalars", []): return None
    vals = [s.value for s in acc.Scalars(tag)]
    if not vals: return None
    tail = vals[-w:]
    return sum(tail) / len(tail)

def mean_std(xs):
    n = len(xs); m = sum(xs)/n
    if n < 2: return m, 0.0
    var = sum((x-m)**2 for x in xs)/(n-1)
    return m, var**0.5

for grp, seeds in GROUPS.items():
    e_list, v_list, c_list = [], [], []
    print(f"\n=== {grp} ===")
    for sd, d in seeds.items():
        e = tail_mean(d, T_E); v = tail_mean(d, T_V); c = tail_mean(d, T_C)
        e_list.append(e); v_list.append(v)
        if c is not None: c_list.append(c)
        rate = v/ZONES*100 if v is not None else None
        print(f"  {sd}: energy={e:.1f}  viol={v:.3f}  每区率={rate:.1f}%  comfort={c:.3f}")
    em, es = mean_std(e_list); vm, vs = mean_std(v_list)
    cm, cs = mean_std(c_list) if c_list else (0,0)
    print(f"  >>> 3-seed: energy {em:.1f}±{es:.1f} | viol {vm:.3f}±{vs:.3f} (每区率 {vm/ZONES*100:.1f}%) | comfort {cm:.3f}±{cs:.3f}")
