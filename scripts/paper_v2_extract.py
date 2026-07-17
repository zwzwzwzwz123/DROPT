# -*- coding: utf-8 -*-
"""Extract training curves + tail-window means for all 3-building runs.
Outputs tidy CSVs into paper_figures_v2/data/ for the v2 figure set.
CPU-only, read-only on log_building. Does not touch training.
"""
import glob, os, json
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "log_building")
OUT = os.path.join(ROOT, "paper_figures_v2", "data")
os.makedirs(OUT, exist_ok=True)

# run group -> {seed: glob pattern}  (each resolves to exactly one dir)
GROUPS = {
    # School: three-way decoupling
    "school_fno_full":    {42: "school_guided_1m_s42_*",   0: "school_guided_1m_s0_*",   1: "school_guided_1m_s1_*"},
    "school_fno_noguide": {42: "school_fno_noguide_1m_s42_*",0: "school_fno_noguide_1m_s0_*",1: "school_fno_noguide_1m_s1_*"},
    "school_mlp":         {42: "school_mlp_1m_s42_SchoolPrimary_Hot_Dry_20260710_144107",
                            0: "school_mlp_1m_s0_*", 1: "school_mlp_1m_s1_*"},
    # OfficeMedium main table (L2/modes6)
    "med_fno":  {42: "diffusion_fno_guided_bcfix_clean_OfficeMedium_Hot_Dry_20260321_001905_scale0.5_100万步",
                 0: "officemedium_fno_1m_s0_*", 1: "officemedium_fno_1m_s1_*"},
    "med_mlp":  {42: "diffusion_mlp_bcfix_clean_OfficeMedium_Hot_Dry_20260323_232609",
                 0: "officemedium_mlp_1m_s0_*", 1: "officemedium_mlp_1m_s1_*"},
    # OfficeMedium aligned rerun (L1/modes4) -- may still be training; extract what exists
    "med_fno_aligned": {42: "officemedium_fno_aligned_1m_s42_*", 0: "officemedium_fno_aligned_1m_s0_*", 1: "officemedium_fno_aligned_1m_s1_*"},
    # OfficeSmall main table (Full=guided seed42 + Apr seed0/1)
    "small_fno_full": {42: "fno_guided——100万步",
                       0: "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260411_100255__guided_seed0",
                       1: "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260416_110146__guided_seed1"},
    "small_mlp": {42: "MLP——100万步",
                  0: "diffusion_mlp_bcfix_clean_OfficeSmall_Hot_Dry_20260415_100054__mlp_seed0",
                  1: "diffusion_mlp_bcfix_clean_OfficeSmall_Hot_Dry_20260420_022234__mlp_seed1"},
}

TAGS = ["test/avg_energy", "test/avg_violations", "test/avg_comfort_mean", "test/reward"]

def find_event(pattern):
    dirs = glob.glob(os.path.join(LOG, pattern))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    d = dirs[0]
    evs = glob.glob(os.path.join(d, "**", "events.out*"), recursive=True)
    if not evs:
        return None
    # largest event file (most data)
    evs.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return evs[0], os.path.basename(d)

def load_scalars(ev):
    ea = EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    steps = None
    for t in TAGS:
        if t in tags:
            sc = ea.Scalars(t)
            out[t] = np.array([x.value for x in sc], dtype=float)
            if steps is None or len(sc) > len(steps):
                steps = np.array([x.step for x in sc], dtype=float)
    out["_step"] = steps
    return out

WINDOW = 8
curve_rows = []   # long-form curve data
summary_rows = [] # tail-window per-seed

for grp, seeds in GROUPS.items():
    for seed, pat in seeds.items():
        res = find_event(pat)
        if res is None:
            print(f"[MISS] {grp} s{seed}: {pat}")
            continue
        ev, dname = res
        sc = load_scalars(ev)
        n = len(sc.get("test/avg_energy", []))
        if n == 0:
            print(f"[EMPTY] {grp} s{seed}: {dname}")
            continue
        step = sc["_step"]
        # curves
        for i in range(n):
            curve_rows.append({
                "group": grp, "seed": seed, "idx": i,
                "step": step[i] if step is not None and i < len(step) else i,
                "energy": sc["test/avg_energy"][i] if i < len(sc.get("test/avg_energy", [])) else np.nan,
                "violations": sc["test/avg_violations"][i] if i < len(sc.get("test/avg_violations", [])) else np.nan,
                "comfort": sc["test/avg_comfort_mean"][i] if i < len(sc.get("test/avg_comfort_mean", [])) else np.nan,
                "reward": sc["test/reward"][i] if i < len(sc.get("test/reward", [])) else np.nan,
            })
        # tail-window mean per seed
        def tail(tag):
            a = sc.get(tag)
            if a is None or len(a) == 0: return np.nan
            return float(np.mean(a[-WINDOW:]))
        summary_rows.append({
            "group": grp, "seed": seed, "n_points": n, "dir": dname,
            "energy": tail("test/avg_energy"),
            "violations": tail("test/avg_violations"),
            "comfort": tail("test/avg_comfort_mean"),
            "reward": tail("test/reward"),
        })
        print(f"[OK]  {grp} s{seed}: n={n}  E={tail('test/avg_energy'):.0f} V={tail('test/avg_violations'):.2f}")

curve_df = pd.DataFrame(curve_rows)
summ_df = pd.DataFrame(summary_rows)
curve_df.to_csv(os.path.join(OUT, "curves.csv"), index=False)
summ_df.to_csv(os.path.join(OUT, "tail_per_seed.csv"), index=False)

# aggregate seed-level -> mean/std per group
agg = summ_df.groupby("group").agg(
    energy_mean=("energy", "mean"), energy_std=("energy", "std"),
    viol_mean=("violations", "mean"), viol_std=("violations", "std"),
    comfort_mean=("comfort", "mean"), comfort_std=("comfort", "std"),
    reward_mean=("reward", "mean"), reward_std=("reward", "std"),
    n_seed=("seed", "count"),
).reset_index()
agg.to_csv(os.path.join(OUT, "group_agg.csv"), index=False)
print("\n=== GROUP AGG ===")
print(agg.to_string(index=False))

# zone counts & param counts (from handoff §9.2, verified from checkpoints)
META = {
    "OfficeSmall":  {"zones": 6,  "fno_params": 30876,  "mlp_params": 210998, "state_dim": 20},
    "OfficeMedium": {"zones": 18, "fno_params": 125384, "mlp_params": 226370, "state_dim": 56},
    "SchoolPrimary":{"zones": 25, "fno_params": 211260, "mlp_params": 235337, "state_dim": 77},
}
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(META, f, indent=2)
print("\nWrote:", OUT)
PY = None
