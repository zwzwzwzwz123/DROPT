#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
权威抽数管线 —— 从 event 文件现算末段窗均值, 落 master CSV。
协议(承 school_tailmean.py / handoff §1.4): 每 seed 取末 W=8 点 test 评估窗均值,
再对 seed 取 mean±std。禁单点 best/min。每区违规率 = avg_violations / zone数。
输出: paper_figures_v2/master_metrics_v2.csv (所有图的唯一数据源, 可追溯到 run 目录)。
只登记【已完训】run; SAC/SAC+MPC 正重跑, 完训后再加进 REG。
"""
import os, sys, glob
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = "c:/Users/zouwei/Desktop/DC/DROPT/log_building"
OUT  = "c:/Users/zouwei/Desktop/DC/DROPT/paper_figures_v2/master_metrics_v2.csv"
W = 8
ZONES = {"OfficeSmall": 6, "OfficeMedium": 18, "SchoolPrimary": 25}

# (building, variant) -> [seed run 目录]。全部已完训、3-seed(或注明)。
REG = {
 ("OfficeSmall","Full"): [
   "fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_小_guidancescale=0.5_100万步",
   "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260411_100255__guided_seed0",
   "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260416_110146__guided_seed1"],
 ("OfficeSmall","NoGuide"): [
   "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260403_085019_小_无引导100万步",
   "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260412_163742__noguide_seed0",
   "diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260417_152156__noguide_seed1"],
 ("OfficeSmall","NoRes"): [
   "diffusion_fno_guided_bcfix_clean_nores_guided_OfficeSmall_Hot_Dry_20260405_143859",
   "diffusion_fno_guided_bcfix_clean_nores_guided_OfficeSmall_Hot_Dry_20260413_181507__nores_guided_seed0",
   "diffusion_fno_guided_bcfix_clean_nores_guided_OfficeSmall_Hot_Dry_20260418_101145__nores_guided_seed1"],
 ("OfficeSmall","NoRes_NoGuide"): [
   "diffusion_fno_guided_bcfix_clean_nores_noguide_OfficeSmall_Hot_Dry_20260404_100818_小_无残差100万步",
   "diffusion_fno_guided_bcfix_clean_nores_noguide_OfficeSmall_Hot_Dry_20260414_160709__nores_noguide_seed0",
   "diffusion_fno_guided_bcfix_clean_nores_noguide_OfficeSmall_Hot_Dry_20260419_075518__nores_noguide_seed1"],
 ("OfficeSmall","MLP"): [
   "diffusion_mlp_bcfix_clean_OfficeSmall_Hot_Dry_20260406_140006",
   "diffusion_mlp_bcfix_clean_OfficeSmall_Hot_Dry_20260415_100054__mlp_seed0",
   "diffusion_mlp_bcfix_clean_OfficeSmall_Hot_Dry_20260420_022234__mlp_seed1"],
 # 【07-19 选 a: 默认档统一, 替换旧手调档】
 ("OfficeMedium","Full"): [
   "officemedium_fno_default_probe_s42_OfficeMedium_Hot_Dry_20260716_103411",
   "officemedium_fno_default_1m_s0_OfficeMedium_Hot_Dry_20260717_194958",
   "officemedium_fno_default_1m_s1_OfficeMedium_Hot_Dry_20260717_195025"],
 ("OfficeMedium","MLP"): [
   "officemedium_mlp_default_1m_s42_OfficeMedium_Hot_Dry_20260717_195822",
   "officemedium_mlp_default_1m_s0_OfficeMedium_Hot_Dry_20260717_235506",
   "officemedium_mlp_default_1m_s1_OfficeMedium_Hot_Dry_20260718_040544"],
 # 【08-02 补录: guidance 解耦第三点。⚠️ s1 用真身 _195310, 不是 _095710(0-eval stale)】
 ("OfficeMedium","NoGuide"): [
   "officemedium_fno_noguide_default_1m_s42_OfficeMedium_Hot_Dry_20260720_111245",
   "officemedium_fno_noguide_default_1m_s0_OfficeMedium_Hot_Dry_20260720_224820",
   "officemedium_fno_noguide_default_1m_s1_OfficeMedium_Hot_Dry_20260721_195310"],
 ("SchoolPrimary","Full"): [
   "school_guided_1m_s42_SchoolPrimary_Hot_Dry_20260708_160354",
   "school_guided_1m_s0_SchoolPrimary_Hot_Dry_20260708_160618",
   "school_guided_1m_s1_SchoolPrimary_Hot_Dry_20260708_160630"],
 ("SchoolPrimary","NoGuide"): [
   "school_fno_noguide_1m_s42_SchoolPrimary_Hot_Dry_20260712_103848",
   "school_fno_noguide_1m_s0_SchoolPrimary_Hot_Dry_20260712_104030",
   "school_fno_noguide_1m_s1_SchoolPrimary_Hot_Dry_20260712_104047"],
 ("SchoolPrimary","MLP"): [
   "school_mlp_1m_s42_SchoolPrimary_Hot_Dry_20260710_144107",
   "school_mlp_1m_s0_SchoolPrimary_Hot_Dry_20260710_144450",
   "school_mlp_1m_s1_SchoolPrimary_Hot_Dry_20260710_144613"],
}
TAGS = {"reward":"test/reward","energy":"test/avg_energy",
        "violations":"test/avg_violations","comfort_mean":"test/avg_comfort_mean"}

def tail_mean(run_dir):
    """返回该 run 末 W 点各指标窗均值 dict; 找不到 event 抛错。"""
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    if not ev:
        raise FileNotFoundError(f"NO EVENT: {run_dir}")
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
    avail = acc.Tags().get("scalars", [])
    out = {}
    for name, tag in TAGS.items():
        if tag in avail:
            series = np.array([s.value for s in acc.Scalars(tag)])
            out[name] = float(np.mean(series[-W:])) if len(series) else np.nan
        else:
            out[name] = np.nan
    out["_n_evals"] = len(acc.Scalars(TAGS["reward"])) if TAGS["reward"] in avail else 0
    return out

rows = []
print(f"抽数 (末 {W} 点窗均值, 3-seed 聚合):")
for (bld, var), dirs in REG.items():
    per = {m: [] for m in TAGS}
    nmin = 1e9
    for d in dirs:
        tm = tail_mean(d)
        nmin = min(nmin, tm["_n_evals"])
        for m in TAGS:
            per[m].append(tm[m])
    agg = {}
    for m in TAGS:
        v = np.array(per[m], float); v = v[~np.isnan(v)]
        # ddof=1 样本标准差(除 n-1)，统一全项目口径(与 _extract_medium_default / _extract_sac_baselines 一致)
        _std = float(np.std(v, ddof=1)) if v.size >= 2 else 0.0
        agg[m] = (float(np.mean(v)), _std)
    zones = ZONES[bld]
    pzv = agg["violations"][0] / zones * 100   # 每区违规率 %
    rows.append({
        "building": bld, "zones": zones, "variant": var, "n_seed": len(dirs),
        "min_evals": int(nmin),
        "energy_mean": round(agg["energy"][0], 1), "energy_std": round(agg["energy"][1], 1),
        "viol_mean": round(agg["violations"][0], 3), "viol_std": round(agg["violations"][1], 3),
        "per_zone_viol_pct": round(pzv, 1),
        "comfort_mean": round(agg["comfort_mean"][0], 3), "comfort_std": round(agg["comfort_mean"][1], 3),
        "reward_mean": round(agg["reward"][0], 3), "reward_std": round(agg["reward"][1], 3),
        "runs": " | ".join(dirs),
    })
    print(f"  {bld:13s} {var:14s} E={agg['energy'][0]:8.1f}±{agg['energy'][1]:5.1f}"
          f"  V={agg['violations'][0]:6.2f} ({pzv:4.1f}%)  C={agg['comfort_mean'][0]:.2f}  n_eval>={int(nmin)}")

# 写 CSV
cols = ["building","zones","variant","n_seed","min_evals","energy_mean","energy_std",
        "viol_mean","viol_std","per_zone_viol_pct","comfort_mean","comfort_std",
        "reward_mean","reward_std","runs"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
import csv
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    wtr = csv.DictWriter(f, fieldnames=cols); wtr.writeheader(); wtr.writerows(rows)
print(f"\n-> master CSV: {OUT}  ({len(rows)} 行)")
