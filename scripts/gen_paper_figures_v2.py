#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成期刊版全套核心图 (PDF+PNG) —— 数字全部从 master_metrics_v2.csv 现读(可追溯到 run)。
先跑 extract_master_metrics.py 生成 CSV, 再跑本脚本。
参数量图用架构常量(stage4 §9.2, 非 run 指标); 训练曲线图从 event 现读。
SAC/SAC+MPC/modes扫描/空间谱: 数据未就绪, 暂缺(完训后加)。
输出: paper_figures_v2/*.pdf|png
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, csv, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(BASE, "paper_figures_v2")
CSV  = os.path.join(OUT, "master_metrics_v2.csv")

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42})
C_FNO, C_MLP, C_NG, C_ACC = "#2166AC", "#B2182B", "#F4A582", "#4D9221"

# ---- 读 master CSV -> D[(building,variant)] = row dict ----
D = {}
with open(CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        for k in ("energy_mean","energy_std","viol_mean","viol_std","per_zone_viol_pct",
                  "comfort_mean","comfort_std","reward_mean","reward_std"):
            r[k] = float(r[k])
        r["zones"] = int(r["zones"])
        D[(r["building"], r["variant"])] = r

BLD3 = ["OfficeSmall", "OfficeMedium", "SchoolPrimary"]
BLAB = ["OfficeSmall\n(6 zones)", "OfficeMedium\n(18 zones)", "SchoolPrimary\n(25 zones)"]

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, name + "." + ext))
    plt.close(fig); print("  ->", name)

print("生成图 (数据源: master_metrics_v2.csv):")

# ===== 图1: 三建筑能耗 FNO vs MLP (双子图, 量级差) =====
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for i, (ax, b) in enumerate(zip(axes, BLD3)):
    fe, fs = D[(b,"Full")]["energy_mean"], D[(b,"Full")]["energy_std"]
    me, ms = D[(b,"MLP")]["energy_mean"],  D[(b,"MLP")]["energy_std"]
    sv = (me - fe) / me * 100
    ax.bar([0,1], [fe,me], yerr=[fs,ms], color=[C_FNO,C_MLP], capsize=5,
           width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Guided-\nDiffFNO","Diff-MLP"])
    ax.set_title(BLAB[i], fontsize=11)
    ax.set_ylabel("Energy (kWh)" if i==0 else "")
    top = max(fe,me)*1.20; ax.set_ylim(0, top)
    ax.text(0.5, top*0.93, f"-{sv:.1f}%", ha="center", fontsize=12, color=C_ACC, fontweight="bold")
    for xi,v in zip([0,1],[fe,me]):
        ax.text(xi, v+top*0.02, f"{v:.0f}", ha="center", fontsize=9)
fig.suptitle("Energy Consumption across Three Buildings (3-seed, 1M steps)", fontsize=13)
fig.tight_layout(); save(fig, "fig1_three_building_energy")

# ===== 图2: 跨规模省幅曲线 (单调递增, 默认档统一后) =====
zones = [D[(b,"Full")]["zones"] for b in BLD3]
sav = [(D[(b,"MLP")]["energy_mean"]-D[(b,"Full")]["energy_mean"])/D[(b,"MLP")]["energy_mean"]*100 for b in BLD3]
fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.plot(zones, sav, "o-", color=C_FNO, lw=2.2, ms=11, mfc="white", mew=2)
for z, s, b in zip(zones, sav, BLD3):
    ax.annotate(f"{b}\n{s:.1f}%", (z,s), textcoords="offset points",
                xytext=(0, 14 if s<45 else -34), ha="center", fontsize=10)
ax.set_xlabel("Number of zones"); ax.set_ylabel("Energy saving vs Diff-MLP (%)")
ax.set_title("FNO advantage grows with building scale\n(unified default protocol)")
ax.set_ylim(0, 60); ax.set_xticks(zones)
fig.tight_layout(); save(fig, "fig2_saving_curve_monotonic")

# ===== 图3: 三建筑每区违规率 =====
fv = [D[(b,"Full")]["per_zone_viol_pct"] for b in BLD3]
mv = [D[(b,"MLP")]["per_zone_viol_pct"]  for b in BLD3]
fig, ax = plt.subplots(figsize=(7, 4.3)); xx=np.arange(3); w=0.36
ax.bar(xx-w/2, fv, w, label="Guided-DiffFNO", color=C_FNO, edgecolor="black", lw=0.6)
ax.bar(xx+w/2, mv, w, label="Diff-MLP", color=C_MLP, edgecolor="black", lw=0.6)
for i in range(3):
    ax.text(xx[i]-w/2, fv[i]+1, f"{fv[i]:.1f}%", ha="center", fontsize=9)
    ax.text(xx[i]+w/2, mv[i]+1, f"{mv[i]:.1f}%", ha="center", fontsize=9)
ax.set_xticks(xx); ax.set_xticklabels(BLAB, fontsize=9)
ax.set_ylabel("Per-zone comfort violation rate (%)")
ax.set_title("Per-zone Violation Rate (normalized by zone count)")
ax.set_ylim(0, 80); ax.legend()
fig.tight_layout(); save(fig, "fig3_violation_rate")

# ===== 图4: 骨干 vs 引导 解耦 (School 三方) =====
mlp = D[("SchoolPrimary","MLP")]; ng = D[("SchoolPrimary","NoGuide")]; full = D[("SchoolPrimary","Full")]
energy = [mlp["energy_mean"], ng["energy_mean"], full["energy_mean"]]
estd   = [mlp["energy_std"],  ng["energy_std"],  full["energy_std"]]
bb = (mlp["energy_mean"]-ng["energy_mean"])/mlp["energy_mean"]*100
gd = (ng["energy_mean"]-full["energy_mean"])/ng["energy_mean"]*100
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(range(3), energy, yerr=estd, capsize=6, color=[C_MLP,C_NG,C_FNO],
       edgecolor="black", lw=0.7, width=0.6)
for i,(v,s) in enumerate(zip(energy,estd)):
    ax.text(i, v+s+350, f"{v:.0f}±{s:.0f}", ha="center", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(["Diff-MLP","FNO\n(NoGuide)","Guided-DiffFNO\n(Full)"])
ax.set_ylabel("Energy (kWh)  —  SchoolPrimary"); ax.set_ylim(0, 16800)
ax.set_title("Decoupling backbone vs guidance (School, 3-seed)")
bbox = dict(boxstyle="round,pad=0.25", fc="white", ec=C_ACC, lw=0.8)
ax.annotate("", xy=(1, ng["energy_mean"]), xytext=(0, mlp["energy_mean"]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.text(0.42, 13000, f"backbone effect\n-{bb:.1f}%", ha="center", va="bottom", fontsize=9.5, color=C_ACC, bbox=bbox)
ax.annotate("", xy=(2, full["energy_mean"]), xytext=(1, ng["energy_mean"]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.text(1.5, 9700, f"guidance\n-{gd:.1f}%", ha="center", va="bottom", fontsize=9.5, color=C_ACC, bbox=bbox)
fig.tight_layout(); save(fig, "fig4_backbone_guidance_decoupling")

# ===== 图5: OfficeSmall 消融 (能耗 + 违规双轴) =====
order = ["Full","NoGuide","NoRes","NoRes_NoGuide","MLP"]
ab_lab = ["Full","w/o\nGuidance","w/o\nResidual","w/o Res\n& Guide","MLP\nbackbone"]
ab_e = [D[("OfficeSmall",v)]["energy_mean"] for v in order]
ab_es= [D[("OfficeSmall",v)]["energy_std"]  for v in order]
ab_v = [D[("OfficeSmall",v)]["viol_mean"]   for v in order]
ab_vs= [D[("OfficeSmall",v)]["viol_std"]    for v in order]
fig, ax1 = plt.subplots(figsize=(8, 4.5)); xx=np.arange(5); w=0.38
ax1.bar(xx-w/2, ab_e, w, yerr=ab_es, capsize=4, color=C_FNO, edgecolor="black", lw=0.6)
ax1.set_ylabel("Energy (kWh)", color=C_FNO); ax1.set_ylim(800, 1060); ax1.tick_params(axis="y", labelcolor=C_FNO)
ax2 = ax1.twinx(); ax2.grid(False)
ax2.bar(xx+w/2, ab_v, w, yerr=ab_vs, capsize=4, color=C_MLP, edgecolor="black", lw=0.6)
ax2.set_ylabel("Comfort violations (count)", color=C_MLP); ax2.set_ylim(0, 1.75); ax2.tick_params(axis="y", labelcolor=C_MLP)
ax1.set_xticks(xx); ax1.set_xticklabels(ab_lab, fontsize=9)
ax1.plot([], [], color=C_FNO, lw=6, label="Energy (kWh)"); ax1.plot([], [], color=C_MLP, lw=6, label="Comfort violations")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_title("Ablation (OfficeSmall, 3-seed): residual NOT essential; guidance cuts violations")
fig.tight_layout(); save(fig, "fig5_ablation_officesmall")

# ===== 图6: 参数效率 (架构常量, stage4 §9.2) =====
fno_p=[30876,125384,211260]; mlp_p=[210998,226370,235337]; ratio=[6.83,1.81,1.11]
fig, ax = plt.subplots(figsize=(7, 4.3)); xx=np.arange(3); w=0.36
ax.bar(xx-w/2, [p/1000 for p in fno_p], w, label="Guided-DiffFNO", color=C_FNO, edgecolor="black", lw=0.6)
ax.bar(xx+w/2, [p/1000 for p in mlp_p], w, label="Diff-MLP", color=C_MLP, edgecolor="black", lw=0.6)
for i in range(3):
    ax.text(xx[i], max(fno_p[i],mlp_p[i])/1000+6, f"{ratio[i]}x fewer", ha="center", fontsize=9, color=C_ACC, fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels(BLAB, fontsize=9)
ax.set_ylabel("Actor parameters (K)")
ax.set_title("Parameter efficiency: FNO advantage largest on small building")
ax.set_ylim(0, 320); ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.0), ncol=2, framealpha=0.95)
fig.tight_layout(); save(fig, "fig6_parameter_efficiency")

# ===== 图7: 三建筑舒适度 comfort_mean (是否在 ±1°C 容差带内) =====
fc = [D[(b,"Full")]["comfort_mean"] for b in BLD3]; fcs=[D[(b,"Full")]["comfort_std"] for b in BLD3]
mc = [D[(b,"MLP")]["comfort_mean"]  for b in BLD3]; mcs=[D[(b,"MLP")]["comfort_std"]  for b in BLD3]
fig, ax = plt.subplots(figsize=(7, 4.3)); xx=np.arange(3); w=0.36
ax.bar(xx-w/2, fc, w, yerr=fcs, capsize=4, label="Guided-DiffFNO", color=C_FNO, edgecolor="black", lw=0.6)
ax.bar(xx+w/2, mc, w, yerr=mcs, capsize=4, label="Diff-MLP", color=C_MLP, edgecolor="black", lw=0.6)
ax.axhline(1.0, ls="--", color="black", lw=1.2)
ax.text(2.4, 1.05, "±1°C comfort band", fontsize=9, color="black")
ax.set_xticks(xx); ax.set_xticklabels(BLAB, fontsize=9)
ax.set_ylabel("Mean temperature deviation (°C)")
ax.set_title("Comfort: FNO stays within band; MLP drifts out on School")
ax.set_ylim(0, 3.4); ax.legend()
fig.tight_layout(); save(fig, "fig7_comfort_mean")

# ===== 图8: 能耗-舒适 tradeoff 散点 (Pareto 视角, OfficeSmall 全变体) =====
fig, ax = plt.subplots(figsize=(6.8, 4.6))
sc = {"Full":(C_FNO,"o"),"NoGuide":(C_NG,"s"),"NoRes":("#5AAE61","^"),
      "NoRes_NoGuide":("#9970AB","D"),"MLP":(C_MLP,"v")}
sclab = {"Full":"Full","NoGuide":"w/o Guidance","NoRes":"w/o Residual",
         "NoRes_NoGuide":"w/o Res&Guide","MLP":"MLP backbone"}
for v,(c,mk) in sc.items():
    r = D[("OfficeSmall",v)]
    ax.errorbar(r["energy_mean"], r["viol_mean"], xerr=r["energy_std"], yerr=r["viol_std"],
                fmt=mk, color=c, ms=11, capsize=3, label=sclab[v], mec="black", mew=0.6)
ax.set_xlabel("Energy (kWh)  ← better"); ax.set_ylabel("Comfort violations (count)  ← better")
ax.set_title("Energy-Comfort Trade-off (OfficeSmall, 3-seed)")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); save(fig, "fig8_tradeoff_scatter")

# ===== 图9: 训练稳定性 (School Full vs MLP, 能耗全程曲线) =====
def series(run_dir, tag="test/avg_energy"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = sorted(glob.glob(os.path.join(BASE,"log_building",run_dir,"events.out.tfevents.*")))
    acc = EventAccumulator(ev[-1], size_guidance={"scalars":0}); acc.Reload()
    s = acc.Scalars(tag); return np.array([x.step for x in s]), np.array([x.value for x in s])
try:
    xf, yf = series("school_guided_1m_s42_SchoolPrimary_Hot_Dry_20260708_160354")
    xm, ym = series("school_mlp_1m_s42_SchoolPrimary_Hot_Dry_20260710_144107")
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(xf, yf, color=C_FNO, lw=1.4, label="Guided-DiffFNO (s42)")
    ax.plot(xm, ym, color=C_MLP, lw=1.4, alpha=0.85, label="Diff-MLP (s42)")
    ax.set_xlabel("Training epoch (test eval index)"); ax.set_ylabel("Test energy (kWh)  —  SchoolPrimary")
    ax.set_title("Training stability: FNO converges tight, MLP oscillates high")
    ax.legend()
    fig.tight_layout(); save(fig, "fig9_training_stability_school")
except Exception as e:
    print("  [跳过 fig9]", e)

print("完成。输出目录: paper_figures_v2/  (9 张图 + master CSV)")
