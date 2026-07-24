#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""追加两张图 (复用 extract_master_metrics 的 REG 目录映射, 保持同源):
 figA: 训练奖励曲线 FNO vs MLP × 三楼 (3-seed 均值±std 带) —— 支柱1 收敛性佐证
 figB: OfficeSmall 消融 2×2 热力图 (残差×引导, 能耗+违规) —— 诚实成分刻画
数据: 逐 epoch test/reward、test/avg_energy、test/avg_violations, 从 event 现读。
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(BASE, "paper_figures_v2")
ROOT = os.path.join(BASE, "log_building")
sys.path.insert(0, os.path.dirname(__file__))
from extract_master_metrics import REG  # 同源目录映射

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42})
C_FNO, C_MLP = "#2166AC", "#B2182B"

def series(run_dir, tag):
    ev = sorted(glob.glob(os.path.join(ROOT, run_dir, "events.out.tfevents.*")))
    if not ev: return None, None
    acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0}); acc.Reload()
    if tag not in acc.Tags().get("scalars", []): return None, None
    s = acc.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])

def seed_band(dirs, tag, npts=245):
    """多 seed 对齐到统一 epoch 轴, 返回 (x, mean, std)。按 epoch index 对齐。"""
    ys = []
    for d in dirs:
        _, y = series(d, tag)
        if y is not None and len(y) > 0:
            ys.append(y[:npts])
    if not ys: return None, None, None
    m = min(len(y) for y in ys)
    ys = np.array([y[:m] for y in ys])
    x = np.arange(m)
    return x, ys.mean(0), ys.std(0)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, name + "." + ext))
    plt.close(fig); print("  ->", name)

print("生成追加图:")

# ===== figA: 训练奖励曲线 FNO vs MLP × 三楼 =====
BLD3 = ["OfficeSmall", "OfficeMedium", "SchoolPrimary"]
BLAB = ["OfficeSmall (6 zones)", "OfficeMedium (18 zones)", "SchoolPrimary (25 zones)"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, b, lab in zip(axes, BLD3, BLAB):
    for var, c, name in [("Full", C_FNO, "Guided-DiffFNO"), ("MLP", C_MLP, "Diffusion-MLP")]:
        x, m, sd = seed_band(REG[(b, var)], "test/reward")
        if m is None: continue
        ax.plot(x, m, color=c, lw=1.8, label=name)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.18)
    ax.set_title(lab, fontsize=11)
    ax.set_xlabel("Training epoch")
    ax.legend(fontsize=9, loc="lower right")
axes[0].set_ylabel("Test reward (3-seed mean ± std)")
fig.suptitle("Training reward: Guided-DiffFNO vs Diffusion-MLP across scales", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, "figA_reward_curves_fno_vs_mlp")

# ===== figB: OfficeSmall 消融 2×2 热力图 (残差×引导) =====
# 行=残差 On/Off, 列=引导 On/Off。cell 值取末8点窗均值。
W = 8
def tail(dirs, tag):
    vals = []
    for d in dirs:
        _, y = series(d, tag)
        if y is not None and len(y) > 0:
            vals.append(np.mean(y[-W:]))
    return np.mean(vals) if vals else np.nan

# 变体映射: (残差, 引导) -> REG key
CELLS = {("On","On"):"Full", ("On","Off"):"NoGuide",
         ("Off","On"):"NoRes", ("Off","Off"):"NoRes_NoGuide"}
rows, cols = ["On","Off"], ["On","Off"]  # 残差, 引导
def grid(tag):
    g = np.zeros((2,2))
    for i,r in enumerate(rows):
        for j,cc in enumerate(cols):
            g[i,j] = tail(REG[("OfficeSmall", CELLS[(r,cc)])], tag)
    return g

g_e = grid("test/avg_energy")
g_v = grid("test/avg_violations")
g_v_pct = g_v / 6.0 * 100  # OfficeSmall 6 区 -> 每区违规率

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, g, title, cmap, fmt in [
    (axes[0], g_e, "Episode energy (kWh)", "Blues_r", "{:.0f}"),
    (axes[1], g_v_pct, "Per-zone violation rate (%)", "Reds", "{:.1f}")]:
    im = ax.imshow(g, cmap=cmap, aspect="auto")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Guidance On","Guidance Off"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Residual On","Residual Off"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, fmt.format(g[i,j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="black")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("OfficeSmall ablation: residual × guidance (3-seed tail mean)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "figB_ablation_heatmap_officesmall")

print("完成。输出目录:", OUT)
