#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成主结果表 (渲染成 PNG/PDF, 可直接插 PPT)。数据源 master_metrics_v2.csv (3-seed)。
列: Building | Method | Energy kWh | Per-zone Viol. % | Comfort MAD degC | Energy Saving."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(BASE, "paper_figures_v2")
CSV  = os.path.join(OUT, "master_metrics_v2.csv")
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 11})

D = {}
with open(CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        D[(r["building"], r["variant"])] = r

BLD = [("OfficeSmall","6"), ("OfficeMedium","18"), ("SchoolPrimary","25")]
C_FNO, C_MLP = "#D6E4F0", "#F7DDDD"

rows = []
for b, z in BLD:
    fno, mlp = D[(b,"Full")], D[(b,"MLP")]
    sav = (float(mlp["energy_mean"]) - float(fno["energy_mean"])) / float(mlp["energy_mean"]) * 100
    for var, r, name in [("Full", fno, "Guided-DiffFNO (ours)"), ("MLP", mlp, "Diffusion-MLP")]:
        e = f"{float(r['energy_mean']):.0f} ± {float(r['energy_std']):.0f}"
        v = f"{float(r['per_zone_viol_pct']):.1f}"
        c = f"{float(r['comfort_mean']):.2f} ± {float(r['comfort_std']):.2f}"
        s = f"{sav:.1f}%" if var == "Full" else "—"
        blab = f"{b}\n({z} zones)" if var == "Full" else ""
        rows.append([blab, name, e, v, c, s])

cols = ["Building", "Method", "Energy (kWh)", "Per-zone\nViolation (%)",
        "Comfort MAD\n(°C, lower=better)", "Energy\nSaving"]

fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.0)

# 表头样式
for j in range(len(cols)):
    c = tbl[0, j]; c.set_facecolor("#1F3864"); c.set_text_props(color="white", fontweight="bold")
# 行底色: FNO 蓝 / MLP 红; 加粗 ours
for i in range(len(rows)):
    is_fno = (i % 2 == 0)
    for j in range(len(cols)):
        cell = tbl[i+1, j]
        cell.set_facecolor(C_FNO if is_fno else C_MLP)
        if is_fno and j in (1, 2, 5):
            cell.set_text_props(fontweight="bold")

fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(os.path.join(OUT, "table1_main_results." + ext), dpi=200, bbox_inches="tight")
print("-> table1_main_results (PNG+PDF)")
