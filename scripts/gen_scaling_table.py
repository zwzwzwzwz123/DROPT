#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Slide 4 scaling 专用小表 (与 table1 同风格)。只放随区数单调、无争议的量:
Zones | Energy Saving | Violation Reduction。不含耦合强度(与省幅不同序,会自曝矛盾)。"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, csv

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(BASE, "paper_figures_v2")
CSV  = os.path.join(OUT, "master_metrics_v2.csv")
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 12})

D = {}
with open(CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        D[(r["building"], r["variant"])] = r

BLD = [("OfficeSmall","6"), ("OfficeMedium","18"), ("SchoolPrimary","25")]
rows = []
for b, z in BLD:
    fno, mlp = D[(b,"Full")], D[(b,"MLP")]
    e_sav = (float(mlp["energy_mean"]) - float(fno["energy_mean"])) / float(mlp["energy_mean"]) * 100
    v_red = (float(mlp["per_zone_viol_pct"]) - float(fno["per_zone_viol_pct"])) / float(mlp["per_zone_viol_pct"]) * 100
    rows.append([b, z, f"{e_sav:.1f}%", f"{v_red:.0f}%"])

cols = ["Building", "Zones", "Energy\nSaving", "Violation\nReduction"]
fig, ax = plt.subplots(figsize=(6.2, 2.7)); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.2)
for j in range(len(cols)):
    c = tbl[0, j]; c.set_facecolor("#1F3864"); c.set_text_props(color="white", fontweight="bold")
for i in range(len(rows)):
    for j in range(len(cols)):
        cell = tbl[i+1, j]; cell.set_facecolor("#D6E4F0")
        if j in (2, 3): cell.set_text_props(fontweight="bold")
fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(os.path.join(OUT, "table2_scaling." + ext), dpi=200, bbox_inches="tight")
print("-> table2_scaling (PNG+PDF)")
