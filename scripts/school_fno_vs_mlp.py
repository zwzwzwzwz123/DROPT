#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""School guided(FNO) vs MLP baseline: 3-seed 末段窗均值对比 (§1.4 口径)."""
import os, sys, glob, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
ROOT="c:/Users/zouwei/Desktop/DC/DROPT/log_building"
N_ZONE=25; W=8
TAGS={"energy":"test/avg_energy","violations":"test/avg_violations",
      "comfort":"test/avg_comfort_mean","reward":"test/reward"}
def tailmean(pat):
    per={k:[] for k in TAGS}
    for s in ["42","0","1"]:
        dl=sorted(glob.glob(f"{ROOT}/{pat.format(s=s)}"))
        if not dl:
            for k in TAGS: per[k].append(np.nan)
            continue
        ev=sorted(glob.glob(os.path.join(dl[-1],"events.out.tfevents.*")))[-1]
        acc=EventAccumulator(ev,size_guidance={"scalars":0}); acc.Reload()
        av=acc.Tags().get("scalars",[])
        for k,t in TAGS.items():
            per[k].append(np.mean([x.value for x in acc.Scalars(t)][-W:]) if t in av else np.nan)
    return {k:(np.nanmean(v),np.nanstd(v)) for k,v in per.items()}
fno=tailmean("school_guided_1m_s{s}_*")
mlp=tailmean("school_mlp_1m_s{s}_*")
print(f"=== School 3-seed 末{W}点窗均值 (mean±std) ===")
for k in TAGS:
    fm,fs=fno[k]; mm,ms=mlp[k]
    imp=(mm-fm)/mm*100 if k in("energy","violations","comfort") else None
    extra=f"  省/降 {imp:.1f}%" if imp is not None else ""
    print(f"  {k:11s}: FNO {fm:.2f}±{fs:.2f}  |  MLP {mm:.2f}±{ms:.2f}{extra}")
fv=fno["violations"][0]; mv=mlp["violations"][0]
print(f"\n  每区违规率: FNO {fv/N_ZONE*100:.1f}%  |  MLP {mv/N_ZONE*100:.1f}%")
