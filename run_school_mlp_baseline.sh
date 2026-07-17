#!/usr/bin/env bash
# School MLP baseline —— 待三个 school_guided_1m run 跑完、GPU 空闲后再启动。
#
# 目的：给 SchoolPrimary 补 Diff-MLP 对照，用来证明 "同条件下 FNO 违规 < MLP"，
#       把 "School 违规 7.7 偏高" 从软肋翻成 "这栋楼本来就难，但 FNO 比谁都好"。
#
# 配置对齐依据（已于 2026-07-08 从运行中的 school_guided_1m_s42 的 paper_metadata.pkl
# 逐项核实）：School guided run 用的是脚本 **默认** 超参（OfficeSmall 档），
# 非 OfficeMedium 那套手调值。因此 MLP 基线只需覆盖 --building-type，其余全取默认即可
# 与 guided run 严格同配置（actor_lr 1e-4 / critic_lr 2e-5 / batch 256 / update 0.5 /
# bc 0.8→0.1@150k / hidden 256 / diffusion_steps 6 / violation_penalty 10 /
# episode_length 168 / expert mpc planning 3）。
#
# 入口脚本 main_building_bcfix_clean.py → algorithm=diffusion_mlp_bcfix_clean，
# 与 OfficeSmall/Medium 主表 MLP 同源，resolver 的 "MLP" matcher 能自动认出。
#
# 用法：GPU 空闲后，先跑 seed 42（主行）。要 3-seed 误差棒再解注 s0/s1。

set -e
cd "$(dirname "$0")"

# --- 主行：seed 42（对齐 guided 主行）---
python main_building_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry \
  --seed 42 --log-prefix school_mlp_1m_s42 2>&1 | tee run_logs/school_mlp_1m_s42.log

# --- 补齐 3-seed（可选；跑完 s42 确认无误后再放开）---
# python main_building_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry \
#   --seed 0 --log-prefix school_mlp_1m_s0 2>&1 | tee run_logs/school_mlp_1m_s0.log
# python main_building_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry \
#   --seed 1 --log-prefix school_mlp_1m_s1 2>&1 | tee run_logs/school_mlp_1m_s1.log
