#!/usr/bin/env bash
# FNO modes 扫描 launcher —— 同楼受控扫谱截断强度 (支柱2/3 从跨楼相关升同楼因果)。
# 用法: bash scripts/_modes_sweep_launch.sh <medium|school> <modes> <seed>
#   例: bash scripts/_modes_sweep_launch.sh school 2 42
#
# 设计 (承 HANDOFF_stage5 §3 + stage4 §6.2 "先单 seed 筛信号"):
#   - anchored 在已有 m4 3-seed 主 run (default 档 Full)，扫描只需补 m2 / m8。
#   - rfft 长: medium=10 (m2/m4/m8=保留20%/40%/80%), school=13 (m2/m4/m8=15%/31%/62%)。
#   - OfficeSmall 不扫 (rfft长4, modes4=零截断, 无真截断点)。
#   - 配置与主表 Full 完全一致 (默认档 + guidance0.5)，仅改 --fno-modes，保证同楼单变量。
#   - 先单 seed(42) 筛信号: 看 test/avg_energy 是否随 modes 变; 有信号的档再补 s0/s1。
# ⚠️ 每点是整训 245ep (~数小时)。RAM: 单 FNO ~2.5GB(1M buffer)，安全上限约6重进程，需错峰。
set -euo pipefail
cd "$(dirname "$0")/.."

BLD="${1:?medium|school}"; MODES="${2:?2|4|8}"; SEED="${3:?42|0|1}"
PY=/c/Users/zouwei/anaconda3/envs/dropt/python.exe

# 逐楼配置 (与主表 Full 默认档完全一致，ground truth from paper_metadata):
#   三栋统一: L1 / guidance0.5 / 默认训练超参; width 随 state_dim 分化 (容量旋钮)。
case "$BLD" in
  medium) BT=OfficeMedium;  WIDTH=64  ;;   # rfft长10, state_dim56
  school) BT=SchoolPrimary; WIDTH=128 ;;   # rfft长13, state_dim77
  *) echo "bad building $BLD (只扫 medium/school; small 零截断不扫)"; exit 1 ;;
esac

# modes 合法性检查 (别扫超过 rfft 长的无效点)
case "$BLD" in
  medium) [ "$MODES" -le 10 ] || { echo "medium rfft长=10, modes=$MODES 超范围"; exit 1; } ;;
  school) [ "$MODES" -le 13 ] || { echo "school rfft长=13, modes=$MODES 超范围"; exit 1; } ;;
esac

PREFIX="${BLD}_fno_modes${MODES}_default_1m_s${SEED}"
LOG="run_logs/${PREFIX}.log"

# 默认档训练超参 (与 Small/School 主表 Full 同; medium 已选(a)统一默认档):
#   bc 0.8 / final 0.1 / decay 150000 / vp 10 / guidance 0.5 / L1
echo "[modes-sweep] $BT modes=$MODES seed=$SEED width=$WIDTH -> $LOG"
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$PY" main_building_fno_guided_bcfix_clean.py \
  --building-type "$BT" --weather-type Hot_Dry --seed "$SEED" \
  --guidance-scale 0.5 --fno-width "$WIDTH" --fno-layers 1 --fno-modes "$MODES" \
  --bc-weight 0.8 --bc-weight-final 0.1 --bc-weight-decay-steps 150000 \
  --violation-penalty 10 \
  --log-prefix "$PREFIX" > "$LOG" 2>&1
