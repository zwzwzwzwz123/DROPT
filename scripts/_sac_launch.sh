#!/usr/bin/env bash
# 统一 SAC / SAC+MPC 基线 launcher —— 逐楼公平配置封死，避免手敲出错。
# 用法: bash scripts/_sac_launch.sh <mpc|sac> <small|medium|school> <seed>
#   mpc = SAC+MPC (rl_baseline_mpc_bcfixclean.py, 带 MPC 行为克隆)
#   sac = 纯 SAC   (rl_baseline_bcfixclean.py, 无专家)
# 公平协议(阶段四 §6.1 + 用户防作弊纠正):
#   - BC schedule 逐楼对齐被比较的 Guided-FNO (ground truth from paper_metadata):
#       Small/School: bc 0.8 / final 0.1 / decay 150000 / vp 10
#       OfficeMedium: bc 1.0 / final 0.6 / decay 200000 / vp 12
#   - 唯一算法侧差异 = SAC 标准优化器: actor_lr=critic_lr=3e-4, update_per_step=1.0
#     (扩散策略的 2e-5 critic_lr 会不公平地废掉 SAC = 旧数据 6.8x 的病根)
#   - buffer 200000, 1M steps
set -euo pipefail
cd "$(dirname "$0")/.."

KIND="${1:?mpc|sac}"; BLD="${2:?small|medium|school}"; SEED="${3:?42|0|1}"
PY=/c/Users/zouwei/anaconda3/envs/dropt/python.exe

# 🔴 medium 行待改 (2026-07-16): 用户决定 OfficeMedium 训练超参统一到默认档
#    (bc_final 0.6->0.1 / decay 200000->150000 / vp 12->10, = 与 small/school 一致)。
#    当前值仍对齐【旧手调 FNO】, 已失效, 请勿用它启动 medium run (SAC_QUEUE #6,7,13,14,15 已 HOLD)。
#    解冻: OfficeMedium FNO@默认档探针验证成立后, 把下行 medium 改成 BCW=0.8;BCF=0.1;BCD=150000;VP=10。
case "$BLD" in
  small)  BT=OfficeSmall;   BCW=0.8; BCF=0.1; BCD=150000; VP=10 ;;
  medium) BT=OfficeMedium;  BCW=1.0; BCF=0.6; BCD=200000; VP=12 ;;   # ← 失效, 待改默认档
  school) BT=SchoolPrimary; BCW=0.8; BCF=0.1; BCD=150000; VP=10 ;;
  *) echo "bad building $BLD"; exit 1 ;;
esac

COMMON="--building-type $BT --weather-type Hot_Dry --seed $SEED \
  --actor-lr 3e-4 --critic-lr 3e-4 --update-per-step 1.0 --buffer-size 200000 \
  --violation-penalty $VP"

if [ "$KIND" = mpc ]; then
  SCRIPT=rl_baseline_mpc_bcfixclean.py
  EXTRA="--bc-coef --expert-type mpc --bc-weight $BCW --bc-weight-final $BCF --bc-weight-decay-steps $BCD"
  PREFIX="sacmpc_fair_${BLD}_1m_s${SEED}"
elif [ "$KIND" = sac ]; then
  SCRIPT=rl_baseline_bcfixclean.py       # 纯 SAC, 无专家/无 BC
  EXTRA=""
  PREFIX="sac_pure_${BLD}_1m_s${SEED}"
else
  echo "bad kind $KIND"; exit 1
fi

LOG="run_logs/${PREFIX}.log"
echo "[launch] $KIND $BT seed=$SEED -> $LOG"
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$PY" "$SCRIPT" $COMMON $EXTRA --log-prefix "$PREFIX" > "$LOG" 2>&1
