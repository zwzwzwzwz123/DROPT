#!/usr/bin/env bash
# OfficeLarge 统一 launcher —— 扩 office 家族到三栋 (Small/Medium/Large) + 给 School 一个近规模楼型对照。
# 用法: bash scripts/_large_launch.sh <full|mlp|noguide|sacmpc|sac> <seed>
#   full    = FNO + guidance0.5   (main_building_fno_guided_bcfix_clean.py)  主表行
#   mlp     = Diffusion-MLP        (main_building_bcfix_clean.py)             算省幅分母
#   noguide = FNO + guidance0.0    (同 FNO 脚本, 训练侧 guidance=0)           骨干解耦/增量第4点
#   sacmpc  = SAC+MPC              (rl_baseline_mpc_bcfixclean.py)            基线
#   sac     = 纯 SAC               (rl_baseline_bcfixclean.py)               基线
#
# 配置依据 (2026-07-22, ground truth from 三栋 paper_metadata.pkl 逐项核实):
#   统一默认档协议 (三栋 FNO/MLP 完全一致, 仅 width 随 state_dim 分化):
#     L1 / modes4 / guidance0.5 / bc0.8 / bc_final0.1 / bc_decay150000 / vp10
#     actor_lr1e-4 / critic_lr2e-5 / batch256 / update0.5 / buffer1M / 245ep
#   OfficeLarge: roomnum=23, state_dim=71 (=3*23+2), rfft长=12, modes4 谱保留 33%.
#   width=96: 满足地板 width>=state_dim (富余25, 比 Medium 的 8 宽松);
#             序列 48/64/96/128 对 state_dim 20/56/71/77 单调, 规律干净.
#   谱保留 100%/40%/33%/31% (Small/Medium/Large/School) 随区数递减, 一致.
# SAC 公平协议 (承 _sac_launch.sh + 防作弊红线):
#   BC schedule 逐楼对齐被比较的 Guided-FNO (Large=默认档 bc0.8/final0.1/decay150000/vp10);
#   唯一算法侧差异 = SAC 标准优化器 actor_lr=critic_lr=3e-4, update_per_step=1.0, buffer200k.
# ⚠️ Large 23 区 ≈ School 规模, 单 FNO 整训 245ep 估 ~24-28h/seed, 满-buffer ~2.5GB RAM.
#    当前高基线 RAM 下 FNO 必须串行 (stage5 §4). 冒烟先行: 加 --total-steps 4096 (=1ep) 验证环境.
set -euo pipefail
cd "$(dirname "$0")/.."

KIND="${1:?full|mlp|noguide|sacmpc|sac}"; SEED="${2:?42|0|1}"
PY=/c/Users/zouwei/anaconda3/envs/dropt/python.exe

# OfficeLarge 固定规格 (见头部)
BT=OfficeLarge
WIDTH=96          # state_dim=71, 地板>=71, 富余25
# ⚠️ max_power fix (2026-07-24): 默认 8000W 对 OfficeLarge 物理欠配 —— 稳态解析显示
#   BASEMENT(需+3.40)+3个PLENUM(+2.21)+DATACENTER_BASEMENT(+1.32) 共5区满功率都到不了26°C,
#   任何策略必崩(FNO塌成零动作/MLP高能耗均~100%违规)。8000W下所需动作mean0.83/max3.40/5区饱和;
#   其余三栋(Small0.15/Medium0.41/School0.28,0饱和)默认8000W够用,故仅本栋暴露。
#   32000W 后 mean0.21/max0.85/0饱和,落回三栋同档 —— 非调参,是按 BEAR 设计给大楼配大机组
#   (max_power 是 plant-sizing 物理参数,BEAR 原生支持逐楼/逐区设,见 utils_building.py:375)。
#   能耗可比性不破: energy=|action|×max_power, 物理制热量固定→读数不随 max_power 变。
MAXPOWER=32000    # OfficeLarge 专属 (三栋 Small/Medium/School 仍默认 8000)
# 统一默认档训练超参 (与 Small/School/Medium 主表 Full 完全一致)
BCW=0.8; BCF=0.1; BCD=150000; VP=10

case "$KIND" in
  full)
    SCRIPT=main_building_fno_guided_bcfix_clean.py
    EXTRA="--guidance-scale 0.5 --fno-width $WIDTH --fno-layers 1 --fno-modes 4 \
      --bc-weight $BCW --bc-weight-final $BCF --bc-weight-decay-steps $BCD --violation-penalty $VP"
    PREFIX="officelarge_fno_full_default_1m_s${SEED}" ;;
  noguide)
    SCRIPT=main_building_fno_guided_bcfix_clean.py
    EXTRA="--guidance-scale 0.0 --fno-width $WIDTH --fno-layers 1 --fno-modes 4 \
      --bc-weight $BCW --bc-weight-final $BCF --bc-weight-decay-steps $BCD --violation-penalty $VP"
    PREFIX="officelarge_fno_noguide_default_1m_s${SEED}" ;;
  mlp)
    SCRIPT=main_building_bcfix_clean.py
    EXTRA="--bc-weight $BCW --bc-weight-final $BCF --bc-weight-decay-steps $BCD --violation-penalty $VP"
    PREFIX="officelarge_mlp_default_1m_s${SEED}" ;;
  sacmpc)
    SCRIPT=rl_baseline_mpc_bcfixclean.py
    EXTRA="--actor-lr 3e-4 --critic-lr 3e-4 --update-per-step 1.0 --buffer-size 200000 \
      --violation-penalty $VP --bc-coef --expert-type mpc \
      --bc-weight $BCW --bc-weight-final $BCF --bc-weight-decay-steps $BCD"
    PREFIX="sacmpc_fair_large_1m_s${SEED}" ;;
  sac)
    SCRIPT=rl_baseline_bcfixclean.py
    EXTRA="--actor-lr 3e-4 --critic-lr 3e-4 --update-per-step 1.0 --buffer-size 200000 \
      --violation-penalty $VP"
    PREFIX="sac_pure_large_1m_s${SEED}" ;;
  *) echo "bad kind $KIND (full|mlp|noguide|sacmpc|sac)"; exit 1 ;;
esac

# 冒烟测试: 传第3个参数 smoke → 只跑 1ep 验证环境加载 (不写主表)
if [ "${3:-}" = smoke ]; then
  EXTRA="$EXTRA --total-steps 4096"
  PREFIX="${PREFIX}_SMOKE"
fi

LOG="run_logs/${PREFIX}.log"
echo "[large-launch] $KIND $BT seed=$SEED width=$WIDTH -> $LOG"
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  "$PY" "$SCRIPT" \
  --building-type "$BT" --weather-type Hot_Dry --seed "$SEED" --max-power "$MAXPOWER" \
  $EXTRA --log-prefix "$PREFIX" > "$LOG" 2>&1
