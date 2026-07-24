#!/usr/bin/env bash
# 无人值守滚动 launcher —— 自动维持 SAC 并行度、跑完一个补下一个，不依赖 Claude 会话。
# 只跑 Small/School 队列（medium 因统一决定 HOLD，全部跳过）。不碰探针、不碰正在跑的 seed42。
# 用法: nohup bash scripts/_sac_rolling.sh > run_logs/_rolling.log 2>&1 &
# 停止: touch run_logs/_rolling.STOP  (脚本每轮检查, 见到就优雅退出, 不杀在跑的 run)
set -uo pipefail
cd "$(dirname "$0")/.."

MAX_PARALLEL=3          # SAC 总并行上限(含正在跑的 seed42; 3 SAC + 探针已实测稳定)
POLL_SEC=300            # 每 5 分钟检查一次
STOP_FLAG=run_logs/_rolling.STOP
LAUNCH=scripts/_sac_launch.sh

# 队列: Small/School/Medium。格式 "kind building seed"。
# ✅ 2026-07-17: medium 解冻(探针坐实统一成立), 已追加到队尾。launcher medium 行已同步改为默认档。
# 先清 SAC+MPC(mpc) 再跑纯 SAC(sac)，与 SAC_QUEUE.md 对应。
QUEUE=(
  "mpc small 0"  "mpc small 1"  "mpc school 0"  "mpc school 1"
  "mpc medium 0" "mpc medium 1"
  "sac small 42" "sac small 0"  "sac small 1"
  "sac school 42" "sac school 0" "sac school 1"
  "sac medium 42" "sac medium 0" "sac medium 1"
)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 数当前活跃的 SAC 训练进程(命令行含 rl_baseline_*bcfixclean.py 的 python)
count_running(){
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'rl_baseline_(mpc_)?bcfixclean' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r' | grep -oE '[0-9]+' | head -1
}

# 判断某 prefix 是否已完训(Training finished)或已在跑(有 log)
is_done(){ grep -aq "Training finished" "run_logs/$1.log" 2>/dev/null; }
has_log(){ [ -f "run_logs/$1.log" ]; }

rm -f "$STOP_FLAG"
log "滚动 launcher 启动。队列 ${#QUEUE[@]} 个(Small/School/Medium)。MAX_PARALLEL=$MAX_PARALLEL。停止: touch $STOP_FLAG"

qi=0
while :; do
  if [ -f "$STOP_FLAG" ]; then log "见到 STOP 标志, 优雅退出(不杀在跑的 run)。"; rm -f "$STOP_FLAG"; break; fi

  # 队列全部处理完?
  if [ "$qi" -ge "${#QUEUE[@]}" ]; then
    log "队列已全部启动完毕。滚动 launcher 结束。剩余靠各 run 自行跑完。"; break
  fi

  running=$(count_running); running=${running:-0}
  if [ "$running" -lt "$MAX_PARALLEL" ]; then
    read -r kind bld seed <<< "${QUEUE[$qi]}"
    prefix="$([ "$kind" = mpc ] && echo sacmpc_fair || echo sac_pure)_${bld}_1m_s${seed}"
    if is_done "$prefix"; then
      log "跳过 $prefix (已完训)"; qi=$((qi+1)); continue
    fi
    if has_log "$prefix"; then
      log "跳过 $prefix (已有 log/在跑, 不重复启动)"; qi=$((qi+1)); continue
    fi
    log "并行 $running/$MAX_PARALLEL, 启动队列#$qi: $kind $bld s$seed -> $prefix"
    nohup bash "$LAUNCH" "$kind" "$bld" "$seed" >/dev/null 2>&1 &
    qi=$((qi+1))
    sleep 60   # 错峰: 避开 torch DLL + BEAR CSV + buffer 分配的启动内存尖峰叠加
  else
    log "并行已满 $running/$MAX_PARALLEL, 等待空位..."
  fi
  sleep "$POLL_SEC"
done
