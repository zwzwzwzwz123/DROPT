#!/usr/bin/env bash
# OfficeLarge 无人值守串行 launcher —— 等当前重进程(s1 等)完训腾 RAM 后, 严格串行跑完全套 15 run.
# 用法: nohup bash scripts/_large_rolling.sh > run_logs/_large_rolling.log 2>&1 &
# 停止: touch run_logs/_large_rolling.STOP   (每轮检查, 优雅退出, 不杀在跑的 run)
#
# 三个保命机制 (FNO 长跑 + 无人值守):
#   1) 严格串行: 任何时刻只允许 1 个重训练进程(main_building*/rl_baseline*), 单 FNO 满-buffer
#      ~2.5GB, 高基线 RAM 下 2 个就 thrash (stage5 §4). 等前一个完训(进程消失)才发下一个.
#   2) RAM 守卫: 发车前要求空闲 RAM >= RAM_MIN_MB (stage5 §4: 单 FNO 需 >4-5GB).
#   3) 崩溃即停: 某 run 进程结束但日志无 'Training finished'(或有 Traceback) → 写 .HALTED
#      停整个队列, 不盲目续跑. 守 Phase1 探针精神: full 42 出问题就停下等人看, 不白烧 14 个 run.
# ⚠️ 队列首个是 full 42 = Phase1 探针; 它完训后可看省幅信号, 觉得不对就 touch STOP 停在这.
set -uo pipefail
cd "$(dirname "$0")/.."

POLL_SEC=180             # 每 3 分钟检查一次
RAM_MIN_MB=4500          # 发一个满-buffer FNO 前要求的空闲 RAM
START_TIMEOUT=600        # 发车后 10min 内必须出现 Epoch#, 否则判启动失败
STOP_FLAG=run_logs/_large_rolling.STOP
HALT_FLAG=run_logs/_large_rolling.HALTED
LAUNCH=scripts/_large_launch.sh

# 队列 "kind seed". 顺序: full/mlp 按 seed 交错 → 尽快拿省幅信号(full42+mlp42=第一个省幅点,
#   Phase1 探针看 office 序列是否单调) → 补齐 3-seed → NoGuide → SAC 基线.
QUEUE=(
  "full 42" "mlp 42"
  "full 0" "mlp 0"
  "full 1" "mlp 1"
  "noguide 42" "noguide 0" "noguide 1"
  "sacmpc 42" "sacmpc 0" "sacmpc 1"
  "sac 42" "sac 0" "sac 1"
)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# prefix 映射 (须与 _large_launch.sh 完全一致)
prefix_of(){
  case "$1" in
    full)    echo "officelarge_fno_full_default_1m_s${2}" ;;
    noguide) echo "officelarge_fno_noguide_default_1m_s${2}" ;;
    mlp)     echo "officelarge_mlp_default_1m_s${2}" ;;
    sacmpc)  echo "sacmpc_fair_large_1m_s${2}" ;;
    sac)     echo "sac_pure_large_1m_s${2}" ;;
  esac
}

# 数所有在跑的训练进程 (FNO/MLP=main_building*, SAC=rl_baseline*); 含本脚本外的(如 s1)
count_running(){
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'main_building|rl_baseline' } | Measure-Object).Count" \
    2>/dev/null | tr -d '\r' | grep -oE '[0-9]+' | head -1
}
free_ram_mb(){
  powershell.exe -NoProfile -Command \
    "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" \
    2>/dev/null | tr -d '\r' | grep -oE '[0-9]+' | head -1
}
is_done(){ grep -aq "Training finished" "run_logs/$1.log" 2>/dev/null; }
has_started(){ grep -aq "Epoch #" "run_logs/$1.log" 2>/dev/null; }
has_crash(){ grep -aqE "Traceback|CUDA out of memory|MemoryError|RuntimeError" "run_logs/$1.log" 2>/dev/null; }

rm -f "$STOP_FLAG" "$HALT_FLAG"
log "OfficeLarge 串行 launcher 启动。队列 ${#QUEUE[@]} run。严格串行 + RAM守卫(>=${RAM_MIN_MB}MB) + 崩溃即停。"
log "停止: touch $STOP_FLAG   (首个 full 42 = Phase1 探针, 完训后可看信号再决定是否续跑)"

for item in "${QUEUE[@]}"; do
  read -r kind seed <<< "$item"
  prefix=$(prefix_of "$kind" "$seed")

  [ -f "$STOP_FLAG" ] && { log "见到 STOP, 优雅退出。"; rm -f "$STOP_FLAG"; exit 0; }
  [ -f "$HALT_FLAG" ] && { log "见到 HALT, 退出。"; exit 1; }
  if is_done "$prefix"; then log "跳过 $prefix (已完训)"; continue; fi
  if [ -f "run_logs/$prefix.log" ]; then log "跳过 $prefix (已有 log, 不重复启动)"; continue; fi

  # 等发车条件: 无重进程在跑(串行) + RAM 充足
  while :; do
    [ -f "$STOP_FLAG" ] && { log "见到 STOP(等待中), 退出。"; rm -f "$STOP_FLAG"; exit 0; }
    r=$(count_running); r=${r:-0}
    ram=$(free_ram_mb); ram=${ram:-0}
    if [ "$r" -eq 0 ] && [ "$ram" -ge "$RAM_MIN_MB" ]; then break; fi
    log "等发车 $prefix: 在跑重进程=$r, 空闲RAM=${ram}MB (需 0 进程 & >=${RAM_MIN_MB}MB)"
    sleep "$POLL_SEC"
  done

  log "发车 $kind s$seed -> $prefix"
  nohup bash "$LAUNCH" "$kind" "$seed" >/dev/null 2>&1 &

  # 等启动: 出现 Epoch# 才算起来; 超时或启动即崩 → HALT
  t=0
  while ! has_started "$prefix"; do
    sleep 20; t=$((t+20))
    if has_crash "$prefix"; then log "❌ $prefix 启动即崩 (Traceback/OOM)。写 HALT 停队列。"; touch "$HALT_FLAG"; exit 1; fi
    if [ "$t" -ge "$START_TIMEOUT" ]; then log "❌ $prefix ${START_TIMEOUT}s 内未见 Epoch#, 疑似卡死。写 HALT。"; touch "$HALT_FLAG"; exit 1; fi
  done
  log "$prefix 已进入训练, 等待完训 (~15h for FNO)..."

  # 等完训: 进程消失后核验 'Training finished'; 无则判崩溃 HALT
  while :; do
    [ -f "$STOP_FLAG" ] && { log "见到 STOP(训练中), 优雅退出(不杀在跑的 $prefix)。"; rm -f "$STOP_FLAG"; exit 0; }
    r=$(count_running); r=${r:-0}
    if [ "$r" -eq 0 ]; then
      sleep 20   # 等日志落盘
      if is_done "$prefix"; then log "✅ $prefix 完训。"; break
      else log "❌ $prefix 进程已结束但日志无 'Training finished' (崩溃?)。写 HALT 停队列, 不盲目续跑。"; touch "$HALT_FLAG"; exit 1; fi
    fi
    sleep "$POLL_SEC"
  done
done

log "🎉 队列全部完训。OfficeLarge 全套 ${#QUEUE[@]} run 结束。"
