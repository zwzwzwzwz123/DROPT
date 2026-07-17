# HANDOFF —— SAC / SAC+MPC 基线诊断与重跑（阶段四 §6.1 第 2 项）

> 日期：2026-07-16。接续 `HANDOFF_stage4_alignment.md` §6.1（"修 SAC 基线 + 补 School 基线"）。
> 目标（用户原话）：不追顶刊，只要内容自洽、故事完整、能发表即可。语言：中文。
> **本文件 07-16 晚重写**：上一版写于机器 thrash 状态，把"本机内存约束/只能串行"当成结论，**经干净重启后实测推翻**（§4）。本版保留已核实的诊断（§1/§2/§3），改正并行度结论，记录已启动的重跑。

---

## 0. 一句话现状

**诊断完成 + 配置对着 ground-truth 核实无误 + 18 个 1M 重跑已启动（滚动池 3 并存）。** 上一版"因内存只能串行"是 thrash 误判：干净重启后空闲 8GB，实测 **3 个 SAC+MPC 并存安全**（RAM 剩 3.7GB / GPU 66%）。下一窗口只需**监控完训 + 抽末段窗均值填主表**。
> 🔴 **【07-16 晚·medium 5 个 run HOLD】** 用户决定把 OfficeMedium 训练超参统一到 Small/School 默认档（bc_final 0.6→0.1、vp 12→10 等 8 项，见 `HANDOFF_stage4_alignment.md` §9）。→ launcher medium 行现锁的旧手调对齐值失效，**SAC_QUEUE #6,7,13,14,15(所有 medium)已 HOLD**，等 OfficeMedium 默认档探针验证后改 launcher 再跑。正在跑的 #2(medium s42 bc0.6)保留作回退保险。**Small/School 12 个 run 不受影响、照常跑**（FNO 本就默认档）。
> 🔴 **防作弊红线（用户 07-16 纠正，务必守）**：SAC+MPC 的 BC 地板（专家模仿强度）**必须逐楼对齐被比较的 Guided-FNO**（Small/School=0.1、OfficeMedium=0.6），**不得为止漂而抬高**。上一版一度用 bc_final=0.5 把 SAC+MPC 拧稳到 ~905——那是"用 MPC 专家掩盖 SAC 缺陷"的作弊，已废弃。唯一允许的算法侧差异 = `critic_lr`（真 bug 修复）。

---

## 1. 【确证】SAC 失败根因 —— 不是"维度太多"，是配置 + 无 BC 锚定

> 用户原猜想："SAC 不行是因为维度太多、RL 难收敛"。**本轮数据否掉了"维度"这一半**：纯 SAC 在**最小的 OfficeSmall(6 区)** 上就崩。真因如下（读源码 + 抽 event 坐实）。

- **病根 1：`critic_lr` 严重偏低**。三栋 SAC/SAC+MPC 旧 run 全部 `critic_lr=2e-5`（Small/School）或 `5e-6`（OfficeMedium 手调），是 **actor_lr 的 1/15 ~ 1/10**。这是从扩散策略借来的保守值（`env/building_config.py:71` `DEFAULT_CRITIC_LR=2e-5`）。SAC 的 actor 完全靠 critic 的 Q 梯度做策略提升，critic 慢一个量级 → actor 在过时 Q 曲面上爬 → **能耗越训越高**（OfficeSmall 5060→5940）。标准 SAC 是 critic_lr = actor_lr = 3e-4。
- **病根 2（更致命）：纯 SAC 修 critic_lr 也救不回**。本轮在 OfficeSmall 试了 critic_lr=3e-4、update_per_step=1.0、reward_scale=0.01 三种修法，能耗仍塌在 ~4600-5300 / comfort 7-13°C。→ **纯在线 off-policy RL 在 BEAR 上无示范锚定就塌进退化盆地**（动作饱和到边界、烧满功率但不控温）。
- **反证（环境可解）**：SAC+MPC（带 MPC 专家 BC）早期摸到过能耗 843（比 FNO 871 还低）、comfort 0.39 —— 但随后被自身 SAC 梯度**漂回 1925**。说明坏的是 SAC 的在线策略提升，不是环境。

**结论一句话**：纯 SAC 不可救（config 全试过），是**诚实的负面结果**（且在最小楼就崩，堵死"没调好/维度太多"两种质疑）。

---

## 2. 【确证】诊断期实测数字（OfficeSmall 诊断 + School 纯 SAC）

> ⚠️ 下表是**诊断期的短 run / 早停 run**，用于定病根；**正式主表数字以 §5 启动的完整 1M 3-seed 为准**，届时替换。

OfficeSmall（6 区，参照 Guided-FNO=871 / comfort 0.44）：

| 配置 | 能耗(kWh) | comfort(°C) | 判定 |
|---|---|---|---|
| 旧纯 SAC (critic_lr 2e-5) | ~5940 | ~8 | 塌 |
| 纯 SAC + critic_lr 3e-4 | ~5300 | ~10 | **仍塌** |
| 纯 SAC + critic_lr + reward_scale 0.01 | ~4600 | ~7 | **仍塌** |
| 旧 SAC+MPC (critic_lr 2e-5) | 早期 843→漂回 1925 | — | BC 摸到好解但守不住 |
| SAC+MPC + critic_lr 3e-4 + **BC地板0.1**（=FNO 口径，公平） | 早期~900 → epoch17 漂到 **~1230 且仍升** | 0.5→1.6 | BC 衰减后 SAC 梯度缓慢漂离好解 |
| ~~SAC+MPC + BC地板0.5~~ | ~~~905（稳住）~~ | ~~0.5~~ | 🔴 **作弊废弃**（BC 锚比 FNO 强 5×）|

- ⚠️ **公平口径下 SAC+MPC 会漂**（bc_final 0.1，与 FNO 同衰减）：这**不是要修的 bug，是发现本身**——同样专家锚衰减节奏下，**FNO 扩散骨干稳钉 871，SAC 却守不住、缓慢漂高**。完整 1M 末段窗均值待 §5 run 出（epoch17 才 ~1230，1M 可能更高）。诚实报告漂移值。
- **纯 SAC School 3-seed（诊断期 epoch27 早停存证）**：能耗 **24012±75** / avg_violations 23.4（每区率 94%）/ comfort 10.45°C，3 seed 从 epoch3 稳到 epoch27，std 仅 75 → 失败铁证。参照：MLP 13368/17.1、FNO 6418/7.08。完整 1M 见 §5。

---

## 3. 【公平协议】重跑配置 —— 已对 ground-truth 逐项核实

> 🔴 **防作弊（§0 红线重申）**：BC schedule = 专家(MPC)模仿强度，是 SAC+MPC 与 Guided-FNO **共享的正则项**，必须逐楼对齐 FNO，不许用结果反推拧参数。

**BC schedule 逐楼对齐 FNO（07-16 从三栋 `paper_data/paper_metadata.pkl` 读出的 ground truth，已确认）：**
| 楼 | `--bc-weight` | `--bc-weight-final` | `--bc-weight-decay-steps` | `--violation-penalty` | 依据 |
|---|---|---|---|---|---|
| OfficeSmall | 0.8 | **0.1** | 150000 | 10 | FNO 用脚本默认（metadata 实证）|
| SchoolPrimary | 0.8 | **0.1** | 150000 | 10 | FNO 用脚本默认（metadata 实证）|
| OfficeMedium | 1.0 | **0.6** | 200000 | 12 | FNO 手调值（metadata 实证）|

**唯一允许的算法侧差异 = SAC 标准优化器（真 bug 修复，非作弊）：**
- `--actor-lr 3e-4 --critic-lr 3e-4`（从 2e-5/5e-6 拉到标准 SAC）。理由：SAC 的 critic 是驱动高斯 actor 策略梯度的唯一来源，必须能学；扩散策略 critic 角色不同，2e-5 能工作。给每个算法各自能工作的优化器是标准做法。
- `--update-per-step 1.0`（标准 SAC）。
- `--buffer-size 200000`：标准 SAC 选择，**不再是为省内存**（内存不是约束了，§4），对公平性无影响（算法内部量）。
- `reward_scale` 三栋默认 0.00035（不动，逐楼与 FNO 一致）。

**⚠️ 预期结果 = 会漂移，如实报告，不许拧参数止漂：** OfficeSmall bc_final 0.1 + critic_lr 3e-4 诊断 run 是能耗从早期 ~900 漂到 epoch17 ~1230 且仍升。这个漂移是**诚实的核心发现**：同等 BC 衰减下 SAC 守不住好解、FNO 扩散骨干稳如磐石。纯 SAC 硬崩 + SAC+MPC 随 BC 衰减漂走 = 统一叙事"BEAR 上 SAC 在线策略提升有害，扩散骨干在同等 BC 下稳定"。

### 3.1 启动方式（已封装，避免手敲出错）
- **统一 launcher**：`scripts/_sac_launch.sh <mpc|sac> <small|medium|school> <seed>`——逐楼公平配置全部封死在脚本里，只需传 3 个参数。
  - `mpc` → `rl_baseline_mpc_bcfixclean.py`（SAC+MPC，log-prefix `sacmpc_fair_<楼>_1m_s<seed>`）
  - `sac` → `rl_baseline_bcfixclean.py`（纯 SAC 无专家，log-prefix `sac_pure_<楼>_1m_s<seed>`）
- 环境两坑（launcher 已内置）：**dropt 环境绝对路径 python.exe**（`/c/Users/zouwei/anaconda3/envs/dropt/python.exe`，git-bash `conda activate` 失效）+ **PYTHONIOENCODING=utf-8**（脚本打印 emoji，gbk 管道秒崩）。launcher 还设 `OMP_NUM_THREADS=3`（12 核下留并行余量）。
- 启动后核对：log 里打印的 config（`actor_lr/critic_lr/bc_weight*/violation_penalty` 逐项）+ 出现 `✓ 专家控制器 'mpc' 初始化成功`（mpc 变体，7 个 env 各一次）。

---

## 4. 【实测·推翻上一版】机器资源与并行度 —— 3 个 SAC+MPC 并存安全

> 🔴 **上一版本节（"16GB RAM 空闲仅 3GB / 绝不并发 ≥2 / 只能串行"）是 thrash 状态误判，本版全部推翻。** 教训：resource 判断必须在机器干净时实测，thrash 时 `wmic/nvidia-smi/tasklist` 全超时会误导成"资源枯竭"。

**干净重启后实测（2026-07-16 晚）：**
- **RAM**：总 16.3GB，**空闲 8.0GB**（不是 3GB）。
- **CPU**：12 核。**GPU**：RTX 3070 8GB，空闲时几乎全空。
- **单 SAC+MPC run 开销（实测）**：School ~2.4GB RSS / Small ~1.1GB / Medium ~1.9GB；GPU 每个仅 ~200MB + util ~30%。
- **3 个 SAC+MPC 并存实测**（School+Medium+Small 各 seed42）：**空闲 RAM 剩 3.7GB、GPU util 66%、每个 ~21-25 it/s（含梯度更新）** → **安全有余**。
- **关键机制**：SAC 对 GPU 压力极小（高斯 actor 单前向，无 FNO 的 6 步反扩散采样）→ handoff 阶段三"3 个 FNO 饱和 GPU"的约束**对 SAC 不适用**。真瓶颈是 RAM 和 CPU 线程，不是 GPU 显存。

**并行度结论：**
- **SAC+MPC（重）：稳定 3 并存**（GPU 66% 是主要看点，第 4 个重 run 会接近饱和拖慢但不崩）。
- **纯 SAC（轻，无 MPC 专家）：可 4 并存**（RAM 每个 ~1.3GB、GPU 更低）。
- 启动仍建议**错峰**（隔 ~60s 启一个），避开 torch DLL + BEAR CSV 解析 + buffer 分配的启动内存尖峰叠加。稳态共存没问题。

**单 run 时长（实测外推）**：SAC+MPC School/Medium ~13h、Small ~9h（1M 步，step_per_epoch=8192 → ~122 epoch）；纯 SAC 更快（无 MPC QP 每步开销）。

---

## 5. 重跑计划与进度（18 run）

> **范围说明（对 handoff 原计划的一处修正）**：原计划"SAC+MPC 9 + 纯 SAC 仅补 School"。本轮改为**纯 SAC 也三栋全用标准 3e-4 重跑**（共 18 run）。理由：旧纯 SAC Small/Medium 用的是 2e-5 bug 档，若只补 School 会导致"纯 SAC 行"三栋 critic_lr 不一致、被审稿人抓；纯 SAC 无 MPC 最便宜，重跑代价小。这样纯 SAC 与 SAC+MPC **只差"有无 MPC-BC"**，是干净消融。
> 队列与状态实时维护在 `run_logs/SAC_QUEUE.md`（唯一真相源）。

### 5.1 队列（滚动池维持 3 并存，先清 SAC+MPC 再跑纯 SAC）
- **SAC+MPC 9**：{small, medium, school} × {42, 0, 1}，prefix `sacmpc_fair_*`。
- **纯 SAC 9**：{small, medium, school} × {42, 0, 1}，prefix `sac_pure_*`。
- **07-16 晚已启动并运行中**：`sacmpc_fair_{small,medium,school}_1m_s42`（seed42 三栋，配置已验证正确、MPC 初始化成功、健康推进）。
- 🔴 **【07-16 更新·medium HOLD】** 因 §0 的统一决定，**5 个 medium run(#6,7 SAC+MPC / #13,14,15 纯SAC)已 HOLD**，等 OfficeMedium 默认档探针验证。可跑的实际是 Small/School 的 12 个 + 正在跑的 medium s42(#2，回退保险)。medium 解冻前，滚动池只从 Small/School 补。
- 滚动规则：某个跑完就从队列补下一个；SAC+MPC 保持 ≤3 并存（或 3 重 + 1 纯SAC），纯 SAC 阶段可 4 并存。**跳过 HOLD 的 medium。**

### 5.2 完训判据 & 抽数
- 完训：log 末 `Training finished` 且跑满 ~122 epoch / 1M 步。
- 抽末段 8 点窗均值：复用 `scripts/school_tailmean.py` 抽数模式（改 RUNS 指向 `sacmpc_fair_*` / `sac_pure_*` 目录 + 按 zone 数归一化每区违规率），出 3-seed mean±std，按阶段三 §2.3bis 口径填主表基线行。

### 5.3 主表将呈现的故事（预期，公平口径）
- **纯 SAC**：三栋全塌（诊断已见 Small ~5940 / School 24012），当"纯在线 RL 对照"，正文一句"纯 SAC 无示范锚定时策略提升发散、在最小楼即失效"。
- **SAC+MPC（bc_final 逐楼对齐 FNO）**：critic_lr 修复后不再是荒唐的 6.8×，但**公平 BC 衰减下仍漂高**（OfficeSmall 诊断 epoch17 ~1230，完整 1M 待测）。FNO 仍赢且有机制含义：**同等专家锚衰减下 FNO 扩散骨干稳定、SAC 守不住**。
- 🔴 **红线**：不得为止漂而抬 SAC+MPC 的 BC 地板高于对应 FNO。critic_lr 是唯一允许的算法侧差异。

---

## 6. 本轮文件改动

- `docs/HANDOFF_sac_baseline.md`（本文件，07-16 晚重写）。
- **`scripts/_sac_launch.sh`（本轮新建，正式工具）**：统一 launcher，逐楼公平配置封死。取代上一版有 bug 的 `run_sacmpc_sweep.sh`（第 43 行 echo 语法错 + 并发启动会 OOM 的假设已过时）——**用 `_sac_launch.sh`，别用 sweep.sh**。
- `run_logs/SAC_QUEUE.md`（本轮新建）：18 run 状态表 + 滚动池规则，唯一真相源。
- `log_building/_aborted_0716/`：本轮把 20 个 thrash 期反复 kill/relaunch 产生的残骸目录（`sacmpc_fix_*`/`sacmpc_bcfloor_*`/`sac_fix_*`，event 全 88~76KB 无数据）挪到此处归档，清爽 log_building。
- `scripts/_diag_sac.py`、`scripts/_diag_sac2.py`（诊断临时工具，留作参考）。
- **代码零改动**：`rl_baseline_bcfixclean.py`、`rl_baseline_mpc_bcfixclean.py`、`env/*` 全没动，修复纯靠命令行传参。

---

## 7. 关键文件索引

- SAC 基线脚本：`rl_baseline_bcfixclean.py`（纯 SAC，`expert_type=None`）、`rl_baseline_mpc_bcfixclean.py`（SAC+MPC，`--bc-coef` 默认 True :247、`--expert-type mpc` :252、`--critic-lr` :230、`--update-per-step` :225、`--bc-weight*` :249-251）。base argparse 在 `main_building.py`（BC 默认 :110-114）。
- 默认超参（病根源头）：`env/building_config.py:71` `DEFAULT_CRITIC_LR=2e-5`、:66 `DEFAULT_ACTOR_LR=3e-4`、:15 `DEFAULT_REWARD_SCALE=0.00035`、:48 `DEFAULT_BUFFER_SIZE=1000000`。
- reward 定义：`env/building_env_wrapper.py:288-327`（violation_penalty × 越界区数，再 × reward_scale）。
- FNO 主表 ground-truth 配置源：三栋 run 的 `paper_data/paper_metadata.pkl`（`d['args']` 是 dict）。
- 旧 SAC run（反推/复用）：`log_building/sac_baseline_bcfixclean_*`（旧纯 SAC，2e-5 塌）、`sac_baseline_mpc_bcfixclean_Office*`（旧 SAC+MPC，critic_lr bug，被本轮 `sacmpc_fair_*` 替换）。
- 抽数脚本模板：`scripts/school_tailmean.py`、`scripts/school_fno_vs_mlp.py`。
- 启动/队列：`scripts/_sac_launch.sh`、`run_logs/SAC_QUEUE.md`。
