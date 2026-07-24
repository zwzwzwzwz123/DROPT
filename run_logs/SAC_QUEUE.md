# SAC 基线重跑队列 (阶段四 §6.1 第2项)

> 目标：替换被 critic_lr bug 拖坏的旧 SAC/SAC+MPC 数据，出 3-seed 公平主表基线。
> 公平协议：BC schedule 逐楼对齐 FNO（防作弊红线），SAC 侧统一标准优化器 3e-4/3e-4/update1.0/buffer200k。
> 启动：`bash scripts/_sac_launch.sh <mpc|sac> <small|medium|school> <seed>`（配置已封死在脚本内）。
> 并行度：实测 3 个 SAC+MPC 并存 = RAM 剩 3.7GB / GPU 66%，安全。纯 SAC 更轻可 4 个。**绝非 handoff 说的"只能串行"。**
> 单 run 时长：School/Medium SAC+MPC ~13h，Small ~9h，纯 SAC 更快（无 MPC QP）。

## 状态表 (18 run)

> 🕘 **最后核对：2026-07-20**（event 抽数实测）。**18 run 全完训**（n_eval=123）。3-seed 聚合已出（见下节）。唯一遗留：medium mpc s42 旧值是手调档 bc0.6 作废，已补跑默认档 s42。

| # | 类型 | 楼 | seed | 状态 | log-prefix |
|---|---|---|---|---|---|
| 1 | mpc | small | 42 | ✅ DONE | sacmpc_fair_small_1m_s42 |
| 2 | mpc | medium | 42 | ✅ DONE (默认档补跑, 旧 bc0.6 已作废归档) | sacmpc_fair_medium_1m_s42 |
| 3 | mpc | school | 42 | ✅ DONE | sacmpc_fair_school_1m_s42 |
| 4 | mpc | small | 0 | ✅ DONE | sacmpc_fair_small_1m_s0 |
| 5 | mpc | small | 1 | ✅ DONE | sacmpc_fair_small_1m_s1 |
| 6 | mpc | medium | 0 | ✅ DONE (默认档) | sacmpc_fair_medium_1m_s0 |
| 7 | mpc | medium | 1 | ✅ DONE (默认档) | sacmpc_fair_medium_1m_s1 |
| 8 | mpc | school | 0 | ✅ DONE | sacmpc_fair_school_1m_s0 |
| 9 | mpc | school | 1 | ✅ DONE | sacmpc_fair_school_1m_s1 |
| 10 | sac | small | 42 | ✅ DONE | sac_pure_small_1m_s42 |
| 11 | sac | small | 0 | ✅ DONE | sac_pure_small_1m_s0 |
| 12 | sac | small | 1 | ✅ DONE | sac_pure_small_1m_s1 |
| 13 | sac | medium | 42 | ✅ DONE (默认档) | sac_pure_medium_1m_s42 |
| 14 | sac | medium | 0 | ✅ DONE (默认档) | sac_pure_medium_1m_s0 |
| 15 | sac | medium | 1 | ✅ DONE (默认档) | sac_pure_medium_1m_s1 |
| 16 | sac | school | 42 | ✅ DONE | sac_pure_school_1m_s42 |
| 17 | sac | school | 0 | ✅ DONE | sac_pure_school_1m_s0 |
| 18 | sac | school | 1 | ✅ DONE | sac_pure_school_1m_s1 |

## ✅ 3-seed 聚合基线 (2026-07-20, `scripts/_extract_sac_baselines.py`, 末8窗 mean±std)

| 方法 | 楼 | 能耗 | 每区违规率 | comfort | n_seed |
|---|---|---|---|---|---|
| SAC+MPC | small | 2493±227 | 70.2% | 3.35 | 3 |
| SAC+MPC | medium | **13754±127**(默认档3seed) | 73.6% | 3.10 | 3 |
| SAC+MPC | school | 21850±407 | 89.9% | 6.88 | 3 |
| 纯SAC | small | 5204±170 | 93.7% | 9.08 | 3 |
| 纯SAC | medium | 17356±209 | 88.0% | 6.76 | 3 |
| 纯SAC | school | 24862±222 | 94.3% | 11.06 | 3 |

- **排序三栋一致**：Guided-FNO < Diff-MLP < SAC+MPC < 纯SAC。RL 基线每区率 70-95%（守不住舒适），坐实 BEAR 上 RL 失效（非 critic_lr bug）。
- ✅ **medium mpc s42 补跑默认档完训（2026-07-20）**：旧值 7789/5.38 = 手调档 bc0.6/vp12（选 a 后作废），已归档 `log_building/DEPRECATED_bc0.6_sacmpc_fair_medium_1m_s42_*` + `run_logs/DEPRECATED_bc0.6_*.log`。干净 3-seed 默认档 = **13754±127 / 每区率 73.6%**（上表已替换）。

## ✅ OfficeMedium 默认档统一已定案(选 a)，洼地消除 (2026-07-19)
- **medium 默认档 3-seed 全完训**（FNO s0/s1+probe_s42、MLP s42/s0/s1，均 245ep）。真实末8窗（`scripts/_extract_medium_default.py`）：**FNO 6985±13 / 每区率 20.5%；MLP 8645±430 / 每区率 38.0%**。
- **结论**：统一默认档后 FNO 几乎不变(7042→6985)、MLP 大幅变差(7202→8645)→ 省幅 2.2%→**19.2%**，三栋 12.4%→19.2%→52.0% 单调、洼地消除。**用户 07-19 定选 (a) 默认档统一。**
- **launcher medium 行=默认档**（BCW0.8/BCF0.1/BCD150000/VP10），选 (a) 定案保留不回退。
- **rolling 已于 07-19 05:54 重启**（含 medium 队列），接管 medium SAC(默认档) + school sac。
- **回退保险 #2**（medium mpc s42 bc0.6, 7789/5.38）：选 (a) 后作废，不进主表。
- **medium SAC 用默认档 vp10**（对齐默认档 FNO），与新主表口径一致。

## ⚠️ RAM 并行约束 (2026-07-17 实测)
- 16.3GB 机器，medium FNO/MLP 均带默认 1M buffer（~2.5GB/进程），SAC 仅 200k buffer。
- **安全上限 ≈ 6 个重进程**。当前 6 个（3 SAC + 2 FNO + 1 MLP）= 4.0GB 空闲，健康。
- **一次性起 8 进程（+3 MLP）触发 thrash**（RAM 掉到 652MB），已杀 3 MLP 回收 6GB。教训：medium 主表 5 run 必须错峰，等 SAC 完训腾槽再补 MLP s0/s1。

## 🟢 无人值守滚动脚本（2026-07-16 启动，PID 26968）
- **`scripts/_sac_rolling.sh`** 已 nohup 启动、**脱离 Claude 会话**（孤儿进程，关会话不影响）。日志 `run_logs/_rolling.log`。
- 职责：维持 SAC 并行 ≤3，某个完训(`Training finished`)就从 Small/School 队列补下一个。队列 10 个 = SAC+MPC{small,school}×{0,1} + 纯SAC{small,school}×{42,0,1}。**medium 全部不在队列（HOLD）**。
- 不碰探针（`main_building_fno_guided`，不匹配 `rl_baseline`）、不重复启动已有 log 的 run。
- **停止**：`touch run_logs/_rolling.STOP`（下一轮检查见到即优雅退出，**不杀在跑的 run**）。
- medium 解冻后（探针验证成立），需手动把 medium 项加进脚本 QUEUE 数组 + 改 launcher medium 行，再重启脚本。

## 滚动池规则
- 保持 ≤3 个 SAC 并存（含正在跑的 seed42；3 SAC + 探针已实测稳定）。某个完训就补下一个。
- 先清 SAC+MPC(#4,5,8,9；#6,7 HOLD 跳过)，再跑纯 SAC(#10-12,16-18；#13-15 HOLD 跳过)。
- 完训判据：log 末尾 `Training finished` 且跑满 245 epoch（1M/8192≈122... 实际 total_steps=1M / step_per_epoch=8192 = 122 epoch）。

## 完训后
- 用 scripts/school_tailmean.py 模式抽末段 8 点窗均值（改 RUNS 指向 sacmpc_fair_* / sac_pure_*，按 zone 归一化每区违规率）。
- 填主表基线行，替换旧 critic_lr-bug 数据。
- ⚠️ **口径纪律（07-17 核对修正）**：状态表里 #1-3 的能耗/违规是**末8窗均值**（已从早先误填的末段单点 2665/7701/21472 更正为 2725/7789/21456）。SAC+MPC 在末段仍漂移（§5.3，medium 末10点 7627~8036 跳动），**单点不可用**（§1.4 血泪教训）。主表一律走末8窗 3-seed 抽数，状态表这几个值仅供参考。
