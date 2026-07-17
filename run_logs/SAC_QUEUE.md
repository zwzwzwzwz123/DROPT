# SAC 基线重跑队列 (阶段四 §6.1 第2项)

> 目标：替换被 critic_lr bug 拖坏的旧 SAC/SAC+MPC 数据，出 3-seed 公平主表基线。
> 公平协议：BC schedule 逐楼对齐 FNO（防作弊红线），SAC 侧统一标准优化器 3e-4/3e-4/update1.0/buffer200k。
> 启动：`bash scripts/_sac_launch.sh <mpc|sac> <small|medium|school> <seed>`（配置已封死在脚本内）。
> 并行度：实测 3 个 SAC+MPC 并存 = RAM 剩 3.7GB / GPU 66%，安全。纯 SAC 更轻可 4 个。**绝非 handoff 说的"只能串行"。**
> 单 run 时长：School/Medium SAC+MPC ~13h，Small ~9h，纯 SAC 更快（无 MPC QP）。

## 状态表 (18 run)

> 🕘 **最后核对：2026-07-17 上午**（wmic/log 实测）。SAC+MPC 已完 #1-5（small 三 seed 全齐 + school/medium s42），#8/#9 在跑，之后进纯 SAC。

| # | 类型 | 楼 | seed | 状态 | log-prefix |
|---|---|---|---|---|---|
| 1 | mpc | small | 42 | ✅ DONE (末8窗 energy 2725/viol 4.26) | sacmpc_fair_small_1m_s42 |
| 2 | mpc | medium | 42 | ✅ DONE (末8窗 energy 7789/viol 5.38) | sacmpc_fair_medium_1m_s42 |
| 3 | mpc | school | 42 | ✅ DONE (末8窗 energy 21456/viol 22.34) | sacmpc_fair_school_1m_s42 |
| 4 | mpc | small | 0 | ✅ DONE | sacmpc_fair_small_1m_s0 |
| 5 | mpc | small | 1 | ✅ DONE | sacmpc_fair_small_1m_s1 |
| 6 | mpc | medium | 0 | 🔴 HOLD | sacmpc_fair_medium_1m_s0 |
| 7 | mpc | medium | 1 | 🔴 HOLD | sacmpc_fair_medium_1m_s1 |
| 8 | mpc | school | 0 | RUNNING (~ep100/123) | sacmpc_fair_school_1m_s0 |
| 9 | mpc | school | 1 | RUNNING (~ep8/123) | sacmpc_fair_school_1m_s1 |
| 10 | sac | small | 42 | queued | sac_pure_small_1m_s42 |
| 11 | sac | small | 0 | queued | sac_pure_small_1m_s0 |
| 12 | sac | small | 1 | queued | sac_pure_small_1m_s1 |
| 13 | sac | medium | 42 | 🔴 HOLD | sac_pure_medium_1m_s42 |
| 14 | sac | medium | 0 | 🔴 HOLD | sac_pure_medium_1m_s0 |
| 15 | sac | medium | 1 | 🔴 HOLD | sac_pure_medium_1m_s1 |
| 16 | sac | school | 42 | queued | sac_pure_school_1m_s42 |
| 17 | sac | school | 0 | queued | sac_pure_school_1m_s0 |
| 18 | sac | school | 1 | queued | sac_pure_school_1m_s1 |

## 🔴 OfficeMedium 全部 HOLD (2026-07-16)
- **原因**：用户决定把 OfficeMedium 训练超参从手调档拉回 Small/School 默认档（bc_final 0.6→0.1、decay 200000→150000、vp 12→10 等 8 项，见 memory [[guided-difffno-officemedium-config-unify]]）。
- launcher 第 21 行 medium 现在锁的是**旧手调 FNO 对齐值**（BCF=0.6/BCD=200000/VP=12），已失效。
- **vp 12→10 影响 reward 口径**，故不止 SAC+MPC，纯 SAC(#13-15) 也 HOLD（COMMON 行给所有类型传 vp）。
- **解冻条件**：OfficeMedium FNO@默认档探针（officemedium_fno_default_probe_s42）验证成立后，改 launcher medium 行 = 默认档，再跑 #6,7,13,14,15。
- 正在跑的 #2（medium mpc s42，PID 22292，bc0.6）**不动**，留作回退保险；探针崩→它是 0.6 成品，探针成→它作废（只损失这一个）。
- Small/School（#1,3,4,5,8,9,10,11,12,16,17,18）**不受影响**，FNO 本就默认档、对齐目标没变，照常跑。

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
