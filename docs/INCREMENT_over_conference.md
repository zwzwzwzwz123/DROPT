# 期刊版项目相对会议论文（DiffFNO, ICCC 2026-07）的增量清单

> 本文档客观描述当前项目相对已投稿会议论文的增量，含实验数据结果；未完成项标注状态，不含主观评价。
> 生成日期：2026-07-17；末次更新：2026-07-21（客观化审查：剥离主观评价/写作策略/外推建议，只留数据、配置差异、代码事实、状态；§6.3 论文处理段移交 HANDOFF_stage5）。
> 数据源：会议论文数字引自 `docs/DiffFNO_ICCC202607_submission.pdf`（Table II 及正文）；当前项目数字引自 `paper_figures_v2/master_metrics_v2.csv`（3-seed，末段 8 点窗均值），个别在训项直接来自 TensorBoard event。
> 口径说明见文末 §7。

---

## 1. 会议论文基线（已投稿内容）

- **标题**：DiffFNO: Fourier-Structured Diffusion Control for IoT-Enabled Building Energy Management。
- **方法构成**：条件扩散策略 + 动作轴 FNO 去噪器 + 门控残差旁路。**不含推理期梯度引导**（正文未涉及 guidance）。
- **评测范围**：BEAR **OfficeSmall / Hot_Dry** 单一建筑（6 区），控制间隔 1 小时，每 episode 168 步。
- **对比方法**：DiffFNO、DiffFNO w/o Residual、Diffusion-MLP、SAC、SAC+MPC（MPC 作物理参照 + 生成 BC 专家）。3-seed 统计。
- **报告指标**：episode HVAC 能耗、comfort violations、final test reward、mean |Δa|（动作平滑度）；另有 Welch PSD 低频占比。
- **会议论文 Table II（OfficeSmall，3-seed，原文数值）**：

| 方法 | 能耗 (kWh) | 违规 | Test Reward | Mean \|Δa\| |
|---|---|---|---|---|
| DiffFNO | 901.3 ± 19.9 | 0.886 ± 0.135 | -0.738 ± 0.086 | 0.0402 ± 0.0090 |
| DiffFNO w/o Residual | 871.1 ± 6.6 | 1.082 ± 0.338 | -0.844 ± 0.207 | 0.0357 ± 0.0109 |
| Diffusion-MLP | 990.9 ± 38.4 | 1.441 ± 0.510 | -1.106 ± 0.321 | 0.0596 ± 0.0162 |
| SAC | 5980.4 ± 27.3 | 5.483 ± 0.046 | -5.055 ± 0.011 | 0.3430 ± 0.0412 |
| SAC+MPC | 1864.6 ± 74.6 | 3.431 ± 0.294 | -2.600 ± 0.222 | 0.1729 ± 0.0117 |

- **PSD 低频占比（≤2 cycles/day）**：MPC 95.2%，DiffFNO 82.6%，Diffusion-MLP 73.7%。
- **推理成本**：K=6 步反扩散，RTX 3070、batch 1 下约 5.2 ms/步、约 31 ms/决策。
- **残差结论（会议论文原文）**：去残差降低能耗、动作更平滑，但**增加 comfort violations、降低 reward**；据此认为残差对"贴近舒适边界的局部修正"有用。

> 对应关系（事实陈述）：会议论文 "DiffFNO"(901.3/0.886) 无 guidance，对应当前项目的 **NoGuide** 配置(900.4/0.865)；会议 "DiffFNO w/o Residual"(871.1/1.082) 对应当前 **NoRes_NoGuide**(868.2/1.076)。当前项目的 **Full** 配置 = 会议模型 + guidance。

---

## 2. 增量一：从单建筑扩到三建筑（跨规模评测）

会议论文仅 OfficeSmall（6 区）。当前项目扩到三建筑，均 BEAR / Hot_Dry / 3-seed / 1M 步。

**三建筑主表（Full = FNO 骨干 + guidance0.5，vs Diffusion-MLP；3-seed 末段 8 点窗均值 mean±std，std 为样本 ddof=1）：**

| 建筑 | 区数 | 结构 | Guided-DiffFNO 能耗 | Diff-MLP 能耗 | 能耗省 | FNO 每区违规率 | MLP 每区违规率 |
|---|---|---|---|---|---|---|---|
| OfficeSmall | 6 | L1/m4/w48 | 870.8 ± 3.1 | 994.0 ± 34.7 | 12.4% | 8.3% | 23.8% |
| OfficeMedium | 18 | L1/m4/w64 | 6984.6 ± 13.3 | 8644.6 ± 430.1 | 19.2% | 20.5% | 38.0% |
| SchoolPrimary | 25 | L1/m4/w128 | 6417.6 ± 21.8 | 13367.9 ± 1344.8 | 52.0% | 28.3% | 68.4% |

- 能耗省幅（12.4% → 19.2% → 52.0%）**单调递增**（三栋统一默认档协议后）。
- 三建筑结构统一为 L1 / modes4，width 随 state_dim 分化（48/64/128，对应 state_dim 20/56/77）；训练超参三栋统一默认档（无逐楼手调）。
- **OfficeMedium 行说明（2026-07-19 定案）**：采用默认档统一协议（与 Small/School 同，选 a）。此前手调档版本（FNO 7041.5 / MLP 7202.1 / 省 2.2%，vp12 等 8 项单独调参）已作废。统一默认档后：MLP 8644.6（每区率 38.0%），FNO 6984.6（7042→6985），省幅 19.2%。
- 配置口径：默认档 OfficeMedium MLP（8644.6）高于手调档（7202.1）；主表前提为三栋统一协议、无逐楼调参。超参档切换下的数值变化：FNO 7042↔6985，MLP 7202↔8645。

**状态**：三建筑均已完成（3-seed 齐，统一默认档协议）。OfficeMedium 默认档 3-seed 于 2026-07-19 完训替换。数据源 `paper_figures_v2/master_metrics_v2.csv`（由 `scripts/extract_master_metrics.py` 生成，std 以此为准）。

---

## 3. 增量二：推理期 critic 梯度引导（guidance）+ 骨干/引导解耦

会议论文无 guidance。当前项目新增推理期 critic 梯度引导（η=0.5），并做了骨干与引导的分离分析。

**3.1 guidance 增量（Full vs NoGuide，同骨干同超参，只差 guidance 开关）：**

| 建筑 | 区数 | Full 能耗 | NoGuide 能耗 | Full 违规 | NoGuide 违规 | guidance 能耗省 |
|---|---|---|---|---|---|---|
| OfficeSmall | 6 | 870.8 ± 3.1 | 900.4 ± 26.1 | 0.496 | 0.865 | ~3% |
| OfficeMedium | 18 | 6984.6 ± 13.3 | 8048.4 ± 173.7 | 3.687 | 6.703 | 13.2% |
| SchoolPrimary | 25 | 6417.6 ± 21.8 | 8267.6 ± 677.9 | 7.084 | 12.748 | ~22% |

- OfficeSmall：违规 0.865→0.496，能耗 900→871（约 3%）。
- OfficeMedium：能耗 8048→6985（13.2%），违规 6.703→3.687（每区率 37.2%→20.5%，降约 45%）。
- SchoolPrimary：能耗 8268→6418（约 22%），违规 12.75→7.08（约 44%）。
- **guidance 能耗省幅随区数单调递增：Small ~3% → Medium 13.2% → School ~22%**（三点，2026-07-22 补齐 OfficeMedium 后）。违规降幅三栋均大（45%/45%/44% 量级）。

**3.2 骨干独立效应（NoGuide vs MLP，均无 guidance、同超参）：**

| 建筑 | 区数 | NoGuide 能耗 | MLP 能耗 | 骨干省能耗 | NoGuide 每区率 | MLP 每区率 |
|---|---|---|---|---|---|---|
| OfficeSmall | 6 | 900.4 | 994.0 | 9.4% | 14.4% | 23.8% |
| OfficeMedium | 18 | 8048.4 | 8644.6 | 6.9% | 37.2% | 38.0% |
| SchoolPrimary | 25 | 8267.6 | 13367.9 | 38.2% | 42.5% | 68.4% |

- 骨干独立省能耗**非单调**：Small 9.4% → Medium 6.9% → School 38.2%（Medium 骨干纯效应最弱）。
- 成分分解（Full vs MLP 总省幅 = 骨干 + guidance 各自贡献，量级读法）：**Medium 总省 19.2% 中骨干仅约 7%、guidance 约 13%（guidance 主导）；School 总省 52% 中骨干约 38% 为主导**。→ 成分贡献场景依赖，非某一成分跨规模恒定主导。

**状态**：OfficeSmall / OfficeMedium / SchoolPrimary 的 NoGuide 全部完成（3-seed）。**OfficeMedium NoGuide 于 2026-07-22 完训补齐**（s42/s0/s1，默认档，能耗 8048.4±173.7 / 每区率 37.2%）。guidance 增量三点齐、能耗省幅单调；骨干独立效应非单调。guidance-scale 扫描（η=0/0.5/1/2）数据在 event 中，**未聚合成表/图（未完成）**。

---

## 4. 增量三：消融矩阵扩展（残差 / 引导）

会议论文仅 OfficeSmall 的残差消融（且无 guidance 维度）。当前项目在 OfficeSmall 上有完整 4 变体 3-seed：

**OfficeSmall 消融（3-seed 末段 8 点窗均值）：**

| 变体 | 能耗 | 违规 | 每区违规率 | reward |
|---|---|---|---|---|
| Full (FNO+guidance) | 870.8 ± 3.1 | 0.496 ± 0.059 | 8.3% | -0.487 |
| NoGuide | 900.4 ± 26.1 | 0.865 ± 0.096 | 14.4% | -0.726 |
| NoRes (有 guidance) | 867.2 ± 3.1 | 0.597 ± 0.066 | 10.0% | -0.548 |
| NoRes_NoGuide | 868.2 ± 5.7 | 1.076 ± 0.417 | 17.9% | -0.840 |
| MLP | 994.0 ± 34.7 | 1.426 ± 0.58 | 23.8% | -1.097 |

- 残差效应（Full vs NoRes，均有 guidance）：能耗 870.8 vs 867.2（std 重叠）；违规 0.496 vs 0.597（每区率 8.3% vs 10.0%，差 1.7pp）。
- 残差效应（NoGuide vs NoRes_NoGuide）：能耗 900.4 vs 868.2；违规 0.865 vs 1.076（每区率 14.4% vs 17.9%，差 3.5pp）。
- **交互读法见 §6**：残差舒适收益 3.5pp（无引导）→ 1.7pp（有引导），guidance 部分吸收残差纠偏职能。

**状态**：OfficeSmall 4 变体已完成（3-seed）。**OfficeMedium / SchoolPrimary 的 NoRes 消融未完成（缺）**——注：这两栋才是谱截断真正发生的楼（保留 40% / 31%），OfficeSmall 为零截断（modes4=rfft长4，保留 100%）。

---

## 5. 增量四：基线重跑（SAC / SAC+MPC，公平协议）

会议论文的 SAC(5980)、SAC+MPC(1864) 使用 critic_lr=2e-5（诊断认定为偏低配置）。当前项目按公平协议重跑（critic_lr=3e-4 等标准 SAC 优化器；BC 地板逐楼对齐 FNO；buffer 200k，1M steps）。

**3-seed 聚合基线（`scripts/_extract_sac_baselines.py`，末段 8 点窗 mean±std）：**

| 方法 | 楼 | 区数 | 能耗 | 每区违规率 | comfort | n_seed |
|---|---|---|---|---|---|---|
| SAC+MPC | OfficeSmall | 6 | 2493±227 | 70.2% | 3.35 | 3 |
| SAC+MPC | OfficeMedium | 18 | 13754±127 | 73.6% | 3.10 | 3 |
| SAC+MPC | SchoolPrimary | 25 | 21850±407 | 89.9% | 6.88 | 3 |
| 纯 SAC | OfficeSmall | 6 | 5204±170 | 93.7% | 9.08 | 3 |
| 纯 SAC | OfficeMedium | 18 | 17356±209 | 88.0% | 6.76 | 3 |
| 纯 SAC | SchoolPrimary | 25 | 24862±222 | 94.3% | 11.06 | 3 |

**与扩散主表并排（三栋一致排序 Guided-FNO < Diff-MLP < SAC+MPC < 纯 SAC）：**

| 楼 | Guided-FNO | Diff-MLP | SAC+MPC | 纯 SAC |
|---|---|---|---|---|
| OfficeSmall | 871 | 994 | 2493 | 5204 |
| OfficeMedium | 6985 | 8645 | 13754 | 17356 |
| SchoolPrimary | 6418 | 13368 | 21850 | 24862 |

- 三栋能耗排序一致：Guided-FNO < Diff-MLP < SAC+MPC < 纯 SAC。SAC/SAC+MPC 每区违规率 70–95%。公平协议下（critic_lr=3e-4）SAC 能耗仍显著高于扩散类，与会议版 SAC(5980)/SAC+MPC(1864) 方向一致。
- **OfficeMedium SAC+MPC 补跑（2026-07-20）**：旧 s42（能耗 7789 / viol 5.38）为手调档 bc_final=0.6/vp=12，与选 (a) 默认档不一致，已作废归档（`log_building/DEPRECATED_bc0.6_*`）。默认档 s42 完训后，3-seed 默认档 = **13754±127 / 每区率 73.6%**（n=3，上表已用此值）。
- **状态**：SAC/SAC+MPC 18 run + medium mpc s42 补跑全完训，6 组 3-seed 聚合全部出齐（上表，均 n=3）。抽数脚本 `scripts/_extract_sac_baselines.py`（每 run 独立 EventAccumulator、末8窗、按 zone 归一化，同前缀选 eval 点最多的真身）。

### 5.1 OfficeMedium 训练超参统一（2026-07-19 定案：采用默认档统一）

- **动机**：OfficeMedium 此前为唯一手调栋，MLP 基线调优与 Small/School 不一致。将 OfficeMedium 8 项训练超参统一到 Small/School 默认档（bc_final 0.6→0.1、vp 12→10 等）。
- **默认档 3-seed 结果（`scripts/_extract_medium_default.py`，末段 8 点窗，std=样本 ddof=1）**：
  - FNO 默认档：6984.6±13.3 / 每区率 20.5% / comfort 0.636。
  - MLP 默认档：8644.6±430.1 / 每区率 38.0% / comfort 1.141。
  - 对比手调档：FNO 7041.5 / 18.5%；MLP 7202.1 / 20.6%。
- **结果**：统一默认档后 FNO 7042→6985，MLP 7202→8645（每区率 20.6%→38.0%）；OfficeMedium 省幅 2.2%→19.2%，三栋省幅曲线由非单调变为单调递增（12.4%→19.2%→52.0%）。
- **定案：采用默认档统一（选 a）。** 主表 OfficeMedium 行已替换（§2）。前提：三栋统一协议、无逐楼调参（默认档 OffMed MLP 高于手调档 7202）。超参档切换下 FNO 变化小（7042↔6985）、MLP 变化大（7202↔8645）。
- **状态：完成。** 5 个 medium run（FNO s0/s1+probe_s42、MLP s42/s0/s1）全部 245ep 完训。medium SAC 基线用默认档（vp10）重跑中（rolling 接管）。

---

## 6. 增量五：guidance×residual 2×2 消融 + checkpoint 权重核查

会议论文对残差的消融只有一维（OfficeSmall 去残差：违约 0.886→1.082、reward −0.738→−0.844，无 guidance 维度），并将 residual 描述为主方法的 essential 组成。当前项目新增两项分析：① OfficeSmall 残差×引导 2×2 消融；② 三建筑训练后 checkpoint 的残差权重核查。

### 6.1 OfficeSmall 2×2 消融（残差 × 引导，每区违规率）

| | 有残差 | 无残差（NoRes） | 残差的舒适收益 |
|---|---|---|---|
| **无引导（Guidance-Off，= 会议版设定）** | 14.4% (NoGuide) | 17.9% (NoRes_NoGuide) | **3.5 pp** |
| **有引导（Guidance-On）** | 8.3% (Full) | 10.0% (NoRes) | **1.7 pp** |

- Guidance-Off 一行（14.4% vs 17.9%，换算每步违约 0.865 vs 1.076）与会议论文数字一致。
- 残差的每区违规收益：无引导时 3.5 pp（14.4% vs 17.9%），有引导时 1.7 pp（8.3% vs 10.0%）。

### 6.2 checkpoint 权重核查（只读，load 完全匹配）

对训练后的三建筑 Full checkpoint 做直接核查：

| 建筑 | residual_gate(sigmoid) | 残差权重均绝对值 | ‖gate·res‖/‖out‖ |
|---|---|---|---|
| OfficeSmall | 0.502 | 0.0105 | 5.74% |
| OfficeMedium | 0.508 | 0.0096 | 4.42% |
| SchoolPrimary | 0.515 | 0.0080 | 4.92% |

- 残差权重较 `nn.Linear` 初始化缩小约 10–20×，对角 > 非对角（近似恒等）；残差输出对最终去噪输出的量级贡献约 4.4–5.7%（真实量级输入上测得）。
- 代码事实（`diffusion/model_fno.py`）：残差 `residual(x)=nn.Linear(action_dim, action_dim)` 作用于含噪动作 x（不含 state），是 FNO 主路里唯一的全频段跨区通路；FNO 沿分区（动作）轴做 rfft、只保留低频；每层并联 pointwise conv（`nn.Conv1d(width, width, kernel_size=1)`：跨通道全连接、沿分区轴不跨区）。门控 `residual_gate` 为单个可学习标量（初始 raw=0.5，sigmoid≈0.622），非逐区/逐通道门控网络。
- 数据关系：§6.1 的 2×2 消融显示残差每区违规收益在有引导下为 1.7pp（无引导 3.5pp）；§6.2 checkpoint 显示三栋残差量级贡献 4.4–5.7%、gate sigmoid 约 0.50–0.52。

> 论文处理/写作策略（residual 措辞、Experiments 小节结构、与会议版和解段）见 `docs/HANDOFF_stage5.md` §2「诚实成分刻画」，本文档不含写作策略，只留数据与代码事实。
> ⚠️ 门控 g 数据现状：`residual_gate` 未写入 TensorBoard，且跨 checkpoint 值本身近乎恒定（sigmoid 全程 0.50–0.58、不随 guidance 系统变化），无"衰减轨迹"可讲。§6.2 表中 residual_gate(sigmoid) 为训练后终值；残差抑制的机制是残差权重训练后近零，非门控关闭。

**状态**：三建筑 checkpoint 核查已完成（只读，未改模型）；OfficeSmall 2×2 消融已完成（3-seed）。Medium/School 的 NoRes 训练消融未做（缺）——现有残差数据来自 OfficeSmall 2×2 消融 + 三栋权重核查，无 Medium/School 直接消融。

---

## 7. 口径与未完成项汇总

**口径说明：**
- 违规指标 `avg_violations` = 每时刻超 ±1°C 容差的区数对时间平均（逐区计数，被区数放大），跨楼比较用"每区违规率"= avg_violations / 区数。
- 会议论文 Table II 与当前 CSV 的绝对值不可直接混比：会议 "DiffFNO"=当前 NoGuide（无 guidance），当前 "Full"=会议模型 + guidance。会议 OfficeSmall MLP 990.9、当前 MLP 994.0（seed 池/窗口口径微差）。
- 所有当前数字为 3-seed 末段 8 点窗均值（除标注单 seed 的探针）。
- **std 口径（2026-07-20 统一）**：全项目一律**样本标准差 ddof=1（除 n-1）**。`extract_master_metrics.py` 已改 `np.std(v, ddof=1)`、CSV 已重生成（旧 ddof=0 版备份 `master_metrics_v2.csv.ddof0_bak`）；`_extract_medium_default.py`/`_extract_sac_baselines.py` 本就是 ddof=1。**旧文档中的 ddof=0 std（如 Small MLP ±28.3、School MLP ±1098）均已作废**，换算关系 std_ddof1 = std_ddof0 × √(3/2)=1.2247（n=3）。均值不变。

**未完成项清单（截至 2026-07-20）：**
| 项 | 状态 |
|---|---|
| OfficeMedium 默认档统一 FNO+MLP 3-seed | ✅ 完成（07-19 定案，主表已替换，见 §2/§5.1） |
| SAC / SAC+MPC 公平协议 3-seed 聚合（18 run + medium mpc s42 补跑） | ✅ 完成（07-20，见 §5）。19 run 全完训、6 组 3-seed 聚合全部出齐（均 n=3） |
| OfficeMedium NoGuide（guidance 增量第三点，默认档 3-seed） | 🔄 进行中（s42 07-20 启动在跑；s0/s1 待 RAM 腾出错峰补） |
| OfficeMedium / SchoolPrimary NoRes 消融 | 未开始（缺） |
| 大楼 modes 扫描 m2/m4/m8（谱截断机制） | 未开始 |
| Conv1d 去噪器对照 | 未开始（需改代码） |
| guidance-scale 扫描 η=0/0.5/1/2 聚合成图 | 数据在 event，未聚合 |
| diffusion_steps 扫描 2/4/6/8/10 | 未开始 |
| 分区轴空间谱补 seed（axis mismatch） | 单 seed 单楼，未补 |

**投稿靶**：会议论文投 IEEE ICCC；期刊靶未定（讨论中：建筑能源类 Energy and Buildings / Applied Energy vs IEEE IoT-J）。

**关联文档**：`docs/HANDOFF_stage5.md`（**当前状态干净快照**）、`docs/HANDOFF_stage4_alignment.md`（阶段四推导史档案）、`docs/HANDOFF_option3_bear_journal.md`（三建筑推导史）、`paper_figures_v2/master_metrics_v2.csv`（主表数据源）。

**已生成论文图**（`paper_figures_v2/`，默认档新数据，2026-07-19）：主管线 `gen_paper_figures_v2.py` 出 fig1-9（含 fig2 单调省幅曲线）；追加 `gen_extra_figures.py` 出 figA（训练奖励曲线 FNO vs MLP×三楼）、figB（OfficeSmall 消融 2×2 热力图）。待补：SAC 基线对比、modes 扫描、消融热力图扩三楼。


