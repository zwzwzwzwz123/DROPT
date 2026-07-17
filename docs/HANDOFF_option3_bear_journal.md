# HANDOFF —— 选 3 定案：留在 BEAR，扩成多建筑期刊版

> 日期：2026-07-08　接续自 `HANDOFF_journal_direction.md`(07-06) 与 `HANDOFF_sinergym_direction.md`(07-07)
> 目标（用户原话）：不追顶刊，只要内容自洽、故事完整、能发表即可（IEEE IoT-J 档）。
> 语言：中文。本文件记录 07-08 这轮的**方向定案 + 新核实事实**，不重复前两份已述内容。

---

## 0. 一句话定案

Sinergym(选 1) 与跨 zone 规模(B) **均已否**（见 §1）。**选定"选 3"**：留在 BEAR，保留会议论文的"FNO 在空间轴"叙事，把单建筑扩成三建筑 + FNO 机制诚实化，写一篇诚实的应用向期刊。**唯一需要新算力的事：SchoolPrimary 补训到 1M 步。** 其余全是盘活磁盘已有数据。

---

## 1. 本轮排除的两条路（有实测，别再走回头路）

### 1.1 Sinergym 跨 zone 规模(原 B 方向) —— 被环境规格否决
- 实测 sinergym 109 个环境，datacenter/5zone 的**动作是 1~2 个全局设定点，不是逐区**：
  - smalldatacenter(1区) / datacenter_dx(4区)：动作都是 `Box(1)`=cooling_setpoint。
  - datacenter_cw(4区)：`Box(2)`=cooling_setpoint + chws_supply_temp（第2维是供水温，**异质物理量，不是多一个区**）。
  - 5zone(6区)：`Box(2)`=heating+cooling setpoint。
- 结论：BEAR 里 `action_dim == zone数`（逐区设定点），所以"FNO 沿动作轴"=="沿空间轴"，改建筑=改动作维，跨规模天然成立。**sinergym 不成立**——动作跟 zone 数无关，长度 1~2 的轴上 FFT 无意义。obs 里 zone 轴也只有 1~2 个温度暴露，谱截断比 OfficeSmall 还退化。**"共享 FNO 骨干跨 zone 数"在标准 sinergym 无处落脚。**

### 1.2 时间轴 FNO(选 1 的挽救方案) —— 用户主动否
- 曾提出把 FNO 作用轴从空间搬到时间(跨控制频率 10min↔15min)。用户否，理由正确：**这会推翻会议论文"FNO 在空间轴"的核心叙事，等于自拆地基。**

---

## 2. 本轮新核实的关键事实（都读了源码/args.pkl/event，非推测）

### 2.1 超参真实设定（此前用户已记不清，现已挖出）
- **diffusion_steps = 6，全建筑硬编码统一**：`main_building_fno_guided_bcfix_clean.py:257` `args.diffusion_steps = 6`（注释"覆盖默认运行超参以加速实验"），**非 argparse 参数，无分支**。3 个有 args.pkl 的 SchoolPrimary run 均为 6，印证。
- ⚠️ **【07-08 晚更正】modes/width/layers 三栋楼其实不统一**——早先"modes=4 全建筑统一"的说法**错了**。从 checkpoint 权重形状（ground truth）反推的真实架构见 §9.2。简记：OfficeSmall=w48/L1/**modes4**、OfficeMedium=w64/L2/**modes6**、SchoolPrimary(旧164k)=w128/L1/modes4。
- ⚠️ **脚本默认值 ≠ 实跑值**：argparse 默认是 `fno-modes=4 / fno-width=48 / fno-layers=1`，但实跑各栋不同（手动传参）。**不能拿默认值反推没存 args 的 run，必须从 checkpoint 权重形状反推。** 主表 guided/mlp run 大多**没存 args.pkl**，但 §9.1 发现 `paper_data/paper_metadata.pkl` 里存了完整 args（含 seed/guidance_scale），这是复现配置的可靠来源。

### 2.2 谱截断真实发生情况（支柱 2 的证据核心）
zone 数为**精确值**（state_dim=3×zone+2 反推，探针日志确认），rfft 长度=⌊zone/2⌋+1。⚠️ **各栋 modes 不同**（§9.2），须用各自真实 modes 算截断：

| 建筑 | zone/动作维 | rfft 长度 | 真实 modes | 是否真截断 | 保留比例 |
|---|---|---|---|---|---|
| OfficeSmall | 6 | 4 | 4 | **否**（保留全部）| 100% |
| OfficeMedium | 18 | 10 | **6** | **是**（砍 4 个高频）| 60% |
| SchoolPrimary | 25 | 13 | 4 | **是**（砍 9 个）| 31% |

⚠️ **modes 本身不统一(4/6/4)是混淆因素**：截断"保留比例"随楼递减(100%→60%→31%)本对支柱 2 有利，但这个趋势混了"zone 变多"和"modes 手调不同"两个因素，须靠 §9.5 的统一 m2/m4/m8 扫描解耦。新 School 1M run 用 modes4（保留比例 31%）。

### 2.3 三建筑真实盘面（从 TensorBoard event 抽，末段 tail_mean）
Guided-DiffFNO vs Diff-MLP，能耗 / 违规：

> 🔴 **【07-13 复盘更正】此表 SchoolPrimary 行的旧 164k 数已作废，权威数据见 §12.1 / §13.4（1M/3-seed）。下表保留仅为历史记录，勿用于论文。当前权威三建筑主表 = §2.3bis。**

| 建筑 | 预算 | Guided-DiffFNO | Diff-MLP | 能耗省 | 违规降 |
|---|---|---|---|---|---|
| OfficeSmall | 1M | 877 / 0.088 | 1106 / 0.266 | 21% | 3× |
| OfficeMedium | 1M | 6978 / 3.31 | 7169 / 4.08 | **仅3%** | 19% |
| SchoolPrimary | ~~164k~~ | ~~~6241 / ~10.3~~ | ~~11138 / 15.0~~ | ~~44%~~ | ~~31%~~ |

（OfficeMedium 另有 SAC+MPC=7150/5.68，也输 FNO。SchoolPrimary 的 mixture_reg run 是 n=1 崩溃早停，忽略。）

### 2.3bis 【07-14 权威三建筑主表 —— 全部 3-seed 齐口径】（能耗 kWh / avg_violations 计数 / 每区违规率；均末段 8 点窗均值 mean±std）
| 建筑 | 区数 | 预算/seed | Guided-DiffFNO | Diff-MLP | 能耗省 | 违规降 |
|---|---|---|---|---|---|---|
| OfficeSmall | 6 | 1M / 3-seed | 871±3 / 0.50 / 8.3% | 994±28 / 1.43 / 23.8% | 12% | 65% |
| OfficeMedium | 18 | 1M / 3-seed | 7016±22 / 3.37 / 18.7% | 7202±26 / 3.71 / 20.6% | **2.6%** | 9% |
| SchoolPrimary | 25 | 1M / 3-seed | 6418±18 / 7.08 / 28.3% | 13368±1098 / 17.11 / 68.4% | **52%** | 59% |

✅ **【07-14】三建筑首次全部 3-seed 齐口径（1M/guidance0.5）**，OfficeMedium 补齐（§14.4），std 极小（FNO±22/MLP±26）。
⚠️ **OfficeMedium 洼地被 3-seed 坐实（省幅 2.6%，非噪声）**：能耗省幅 12%→2.6%→52% 确认非单调，"结构先验随区数单调增强"叙事排除，走"耦合结构调制"退路（§13.4/第5节）。
⚠️ **OfficeSmall 违规用 3-seed 计数口径（Comfort Violations，Full=0.50），非单 seed 主表的 Violation Rate 0.088（§10.5 口径说明，禁混）。**

---

## 3. 两个必须处理的诚实问题（藏不住，审稿人必问）

1. ~~**预算不齐**：SchoolPrimary 只训 164k 步~~ **【07-13 已解决】** School 1M/3-seed 完训（§11/§12/§13），三建筑预算已齐（OfficeMedium 也是 1M）。剩余缺口只是 OfficeMedium 的 seed 数（单→3）。
2. **OfficeMedium 洼地** **【07-14 更新：3-seed 彻底坐实】**：三建筑全 3-seed 齐口径后，能耗省幅 12%→**2.6%**(Medium,±22 std)→**52%**(School) 非单调，洼地**是真实现象非噪声非未收敛假象**（§14.4）。"结构先验随区数增强"单调曲线不成立。候选成因：耦合强度差异（OfficeMedium 耦合最强 10.65，§10.4）/ BC 专家质量。退路见 §13.4 / 第 5 节支柱 2 叙事降级为"结构先验受耦合结构调制"。**这是选 3 里最需要动脑、非抽数据能解决的点。** 　🔴 **【07-16 后续】** 用户决定把 OfficeMedium 训练超参统一到默认档以消除"MLP 调优不一致"混淆变量（洼地/省幅曲线的干净度问题），最新进展见 `HANDOFF_stage4_alignment.md` §9 + memory `guided-difffno-officemedium-config-unify`。

---

## 4. 方法决策：modes 与 diffusion_steps 都不改（结论 + 理由）

- **diffusion_steps 保持 6，不随环境改**：BEAR 线性、动作分布简单，6 步已把违规压到 0.088 / action MSE 0.001；加步数只拖慢采样（per-step 6 步反扩散是推理瓶颈），换不来精度；每栋调不同步数会砸掉跨建筑可比性。
- ~~**modes 保持 4，全建筑统一**：统一配置跨三建筑都稳，是选 3 的资产而非负债。~~ 🔴 **【07-08 晚/07-14 复盘更正】此说法错误**：实测三栋 modes 是 **4/6/4 不统一**（§2.1/§9.2 checkpoint ground truth）。"全建筑 modes=4 统一"是早期误判，已被推翻。modes 不统一是 §9.5 的遗留清理项，也是支柱 2 谱截断对比的混淆因素（须靠统一 m2/m4/m8 扫描解耦）。
- **若要碰，碰成消融不是调参**：
  - `diffusion_steps ∈ {2,4,6,8,10}` 扫描（推理期，不重训，便宜）→ 论证"6 是性价比拐点"，进附录。
  - `modes` 的 m2/m4/m8 扫描（大建筑上重画）是**支柱 2 主线证据**，不是附录。

---

## 5. 文章三根支柱（选 3 叙事）

- **支柱 1｜单→三建筑**：解决会议版"只测一栋"的最大送审风险。OfficeMedium/SchoolPrimary 提进主表。
- **支柱 2｜FNO 机制诚实化**：会议版声称"谱截断=平滑先验"，但 OfficeSmall 零截断（§2.2）。**把它变研究问题**："结构先验何时真起作用？"随区数/耦合增强。m2/m4/m8 在大建筑重画坐实。⚠️ 受 §3.2 洼地拖累，须先解释清楚。
- **支柱 3｜修 axis mismatch**（会议版软肋）：FNO 滤**分区轴**，但平滑性证据(mean|Δa|、PSD)测**时间轴**，因果链没接上。须补分区轴空间谱证据，或把时间平滑归因讲清（残差门控+BC+guidance）。**不修则保留会议叙事的意义打折。**

---

## 6. 落地顺序（按"先盘活、后补算力、最后写"）

1. ✅ **补 SchoolPrimary 到 1M 步 —— 已于 07-08 启动，见 §9.4。** 注意：**不是**照旧 164k run 续配置，而是用**主表协议重跑**（w128/L1/modes4/**guidance0.5**/**seed 42,0,1**）。此前此处写的"width=64/layers=2/seed=0"是从 mixture run 误推的，已被 §9.2（checkpoint 反推）+ §9.1（paper_metadata 实证 seed=42/guidance0.5）推翻，勿用。
2. **聚合已有散数据成论文级表/图**：η 扫描、m2/m4/m8、三建筑主表——数值在 event 里，抽成 CSV+图。低成本高回报。
3. **补支柱 2/3 机制实验**：大建筑重画谱截断扫描 + 补分区轴空间谱证据。
4. **补 3 seed(0/1/42)** 坐实主结论（前两份 handoff 血泪教训：单 seed 只筛信号）。
5. **写作**：guidance 降级为"已验证组件"（QGPO/QGF 撞车，别当核心新机制）、统一残差叙事（NoRes 近 Pareto）、补显著性。

---

## 7. 开放问题（下一轮待查）

1. ~~SchoolPrimary 当时确切训练命令~~ **【已解决】** 从 `run_logs/val_fno_w*_school_s0.log` + paper_metadata 知：旧 run 是 guidance=0/seed0/164k。已不重要——§9.4 改用主表协议(guidance0.5/seed42/1M)重跑，不续旧配置。
2. **OfficeMedium 洼地成因（§3.2）** ——补到 1M 后 SchoolPrimary 数字是否变，可能消解或坐实洼地。**仍开放，最需动脑。**
3. ~~OfficeMedium/SchoolPrimary 确切 zone 数~~ **【已解决】** state_dim=3×zone+2 反推：OfficeSmall=6、OfficeMedium=**18**、SchoolPrimary=**25**（精确值，探针日志印证）。
4. ~~guided 主表 run 无 args.pkl 复现风险~~ **【已缓解】** `paper_data/paper_metadata.pkl` 存了完整 args（含 seed/guidance_scale/架构）。后续新 run 仍应主动存 args。
5. **modes 跨楼不统一(4/6/4)对支柱 2 的影响（§9.5）**：谱截断对比被 modes 差异混淆，须在统一 modes 扫描协议下重做才干净。

---

## 8. 关键文件索引（本轮新增/确认）

- 训练脚本：`main_building_fno_guided_bcfix_clean.py`（modes 定义 :171、diffusion_steps 硬编码 :257）。
- FNO 去噪器：`diffusion/model_fno.py`（SpectralConv1d :16、写死维度处 state_mlp:80 / residual:113）。
- 三建筑数据：`log_building/*OfficeSmall*`、`*OfficeMedium*Hot_Dry*100万步`、`*SchoolPrimary*Hot_Dry*`。
- 主表/消融 CSV：`log_building/table_1m_metrics.csv`、`ablation_summary.csv`、`fno_modes_2_vs_4_psd_summary.json`。
- 有 args.pkl 的参考 run（反推配置用）：`log_building/diffusion_fno_mixture_*SchoolPrimary*/args.pkl`。

---

## 9. 【07-08 晚追加】width/guidance/seed 核实 + SchoolPrimary 1M 正式 run 已启动

这一节是落地顺序 §6.1 的执行记录，推翻/修正了前文几处，并把 §6.1 的"补 SchoolPrimary 到 1M"真正跑了起来。

### 9.1 guidance 训练期到底开没开 —— 已从代码 + paper_metadata.pkl 坐实
- **机制**：guidance 在 `diffusion/diffusion.py:178-182` 的 `p_mean_variance` 注入，被 `p_sample` 反向采样调用。训练期 collector 采样与 test 走同一条路 → **只要 `guidance_scale>0`，guidance 就参与训练期动作生成，不只是事后画图。**
- **会议版旗舰确实训练期开了 guidance**：读 `log_building/*OfficeSmall*__guided_seed0/paper_data/paper_metadata.pkl` → guided 变体 `guidance_scale=0.5`，noguide 变体 `=0.0`。
- **主表口径统一到 guidance=0.5**：读 `fno_guided——100万步/paper_data/paper_metadata.pkl`(OfficeSmall 主表) 与 OfficeMedium 主表 → 均 `guidance_scale=0.5`。
- ⚠️ **注意区分**：`guidance_scale`(训练+eval 都用) vs `paper_guidance_scale=5.0`(只在事后画对比图用)。mixture 那几个 run 是 `guidance_scale=0.0 + paper_guidance_scale=5.0`（训练不开、只事后评估开），不是主表口径。

### 9.2 三栋楼主表真实架构（从 checkpoint 权重形状反推，ground truth）
| 建筑 | seed | width | layers | modes | 动作维(zone) | state_dim | FNO actor 参数 | MLP actor 参数 | MLP/FNO |
|---|---|---|---|---|---|---|---|---|---|
| OfficeSmall | 42 | 48 | 1 | 4 | 6 | 20 | 30,876 | 210,998 | **6.83×** |
| OfficeMedium | 42 | 64 | 2 | 6 | 18 | 56 | 125,384 | 226,370 | 1.81× |
| SchoolPrimary(旧164k) | 0 | 128 | 1 | 4 | 25 | 77 | 211,260 | 235,337 | **1.11×** |

- **基线(MLP/SAC)宽度三栋楼固定 256**（hidden_dim），只有输入/输出维随环境变。critic 三栋都是 256 宽 MLP，公平。
- ⚠️ **"FNO 更小还更好"的优势随建筑收窄**：OfficeSmall 6.8× → School 仅 1.1×（因 width 从 48 涨到 128，参数追上 MLP）。这是 width 规律要解决的核心问题。

### 9.3 width 规律 = **width ≥ state_dim**（已用实验+机制坐实）
- **实验**（`run_logs/val_fno_w48/w128_school_s0.log`，seed0/guidance0/164k，末段8点窗均值±std）：
  - w48: 能耗 6241±195 / 违规 **10.31±1.28** / reward -6.81±0.78 / FNO 34k
  - w128: 能耗 6350±85 / 违规 **8.60±0.67** / reward -5.78±0.41 / FNO 211k
  - → **w48 在 School 违规明显掉点**（差 1.7，近 std 和边缘），"固定 width=48 通吃"假设被否。
- **机制**：`state_mlp` 首层 `Linear(state_dim, width)`。School state_dim=77 挤进 width=48 = **条件编码信息瓶颈**。width≥state_dim 才无瓶颈。
- **规律自洽性**：三栋楼原始 width(48/64/128) 对 state_dim(20/56/77) **全部满足 ≥**——用户当初的选择隐含遵循这条，只是没显式讲。规则表述："**width ≥ state_dim，且取验证饱和的最小值，且 FNO 参数 < MLP**"（下界约束+最小化，非精确函数）。
- ⚠️ caveat：该实验是 guidance=0 / 单 seed / 164k。guidance=0.5 下 w48 违劣势可能被补一部分（guidance 直接压违规）。精确幅度待正式 run。
- 🔴 **【07-13 复盘更正】此处 w128/164k 数（能耗 6350/违规 8.60）已被更可靠数据取代**：本轮 w128/guidance0/**1M/3-seed** NoGuide 正式 run（§13.4）为能耗 8268±553/违规 12.75，与此处 164k 单 seed 的 8.60 差距大（长训 + guidance0 训练不稳，NoGuide std=553）。且此处引用的 `val_fno_w48/w128_school` run 目录在 log_building 已不存在（仅剩 run_logs stdout），本身有"撞目录/checkpoint 互覆盖"警告（§1.3 / §2.3 注 c）。**width≥state_dim 规律的机制论证仍成立，但具体数值以 §13.4 为准。**

### 9.4 SchoolPrimary 1M 正式 run —— **已于 07-08 16:xx 启动，3 seed 并行**
- **配置**（对齐主表协议）：`--building-type SchoolPrimary --weather-type Hot_Dry --fno-width 128 --fno-layers 1 --fno-modes 4 --guidance-scale 0.5`，1M 步(245 epoch)，**seed ∈ {42, 0, 1}**。
- **seed=42 是主行**（对齐 OfficeSmall/OfficeMedium 主表的 seed=42），0/1 凑齐 3-seed 协议。
- **log-prefix**：`school_guided_1m_s42 / _s0 / _s1`（独立，防撞目录）。stdout：`run_logs/school_guided_1m_s{42,0,1}.log`。
- **资源**：RTX 3070 8GB，单 run 仅 ~600 MiB 显存，三个并行共 ~1.9GB，安全。
- **速度**：6:45/epoch → 1M ≈ **27.5h**（三个并行同时完成，约 07-09 20:00 前后）。
- **完成后要做**：抽三 seed 末段窗均值，替换 §2.3 表里 SchoolPrimary 那行的 164k 旧数（很可能同时消解或坐实 §3.2 的 OfficeMedium 洼地）。

### 9.5 L/modes 跨楼不统一 —— 遗留清理项（属支柱 2，不在本轮）
- L: 1/2/1，modes: 4/6/4，都非单调、明显手调。**width/L 是容量旋钮应固定或按规则，modes 是被研究的机制应按统一扫描协议(m2/m4/m8 选验证最优)定**，讲成"实验选择"而非"手调"。本轮 School run 沿用 L1/modes4 未动，留待支柱 2 统一处理。

---

## 10. 【07-08 更晚追加】离线分析产出 + 违规口径 + School MLP baseline 待跑

> 这一节全部在**不占 GPU、不动 School run** 的前提下做（纯读磁盘 / 实例化 env 一次）。结论**按证据强度分级**：确证级=有 3-seed std 或源码支撑；待验证级=单 seed / 20% 进度 / 曾被自我推翻。**待验证级严禁写成论文结论。**
> ⚠️ 教训：本轮我一度只跑一次单栋数据就下"会议版机制错了"的重结论，随后自查用方差分解推翻，收敛成 §10.3 的谨慎版。后续沿用"下重结论前先自查能否翻车"。

### 10.1 School run 进度核对（截至 07-08 22:12）
- 三 seed(42/0/1)齐步 **Epoch ~50/245（约 20%）**，无掉队/发散/崩溃。~7.1min/epoch，ETA 修正为 **约 07-09 21:00**（略晚于 §9.4 的 20:00 估计）。
- 20% 处 test 指标（**远未收敛，仅趋势，不可当终值、不可跨楼比**）：能耗 s42/s0/s1 ≈ 6483/6343/6374，违规 ≈ 8.05/8.67/7.10，comfort_mean ≈ 0.80，reward ≈ -5.4。**三 seed 散布很小** → 跑完的 3-seed 聚合大概率干净。
- 主体已从起点(能耗~16000/违规~16)降到平台(~6360/~7.7)，最近 5 点在平台震荡，剩 80% 可能再降一点。
- seed42 stdout 在刷 `Critic梯度过大…裁剪到20.0`，是 `build_guidance_fn` 梯度裁剪正常兜底（School 25 区 Q 梯度大），非故障。

### 10.2 违规指标口径 —— 【确证级，源码坐实，直接影响论文好不好看】
- **`avg_violations` = 每时刻"超 ±1°C 容差的区数"再对时间平均**（`env/building_env_wrapper.py:415` `comfort_violation=int((abs_delta>tol).sum())`，:485-486 除以步数）。是**逐区计数，天然被 zone 数放大**，不是比率。
- 后果：School 7.7(25 区)vs OfficeSmall 0.5(6 区)"差 15 倍"是**假象**。归一化成**每区违规率** = 31% vs 8%（真实差距，区数差只解释一部分）。
- ⚠️ 诚实提醒：**归一化后 31% 仍不算好看**，且 School comfort_mean 0.80°C > OfficeSmall 0.44°C，确实贴近 ±1°C 边界，是真差距不是错觉。
- **另一指标 comfort_mean（平均每区温度偏差,°C）= `abs_delta.mean()`**，是"整体达标度"，不被区数放大。School 0.80°C 仍在 ±1°C 容差带内。
- **处理办法（用户已认可前三条）**：① 跨楼一律归一化成每区违规率/违规时间占比；② 论文主指标改用 comfort_mean(°C)，violations 降为辅助（诚实且不误导——避免把"整体达标、局部偶尔超线"描述成"大面积违规"）；③ 主打相对优势（同楼 FNO vs MLP）。**红线：不得事后放宽 ±1°C 容差凑数**（除非引 ASHRAE 依据且对所有基线一致重算）。

### 10.3 分区轴空间谱（脚本 A，支柱 3）—— 【混合：一条确证 + 一条待验证】
- 脚本：`scripts/paper_spatial_spectrum.py`；产出 `paperfigure_spatial_spectrum/`。源码已核实 FFT 沿 action=zone 轴(`model_fno.py:37`)，zone 按几何文件房间序编号、connectmap 是一般图非 1D 链 → **"空间频率"是索引顺序上的先验，非真实物理相邻**（诚实 caveat，必须写）。
- **【确证级】** 方差分解（OfficeMedium，不减均值）：动作 **84~94% 是共模**（各区一起开关）；FNO 相比 MLP 主要把**跨区空间差异方差压小约 3×**(0.0019 vs 0.0060)，但**总动作、总功率与 MLP 相近**（总方差仅小 14%）。即 FNO 更"平滑"主要体现在那占比小的空间差异分量上。
- **【待验证级，勿写成结论】** 归一化频谱形状 FNO 并不比 MLP 更低通(低模占比 0.28 vs 0.25)，且有 Nyquist 尖峰。据此**曾**推断"会议版'截断→低通空间动作'因果链错了"——但：(a) 是减均值后在~1%残余里看形状，(b) 仅 1 seed/1 楼，(c) 尖峰可能是 OfficeMedium 房间"CORE/PERIMETER/PLENUM 循环编号"的真实类型交替，(d) "残差旁路放行高频"纯推测、未做关残差消融验证。**故仅记为"疑点"**：会议版机制解释可能不准确，真实机制更可能是"截断=权重空间正则(压幅度) + 残差/BC/guidance 共同致时间平滑"，**需多 seed/多楼/关残差消融才能定。**

### 10.4 耦合结构 / 洼地假说（脚本 B，支柱 2）—— 【待验证级，是框架不是结论】
- 脚本：`scripts/paper_coupling_structure.py`；产出 `paperfigure_coupling/`。三栋楼各 reset 读 `env.bear_env.A_d`/`connectmap` 一次算得（A_d 只依赖几何+time_resolution，与天气/location 无关，三楼可比）。
- 指标（越界数据点见 CSV）：offdiag_coupling_ratio Small/Med/School = 4.67/**10.65**/4.64；bandedness(A_d，低=带状=索引-FFT先验对齐) = 1.14/0.78/**0.55**。
- 🔴 **【07-14 复盘更正】`coupling_structure.csv` 的 modes/modes_retained/truncation_frac 三列对 OfficeMedium 错误**：脚本 `paper_coupling_structure.py:39` 硬编码 `MODES=4`，但 OfficeMedium 实际 **modes=6**（§9.2 ground truth）→ CSV 把 OffMed 记成 modes4/截断0.6，实际应 modes6/截断0.4（保留 6/10）。**截断口径以 §2.2 表为准**（Small/Med/School 保留 100%/60%/31%）。**耦合指标(offdiag/bandedness/tau)不依赖 modes，仍正确。** energy_saving_pct 列(21/3.0/空)也过时，以 §2.3bis(12%/2.6%/52%)为准。**要拿此 CSV 画支柱 2 图须先修 MODES 或忽略这几列。**
- **假说**：FNO 的 1D 低通先验只有当"强耦合在索引顺序上带状"时才对齐；School 最带状(0.55)预测它比 OfficeMedium(0.78)省得多 → 待明晚 School 干净数字检验。
- ⚠️ **两个不能忽略的坑**：(a) OfficeSmall 最散(1.14)却最省(21%)，但它**零有效截断**(modes4=rfft长4)，是**对照组不是数据点**，naive"越带状越省"律在它身上失效；(b) OfficeMedium 同时是**耦合最强(10.65)**，洼地可能是"最难控制问题"而非"先验错位"——**3 个点解不开这个混淆，不能断言带状度是主因。**

### 10.5 OfficeSmall 3-seed 重新分析 —— 【确证级，改写主表口径】
读 `paperfigure_bcfixclean_smalloffice_multiseed/compare_energy_violations_mapping.csv`（3-seed，均值±std）：
- ⚠️ **两表违规口径不同，禁混**：单 seed 主表 `table_1m_metrics.csv` 列名 **Violation Rate**(Full=0.088，比率)；3-seed 聚合列名 **Comfort Violations**(Full=0.500，计数)。差 5.7× 是口径不是矛盾。
- **单 seed 主表数字不可用**：MLP 能耗单 seed 1106 vs 3-seed 990.9±38.4（差>3σ，单 seed 那次是偏差样本）。**论文一律用 3-seed 聚合，弃 `table_1m_metrics.csv` 绝对值。**
- **确证结论**（能耗 kWh，Comfort Violations 计数）：
  - FNO 真赢 MLP：871.5±3.2 vs 990.9±38.4（省 12%，超 std）；违规 0.500 vs 1.441。**核心卖点，稳。**
  - **guidance 贡献只在压违规不在省能耗**：Full vs NoGuide 违规 0.500±0.030 vs 0.886±0.135（降~44%），能耗几乎无差(871 vs 901)。
  - **残差非 essential（打脸 ICCC 稿）**：Full/NoRes/NoRes&NoGuide 能耗 871.5/867.8/871.1 **std 重叠、统计分不出**；NoRes 违规 0.614 亦接近 Full 0.500。**期刊版残差叙事必须改口，不能称 Full 能耗最优。**
  - SAC 基线极差(5980±27，6.8×能耗)，审稿人必问，须重调或明确解释。
- OfficeMedium 仍是**单 seed**（每变体 n=1）：FNO 6987 vs MLP 7170 仅省 2.5%（=洼地），**无 std，幅度可能是噪声，成因不可下结论**；坐实需补 OfficeMedium 多 seed（不在当前计划）。
- 附带：`run_logs/constrained_eval_officemedium*` 有投影约束实验残留（violations 512→13 但能耗 7011→7077 涨），印证 §5 "投影降违规但涨能耗且撞 MPC"，可作附录"为何不走投影"证据，不必重跑。

### 10.6 School MLP baseline —— 【确证级，命令已核准，待 GPU 空闲跑】
- **动机**：化解 §10.2"违规偏高"最有力的武器 = 证明同楼 FNO 违规 << MLP。**但 School 现无 MLP baseline 在跑，此对比暂缺，是当前最大缺口。**
- **资源判断**：显存够（空闲 4.1GB，MLP 单 run ~600MiB 不 OOM），但**瓶颈是算力**（现 GPU 利用率 46%，加第四个会拖慢三个主 run）。**决定：等三 guided run 明晚跑完再跑，不现在挤。**
- **命令已核准**（写入 `run_school_mlp_baseline.sh`）：`python main_building_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry --seed 42 --log-prefix school_mlp_1m_s42`（s0/s1 补误差棒）。
- **核准依据**：逐项核对运行中 School-guided 的 `paper_metadata.pkl`（ground truth），其超参=脚本默认(OfficeSmall 档)，故 MLP 只需覆盖 `--building-type` 即自动对齐 11 项超参。⚠️ **关键**：不可照抄 OfficeMedium MLP 配置——它是**单独手调**的(actor_lr 5e-5/batch 512/critic_lr 5e-6/violation_penalty 12/update 0.25)，与 School-guided 不符，照抄会导致对比不公平。

### 10.7 本轮新增文件索引
- 分析脚本：`scripts/paper_spatial_spectrum.py`(A)、`scripts/paper_coupling_structure.py`(B)、`scripts/_peek_school_events.py`(抽 School event 标量的临时工具)。
- 产出：`paperfigure_spatial_spectrum/`、`paperfigure_coupling/`。
- 待跑脚本：`run_school_mlp_baseline.sh`（含核准依据注释）。
- School run 目录：`log_building/school_guided_1m_s{42,0,1}_SchoolPrimary_Hot_Dry_20260708_*`。

---

## 11. 【07-10 追加】School guided 1M 完训 + 3-seed 干净数据 + MLP baseline 三 seed 已启动

> 本轮：核对 §9.4 启动的三 guided run 是否收尾 → 抽 3-seed 末段窗均值（论文口径）→ 启动 §10.6 待跑的 MLP baseline。**GPU 相关操作全部实测，非推测。**

### 11.1 三 guided run 全部干净完训（§9.4 收尾）
- 三 seed(42/0/1) 均跑满 **Epoch #245 / train_step 1,003,520**，stdout 结 `Training finished`，无 traceback/发散。日志末次写入 07-09 21:10~21:21，对上 §9.4 的 ETA。
- best_reward：s42=-4.470(#233)、s0=-4.370(#118)、s1=-4.461(#4)。⚠️ **best 出现的 epoch 无意义**（s1 在 #4），是单点最优；论文口径一律用末段窗均值（§1.4），见 §11.2。

### 11.2 School 3-seed 末段窗均值（窗=8 点）—— 【确证级，替换 §2.3 SchoolPrimary 旧 164k 行】
抽数脚本：`scripts/school_tailmean.py`（本轮新增，`--window` 可调；每区违规率按 zone=25 归一化）。三 seed 各自末段 8 点窗均值，再取 seed 间 mean±std：

| 指标 | 3-seed mean ± std | 备注 |
|---|---|---|
| energy (kWh) | **6418 ± 18** | std 极小 |
| avg_violations (计数) | **7.08 ± 0.22** | → **每区违规率 28.3%** |
| comfort_mean (°C) | **0.760 ± 0.015** | 仍在 ±1°C 容差带内 |
| reward | -4.89 ± 0.14 | |

- **对比 §2.3 旧 164k 行**(能耗~6241/违规~10.3，那是 w48/guidance0/欠训)：补到 **1M+w128+guidance0.5** 后，违规 10.3→7.08（每区率 31%→28%），comfort_mean 0.80→0.76°C。能耗 6241→6418 基本持平（旧数是欠训 w48，不可比）。
- **三 seed std 极小**（能耗±18、违规±0.22）→ 这行现在是**干净的论文级数据**，印证 §10.1 "3-seed 聚合大概率干净" 的预判。**§2.3 主表 SchoolPrimary 那行应替换为本表数值。**
- ⚠️ **OfficeMedium 洼地(§3.2)判定仍待 MLP baseline**：旧 44% 省幅是对旧 MLP(11138) 算的，不可比。School 补到 1M 后的**真实 FNO-vs-MLP 相对省幅**要等 §11.3 的 MLP baseline 出末段数才能算——届时才能判洼地消解还是坐实。**这是本轮未闭环、下一轮第一件事。**

### 11.3 School MLP baseline —— 三 seed 已启动（§10.6 落地）
- **状态**：三 seed(42/0/1) **已于 07-10 14:xx 启动并行训练**（§10.6 "等 guided 跑完再跑" 的时机已到，guided 已于 07-09 晚完训腾出算力）。
- **配置**：`main_building_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry --seed {42,0,1} --log-prefix school_mlp_1m_s{42,0,1}`。algorithm 已核实=`diffusion_mlp_bcfix_clean`，只覆盖 building-type，其余取默认自动对齐 guided（§10.6 核准依据）。
- **速度**：~25 it/s（比 guided 的 9.8 step/s 快，因无 6 步反扩散采样）→ 1M 步 ETA **明显早于 guided 的 27h**，约当晚~次日完成。stdout：`run_logs/school_mlp_1m_s{42,0,1}.log`。
- ⚠️ **两个本轮踩到的环境坑（重启必看）**：
  1. **必须用 conda `dropt` 环境**：base 环境(anaconda3 根)也有能 import 的 torch，但 run 要在 dropt 下。本机 `conda activate` 在 git-bash 里因 cygwin 路径转换失效，**直接用绝对路径** `/c/Users/zouwei/anaconda3/envs/dropt/python.exe` 调用。dropt 已核实 torch2.7.1+cu118/tianshou0.5.1/CUDA可用。
  2. **必须设 `PYTHONIOENCODING=utf-8`**：脚本启动打印含 `⚠️` emoji，经 `tee` 管道时 stdout 退回 Windows gbk 编码 → `UnicodeEncodeError` 秒崩。前两次启动就死在这。正确启动命令：
     `PYTHONIOENCODING=utf-8 /c/Users/zouwei/anaconda3/envs/dropt/python.exe main_building_bcfix_clean.py ...`
- **GPU 实测**：三 guided 跑完后空闲 5GB。三 MLP 并行时 used 3.9GB / free 4.2GB / util 42%，**显存与算力均有余**（§10.6 曾担心"加第四个拖慢"，现三 guided 已退出故无冲突）。

### 11.4 本轮新增文件
- `scripts/school_tailmean.py`：School 3-seed 末段窗均值抽取（论文口径，替代 §10.7 的 `_peek_school_events.py` 临时工具）。
- MLP baseline run 目录：`log_building/school_mlp_1m_s{42,0,1}_SchoolPrimary_Hot_Dry_20260710_*`（训练中）。
- stdout：`run_logs/school_mlp_1m_s{42,0,1}.log`。

### 11.5 下一轮待办（承接）
1. **MLP baseline 完训后抽 3-seed 末段窗均值**（复用 `school_tailmean.py`，改 RUNS 指向 mlp 目录），算 School 的**真实 FNO-vs-MLP 相对省幅/降违规**。
2. **据此判定 OfficeMedium 洼地(§3.2)**：三建筑同为 1M+多 seed+同口径后，"结构先验随区数增强"曲线是否干净，洼地消解还是坐实。
3. 用 §11.2 数值**替换 §2.3 主表 SchoolPrimary 行**，三建筑主表首次全部对齐(1M/guidance0.5/多 seed)。

---

## 12. 【07-10 更晚】School MLP baseline 完训（s42 确证 + s0/s1 收尾中）—— 主故事强化、洼地坐实

### 12.1 完训数据 —— 【确证级，3-seed 末段 8 点窗均值】
School MLP 三 seed(42/0/1) 全部跑满 245 epoch / 1M 步，`Training finished`，无进程残留。抽 3-seed 末段窗均值 vs guided（`scripts/school_fno_vs_mlp.py`）：

| 指标 | MLP 3-seed | guided(FNO) 3-seed | FNO 相对优势 |
|---|---|---|---|
| energy (kWh) | **13368 ± 1098** | 6418 ± 18 | **省 52%** |
| avg_violations | **17.11 ± 0.68** | 7.08 ± 0.22 | **降 59%** |
| comfort_mean (°C) | 3.06 ± 0.55 | 0.76 ± 0.01 | 好 75% |
| reward | -12.03 ± 0.60 | -4.89 ± 0.14 | — |
| **每区违规率** | **68.4%** | **28.3%** | |

- ⚠️ **口径公平性已核实（关键，防审稿人质疑）**：MLP 末段窗均值远高于其全程 min(s42: 11844 vs min 6462)，一度疑似末段发散→用末段窗均值不公。**实查 s42 后半程轨迹排除**：后 50% 能耗 mean 11039 / std 1064 / min 8902 / max 13964，末段 20 点全在 11000~14000 稳定震荡，无一接近 6462。**min 是训练早期偶发低点，末段窗均值才是 MLP 真实稳态**，口径成立、不冤枉 MLP。（对照 §1.4：反向验证——不仅"不能用 min"，还确认"末段窗均值不是被尖峰放大"。）
- **三 seed 对比极干净**：FNO std 极小(能耗±18/违规±0.22)，MLP 一致地差(三 seed 均在 12000~14000 高位，energy std 1098 是高位震荡非偶发)。
- **连训练稳定性都是 FNO 赢**：guided 平稳收敛，MLP 高位剧烈震荡。
- **决定性证据**：School 上 **MLP 每区违规率 68%（2/3 区违规、彻底失控）vs FNO 28%**——这是化解 §10.2"School 违规偏高"的最强武器：不是 FNO 差，是这楼极难、MLP 崩了，FNO 是唯一压得住的。

### 12.2 主故事：大好，超出预期（3-seed 确证）
- School 上 FNO **省 52% 能耗 / 降 59% 违规**（3-seed，std 极小），**比 OfficeSmall(省 12%) 优势大得多**。"这楼本来就难(§10.2 违规偏高)，MLP 崩得厉害(每区违规率 68%)、FNO 是唯一压得住的(28%)"——3-seed 硬数据支撑。**支柱 1 + 化解违规偏高，锁死。**

### 12.3 机制故事：OfficeMedium 洼地被坐实（支柱 2 风险兑现）
> 📌 **【07-14 更新】此节 OfficeMedium 当时是单 seed，现已补 3-seed 坐实（省幅 2.6%±22，§14.4）。最终省幅数以 §2.3bis / §14.4 为准，下表 2.5% 是当时单 seed 值。**

三栋同口径能耗省幅（⚠️ 下表 School/Small 为 3-seed，OfficeMedium 为当时单 seed）：

| 建筑 | 区数 | 谱截断 | FNO 相对 MLP 省幅 |
|---|---|---|---|
| OfficeSmall | 6 | 零截断(modes4=rfft4) | 12% |
| OfficeMedium | 18 | 砍 4(modes6) | **~2.5%** ← 洼地 |
| SchoolPrimary | 25 | 砍 9(modes4) | **~52%** |

- **"截断越狠优势越大"单调叙事彻底不成立**：中间那栋反而最小。§3.2 洼地不但没消解，被 MLP 的差坐实。
- **退路（沿用 §10.4 判断，不致命）**：叙事从"随区数单调增强"改为**"结构先验普遍有效，但幅度受耦合结构调制"**——OfficeMedium 耦合最强(offdiag 10.65)、是最难控问题，用它解释洼地，比强画单调曲线诚实。
- ⚠️ **前提：OfficeMedium 必须补 3-seed**。它现在 2.5% 是单 seed，可能是噪声，无 std。要认真讲支柱 2 并解释洼地，**OfficeMedium 补 3-seed 从"可选"升为"几乎必须"**。主故事(支柱1)已不需要它，但机制故事(支柱2)需要。

### 12.4 状态与待办
- ✅ **School MLP baseline 三 seed 全部完训**，3-seed 对比已出（§12.1），数据干净。§2.3 主表 SchoolPrimary 行可用 §11.2(FNO) + §12.1(MLP) 数值替换。
- **下一轮核心决策点**：**是否为救支柱 2 补 OfficeMedium 3-seed**（FNO+MLP 各 3 seed）。现状：
  - 主故事(支柱1)**已不需要**它——OfficeSmall+School 两栋 3-seed 已够撑"FNO 稳赢 MLP"。
  - 机制故事(支柱2)**需要**它——洼地(2.5%)现为单 seed 无 std，可能是噪声；要讲"结构先验受耦合调制"必须先坐实洼地真实存在。
  - 若补：约 6 个 run（3 seed × FNO/MLP），OfficeMedium ~1M 步机时。可参考本轮并行经验（3070 显存足、util 有余）。
- 本轮新增文件：`scripts/school_fno_vs_mlp.py`（3-seed FNO-vs-MLP 对比）。

---

## 13. 【07-10 最晚】发现主对比混了 guidance 变量 → School FNO-NoGuide 三 seed 已启动

### 13.1 关键发现：§12 的 FNO-vs-MLP 对比不是纯骨干对比
- **核实**（读 run 配置，非推测）：guided FNO 三栋都 `guidance_scale=0.5`；**MLP baseline 三栋都无 guidance**——`main_building_bcfix_clean.py` 里**根本没有 guidance 代码**（grep 无任何 guidance/build_guidance），`diffusion_mlp_bcfix_clean` 算法结构上不支持。MLP 的 `paper_guidance_scale=5.0` 只是事后画图用（§9.1），训练期不参与。
- **后果**：§12 的 School 省 52% / 降 59% 混了**两个变量**——骨干(FNO vs MLP) + guidance(0.5 vs 0)。
- **口径本身不算错**：这是论文一贯的"Guided-DiffFNO 完整系统 vs Diff-MLP 基线"系统级对比，且三栋 MLP 口径一致(都无 guidance)，横向可比。**但它是系统级对比，不是"FNO 骨干 > MLP 骨干"的纯净因果证据。**
- **能耗结论仍稳**：§10.5 OfficeSmall 3-seed 消融已证 guidance 主要压违规、几乎不省能耗(Full vs NoGuide 能耗 871 vs 901 无差、违规降44%)。故 School 52% 能耗省幅主要来自 FNO 骨干；但 59% 违规降幅里 guidance 有实质贡献，不能全记骨干。

### 13.2 School 无现成 FNO-NoGuide 数据（已查证）
- 现有 School FNO run 全不合用：早期 guided run(0705/0706)是 guided 算法且欠训(33/40 epoch，那个 6497 就是旧164k)；mixture 变体骨干是双路非纯 FNO，唯一有 args 的 guidance=0 但只跑 1 epoch。
- → 要纯净解耦骨干，**必须新跑 FNO-NoGuide**。

### 13.3 School FNO-NoGuide 三 seed 已启动（07-10）
- **命令**（与 guided 主 run 逐项核对，只差 guidance 一个变量）：
  `main_building_fno_guided_bcfix_clean.py --building-type SchoolPrimary --weather-type Hot_Dry --fno-width 128 --fno-layers 1 --fno-modes 4 --guidance-scale 0.0 --seed {42,0,1} --log-prefix school_fno_noguide_1m_s{42,0,1}`
- ⚠️ **`--fno-width 128` 必须显式传**：脚本默认 48(会退回 w48 违规掉点坑，§9.3)。其余超参(actor_lr/critic_lr/violation_penalty/total_steps=1M/epoch=245/diffusion_steps=6…)由 building-type + 硬编码(:256-267)自动对齐，已逐项核实与 guided s42 一致。
- **启动同 §11.3 两坑**：dropt 环境绝对路径 python.exe + PYTHONIOENCODING=utf-8。
- **状态**：三 seed 并行 Epoch #1，GPU used 3.9GB/free 4.2GB/util 88%(guided 算法含 6 步反扩散采样，比 MLP 吃算力)。ETA 参考 guided ~27h(比 MLP baseline 慢)。log: `run_logs/school_fno_noguide_1m_s{42,0,1}.log`。
- **完训后出 School 三方对比**：FNO-NoGuide vs MLP = 纯骨干效应；FNO-Full vs FNO-NoGuide = guidance 增量(顺带补 School guidance 消融，呼应 §10.5 OfficeSmall)。可复用 `scripts/school_fno_vs_mlp.py` 改指向。
- **意义**：加固支柱 1 因果归因——把"52% 归 FNO 骨干"从 OfficeSmall 外推变成 School 直接证据，堵审稿人"到底 FNO 还是 guidance"必问。优先级高于补 OfficeMedium 3-seed(那个只服务风险已兑现、有退路的支柱 2)。

### 13.4 【07-13 完训】School 三方解耦结果 —— 【确证级，3-seed 末段窗均值】
三 seed(42/0/1) 全部跑满 245 epoch / 1M 步完训。三方对比（末段 8 点窗均值 mean±std）：

| 变体 | 能耗(kWh) | 违规 | 每区率 | comfort°C |
|---|---|---|---|---|
| FNO-Full (g0.5) | 6418±18 | 7.08±0.22 | 28.3% | 0.76±0.01 |
| FNO-NoGuide | 8268±553 | 12.75±0.20 | 51.0% | 1.56±0.23 |
| MLP | 13368±1098 | 17.11±0.68 | 68.4% | 3.06±0.55 |

**解耦**：
- **纯骨干效应(NoGuide vs MLP)**：能耗省 **38.2%** / 违规降 25.5%。**✅ 达成主目的**：FNO 骨干独立于 guidance 就碾压 MLP，是大信号(std 小)。支柱 1 因果归因坐实——"52% 不是全靠 guidance"，有 School 直接证据。
- **guidance 增量(Full vs NoGuide)**：能耗省 **22.4%** / 违规降 44.4%。

⚠️ **意外发现(与 §10.5 冲突，重要)**：**guidance 在 School 上实质省能耗 22%**，而 §10.5 OfficeSmall 结论是"guidance 几乎不省能耗(871 vs 901)"。上一轮预测"NoGuide 未收敛、会降到接近 Full"被**证伪**——NoGuide 完训停在 8268，没靠近 6418。
- **不是错误，是真实跨建筑差异**：guidance 价值随建筑规模/难度增强(小楼只压违规、大楼兼省能耗)。反而给 guidance 这个"拥挤赛道组件"(撞 QGPO/QGF，见 HANDOFF_journal_direction §4)找回论文价值。
- **机制线索**：NoGuide 能耗 std=553(Full 仅 18)，无 guidance 时 School 训练不稳。guidance 在大楼可能主要作用是"稳定收敛到更好解"，非纯推理期能量引导。**开放问题。**
- ⚠️ **写作红线**：论文"guidance 几乎不省能耗"这句(基于 OfficeSmall)**不能外推**，必须按楼分别陈述。
- comfort 也是干净阶梯 0.76/1.56/3.06：**NoGuide 的 1.56 已超出 ±1°C 容差带(违规率51%)，Full 0.76 在带内** → guidance 对把 comfort 拉回容差带关键。

---

## 14. 【07-13 复盘】文档核查结论 + OfficeMedium 补 3-seed 的精确配置（补实验前必读）

### 14.1 复盘核查结论（本轮拿 event/paper_metadata 逐项复核）
- ✅ **§11/§12/§13 本轮数据全部 event 复核无误**：School guided 6418±18/7.08、MLP 13368±1098/17.11、NoGuide 8268±553/12.75（=三 seed 7787/9043/7973 均值）、OfficeMedium FNO 6988/3.20、MLP 7191/3.88。
- 🔴 **修正 3 处 append-only 遗留的过时/误导点**：§2.3 SchoolPrimary 旧 164k 行已划删+加 §2.3bis 权威主表；§3.1/§3.2 已解决项标注（洼地坐实非假象）；§9.3 的 w128/164k 数（8.60）标注被 §13.4 的 1M/3-seed（12.75）取代、且 val run 目录已不存在。

### 14.2 ⚠️ OfficeMedium 补 3-seed 是坑，不能照 School 方式补
- **OfficeMedium 主表 run 用整套独立手调超参**（从 `.../paper_data/paper_metadata.pkl` 读出，ground truth），**与 OfficeSmall/School 的脚本默认档完全不同**：
  - 架构：`--fno-width 64 --fno-layers 2 --fno-modes 6`（School 是 w128/L1/m4）
  - 训练：`--actor-lr 5e-5 --critic-lr 5e-6 --batch-size 512 --violation-penalty 12 --update-per-step 0.25 --bc-weight 1.0 --bc-weight-decay-steps 200000`
  - 对齐项：`--guidance-scale 0.5`、total_steps 1M、diffusion_steps 6、energy_weight 0.4、temp_weight 0.6、seed 42(主行)
- ⚠️ **这些手调参数的 argparse 名需先核对**（脚本里是否叫 `--actor-lr`/`--update-per-step` 等，连字符 vs 下划线），补前务必先 dry-run 打印 config 逐项比对 paper_metadata，确认新 run 与主表 run **只差 seed**。
- **MLP 侧同理**：OfficeMedium MLP run（`diffusion_mlp_bcfix_clean_OfficeMedium_..._20260323`）也需核对其 paper_metadata 手调超参（§10.5 记为 actor_lr 5e-5/batch 512/critic_lr 5e-6/violation_penalty 12/update 0.25），补 MLP 3-seed 要对齐它、不是对齐 School MLP。
- **规模**：FNO+MLP 各补 seed 0/1（seed42 已有）= 4 个 run。可参考本轮并行经验（3070 显存足）。
- **启动**：dropt 环境绝对路径 python.exe + PYTHONIOENCODING=utf-8（§11.3 两坑）。

### 14.3 补实验优先级（复盘后确认）
1. **必做（发表门槛）**：OfficeMedium FNO+MLP 补 seed 0/1，补齐主表唯一缺口的误差棒。
2. **强烈建议**：SAC 基线差 6× 的解释/重调（审稿人必问）。
3. **选做（加分）**：统一 modes 的 m2/m4/m8 扫描（支柱 2 机制）、分区轴空间谱（支柱 3 axis mismatch）。

### 14.4 【07-13 已启动】OfficeMedium 补 3-seed —— 4 run 并行训练中
- **已启动 4 run**：FNO(guided) seed 0/1 + MLP seed 0/1（seed42 已有主表 run）。log-prefix `officemedium_{fno,mlp}_1m_s{0,1}`，stdout `run_logs/officemedium_{fno,mlp}_1m_s{0,1}.log`。
- **配置逐项核对通过**（对照两个主表 run 的 paper_metadata，ground truth，只差 seed）：
  - 共有手调训练参数：`--actor-lr 5e-5 --critic-lr 5e-6 --batch-size 512 --bc-weight 1.0 --bc-weight-final 0.6 --bc-weight-decay-steps 200000 --update-per-step 0.25 --violation-penalty 12`
  - FNO 专有：`--fno-width 64 --fno-layers 2 --fno-modes 6 --guidance-scale 0.5`（MLP 不传这些，脚本不支持）
  - 已实测启动后打印 config 与主表 s42 全项一致（仅 seed 不同）。
- **GPU 容量结论（实测）**：显存 free 5.4GB → 上限约 6-7 run；但算力瓶颈是 FNO 的 6 步反扩散采样，**3 个 FNO 并行即到 88-90% util 饱和**。本批 2FNO+2MLP 并行 util 86%、显存 4.5GB used，是最优；再加不会更快（时间片轮转）。
- ✅ **【07-14 完训】4 run 全部跑满 245 epoch（扛过一次会话重启，进程续跑未断）**。OfficeMedium 3-seed 最终（末8窗，18区）：
  - FNO 7016±22 / 违规 3.37±0.19 (每区率18.7%) / comfort 0.61±0.02
  - MLP 7202±26 / 违规 3.71±0.17 (每区率20.6%) / comfort 0.65±0.01
  - 能耗省 **2.6%** / 违规降 9.2%。各 seed 明细 FNO 6988/7040/7020、MLP 7191/7238/7178，**std 极小、洼地坐实非噪声**。已替换 §2.3bis。
- ⚠️ 启动同两坑：dropt 环境绝对路径 python.exe + PYTHONIOENCODING=utf-8。
- 新增 run 目录：`log_building/officemedium_{fno,mlp}_1m_s{0,1}_OfficeMedium_Hot_Dry_2026071*`。
