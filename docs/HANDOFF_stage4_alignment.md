# HANDOFF 阶段四 —— 结构对齐 + 支柱 2 机制

> 日期：2026-07-14 创建，**07-15 复盘更新**。接续 `HANDOFF_option3_bear_journal.md`(阶段三, 07-08~07-14)。
> 目标（用户原话）：不追顶刊，只要内容自洽、故事完整、能发表即可。~~IEEE IoT-J 档~~ **投稿靶待定**：07-15 讨论认为建筑能源类(Energy and Buildings / Applied Energy)对本文 archetype 比 IoT-J 更契合，定稿前比较两类（§6.5）。语言：中文。
> **本文件是"当前状态的干净快照 + 新阶段起点"，不是 append 日志。** 阶段三 handoff 已 380+ 行、层积过多；本文只提炼**已核实的确证结论 + 阶段四新决策**，历史推导细节指向阶段三对应节（§编号沿用阶段三文件）。所有数字均在 07-14/07-15 复盘中拿 event / paper_metadata / 源码逐项核实。
> **07-15 复盘要点**：①洼地归因混了 MLP 调优不一致变量(§1)；②支柱 2 reframe 为"成分价值场景依赖、不赌单调"、guidance 发现提上来并入(§4)；③OfficeMedium NoGuide 升为决胜实验、guidance 有偏实现拆雷(§6.2)；④新增"稳妥版叙事"底线方案(§4bis)；⑤换靶讨论(§6.5)。

---

## 0. 一句话现状

阶段三已闭环：**三建筑主表全部 3-seed 齐口径、支柱 1（FNO 跨规模稳赢 MLP）锁死、OfficeMedium 洼地 3-seed 坐实**。
阶段四核心动作：**把 OfficeMedium 的结构参数从异类(L2/modes6)拉回与另两栋对齐(L1/modes4)**——因为复盘发现 OfficeSmall 与 SchoolPrimary 早已对齐(都 L1/modes4)，唯一异类是 OfficeMedium，且当初那套 L2/modes6 是随手设的、无依据。对齐后三栋结构统一、支柱 2 谱截断因果链才干净。
✅ **【07-15 21:15 完成】OfficeMedium FNO 对齐版 3-seed 全部跑满 245ep**（进程已退），正式末段 8 点窗：能耗 **7042±16** / 违规 3.34±0.12(每区率18.5%) / comfort 0.606±0.011（各 seed 7049/7019/7057）。已替换 §1 主表。**对齐无损坐实**（vs 原 L2/m6 版 7016±22，std 重叠、无显著差）。抽数脚本 `scripts/officemedium_aligned_tailmean.py`。**阶段四结构对齐动作至此闭环。**

🔴 **【07-16 进行中·阶段四未全闭环】** 结构对齐(L1/m4)已闭环,但用户新开"训练超参统一"子动作:把 OfficeMedium 从手调档拉回 Small/School 默认档(解决 §1 line 32 洼地混淆变量)。探针 `officemedium_fno_default_probe_s42` 在跑(判据见 §9.2),5 个 medium SAC(#6,7,13,14,15)HOLD 等它。**主表 OfficeMedium 行(7042)在探针验证成立后将被默认档 3-seed 重跑替换**;探针崩则退回手调档(7042 保留)。3 个 SAC s42(small/medium/school)+ 无人值守滚动脚本(PID 26968)在跑,真相源 `run_logs/SAC_QUEUE.md`。

---

## 1. 【确证】三建筑主表（全 3-seed，1M，guidance0.5，末段 8 点窗均值 mean±std）

能耗 kWh / avg_violations 计数 / 每区违规率(=violations/zone数)：

> 🔴 **【07-16 决定·OfficeMedium 行将被重跑替换，见 §9】** 用户决定把 OfficeMedium 的训练超参从手调档统一到 Small/School 默认档（解决下方 line 32 的"MLP 调优不一致"混淆变量）。下表 OfficeMedium 行(7042/7202)是**手调档**数据，统一探针验证成立后将被**默认档 3-seed 重跑**替换。Small/School 行不受影响（本就默认档）。

| 建筑 | 区数 | 结构 | Guided-DiffFNO | Diff-MLP | 能耗省 | 违规降 |
|---|---|---|---|---|---|---|
| OfficeSmall | 6 | L1/m4/w48 | 871±3 / 0.50 / 8.3% | 994±28 / 1.43 / 23.8% | 12% | 65% |
| OfficeMedium | 18 | L1/m4/w64 | 7042±16 / 3.34 / 18.5% | 7202±26 / 3.71 / 20.6% | **2.2%** | 10% ⚠️手调档待重跑 |
| SchoolPrimary | 25 | L1/m4/w128 | 6418±18 / 7.08 / 28.3% | 13368±1098 / 17.11 / 68.4% | **52%** | 59% |

- 📌 **数据源(单一真相)**：表中数字统一以 `paper_figures_v2/master_metrics_v2.csv`(3-seed, 末段 W=8 窗均值, 从 event 现算, 可复现)为准。**【07-16 校准】OfficeSmall MLP 由早期抽数 991±38/1.44/24% 更正为权威 994±28/1.43/23.8%**(seed 池不变, 仅统一窗口口径)；论文图与 `docs/PPT_SLIDE_CONTENT.md` 同源。抽数管线 `scripts/extract_master_metrics.py`。
- ✅ **【07-15 完成对齐】OfficeMedium 行已用对齐版(L1/m4/w64) 3-seed 跑满 245ep 替换**：7042±16 / 3.34（各 seed 7049/7019/7057，std±16 极小）。三栋现全部 **L1 + modes4 统一**，width 按 state_dim 分化(48/64/128)。
- ✅ **对齐无损坐实**：对齐版(L1/m4) vs 原版(L2/m6) = 能耗 7042±16 vs 7016±22（std 重叠、统计无差异）、违规 3.34 vs 3.37、comfort 均 0.61 → **简化结构零性能损失**，"统一最简结构"可辩护、非妥协；原版 L2/m6 是过度参数化的随手值。MLP(7202) 不变（无 modes/layers）。
- ⚠️ **违规口径**：`avg_violations`=每时刻超±1°C 容差的**区数**再对时间平均（逐区计数，被 zone 数放大），故跨楼必须归一化成每区违规率。禁与单 seed 老表 `table_1m_metrics.csv` 的 Violation Rate(比率) 混用。详见阶段三 §10.2 / §10.5。
- **洼地**：能耗省幅 12%→2.2%→52%（对齐版）**非单调**，OfficeMedium 是洼地，3-seed 坐实非噪声(std±16)。支柱 2 叙事因此不能讲"随区数单调增强"，走"耦合结构调制"退路（§4 支柱 2）。**注：对齐(L1/m4)与原版(L2/m6)洼地都在(2.2% vs 2.6% 无显著差)，故洼地非架构选择造成，是楼本身性质。**
- 🔴 **【07-15 复盘新增·洼地归因的混淆变量】** 省幅% = (MLP−FNO)/MLP，**分母 MLP 强弱直接决定省幅**。三栋 MLP 基线调优程度**不一致**：**OfficeMedium MLP 是唯一单独手调的**(actor_lr 5e-5/batch512/vp12/update0.25，07-15 grep 核实)，School/OfficeSmall MLP 是脚本默认档(actor_lr 1e-4/batch256/vp10/update0.5)。→ OfficeMedium 有更强基线(省幅被压低)、School MLP 崩到 68% 违规率是弱基线(省幅虚高)。**故 12%→2.6%→52% 曲线形状部分由"MLP 基线调优不一致"驱动，不纯是耦合结构。** ⚠️ **边界**：楼内对比(支柱1)**不受影响**——每栋 FNO 与 MLP 用同一套超参(OffMed 两者都手调、School 两者都默认)，楼内公平；受影响的只是**把跨楼省幅%并排成"结构先验随耦合调制"曲线**(支柱2机制叙事)的干净度。诚实解读：School 52% 里有一部分来自"MLP 没调好"，MLP 被认真调时(OffMed)FNO 骨干优势缩到 2.6%。审稿人大概率想到这层("是不是 School MLP 没调好才显 FNO 神")，支柱2叙事须主动承认此混淆、或把三栋 MLP 拉到同等调优档重比。
  - ✅ **【07-16 决定·正面解决此混淆】** 用户选择"拉到同等档重比"这条路：把 OfficeMedium(唯一手调栋)的 8 个训练超参统一到 Small/School 默认档，使三栋 FNO 与 MLP 全部同档。这样跨楼省幅曲线不再混"调优不一致"变量。执行=探针先行(见 §9)，探针验证成立才铺开。**注意 width 不统一**(按 state_dim 规律 48/64/128，容量旋钮非训练超参，统一反违规律 + 毁"参数少一量级"卖点)。

---

## 2. 【确证】架构真相 —— 复盘更正："不是三栋全乱，是 OfficeMedium 一栋异类"

从 paper_metadata（ground truth，07-14 核实）：

| 建筑 | width | layers | modes | state_dim | rfft长 | 谱截断保留 |
|---|---|---|---|---|---|---|
| OfficeSmall | 48 | **1** | **4** | 20 | 4 | 100%(零截断) |
| SchoolPrimary | 128 | **1** | **4** | 77 | 13 | 31% |
| OfficeMedium | 64 | **2** | **6** | 56 | 10 | 60% |

- **OfficeSmall 与 School 早已对齐**：都 L1 + modes4，只有 width 不同(48 vs 128)。
- **width 不同是有规律、可辩护的**：width 随 state_dim 涨（48/64/128 对 20/56/77，全满足 width≥state_dim，阶段三 §9.3）。width 是**容量旋钮**，按输入维度定，三栋不同天经地义、不该统一。
- **唯一异类 = OfficeMedium**：layers=2、modes=6 两个结构/机制参数都跟另两栋不一样。用户确认这套是**当初随手设的、无依据**。
- ⚠️ **阶段三 "modes 4/6/4 三栋手调不统一" 的表述夸大了问题**：真实情况是 layers/modes 这两个机制参数 Small/School 本就统一在 L1/modes4，只差 OfficeMedium 一栋。**这是阶段四的出发点。**

---

## 3. 阶段四动作：OfficeMedium 结构对齐重跑（✅ 已完成 07-15）

### 3.1 配置决策（只改机制参数，width/训练超参保持）
- **改**：`--fno-layers 2→1`、`--fno-modes 6→4`（对齐 Small/School 机制参数）。
- **保持 `--fno-width 64`**：符合 width≥state_dim(56) 规律，是容量旋钮非机制参数，统一反而违反规律。
  - ⚠️ **width 决策的诚实 caveat（07-14 深思后留痕）**：w64 与 L2/m6 是同一批"随手设"值，本次只判定 layers/modes 为待改、保留 w64，理由是 **width 是本次对照的"被控协变量"**——保持与主表一致(w64)才能让新旧两版可比、维持"只变 layers/modes"的单变量干净性；若此刻同时动 width 会污染对照。
  - **w64 不会触瓶颈**：64≥56，未跨过 §9.3 的 width<state_dim 掉点线（School w48<77 才掉点）。但**富余仅 8，是三栋最小**(Small 48-20=28 / School 128-77=51 / Med 64-56=8)。
  - **线索**：若对齐版(L1/m4/w64)省幅意外大跌，**第一个排查 width 是否偏小**（富余最小）——但当前无理由预先改，width 的"是否最优"是独立问题，应单独扫描(w48/w64/w96)而非混入本次对照。
- **保持 OfficeMedium 手调训练超参**（这些是"怎么训"，与结构无关，改了会引入混淆）：
  `--actor-lr 5e-5 --critic-lr 5e-6 --batch-size 512 --bc-weight 1.0 --bc-weight-final 0.6 --bc-weight-decay-steps 200000 --update-per-step 0.25 --violation-penalty 12 --guidance-scale 0.5`
- **MLP 不重跑**：MLP 无 modes/layers 概念，现有 3-seed(7202±26) 照用，直接与新 FNO 对齐版比。
- **只重跑 FNO 对齐版 3 seed(42/0/1)**，log-prefix `officemedium_fno_aligned_1m_s{42,0,1}`。
- ✅ **完训（07-15）**：三 seed 全部跑满 **245/245 ep**，config 已 grep 核实 = L1/modes4/width64（§3.1 目标达成，只变 layers/modes）。末段 8 点窗均值：能耗 **7042±16**（各 seed 7049/7019/7057）/ 违规 **3.34±0.12** / comfort 0.61±0.01。已替换 §1 主表 OfficeMedium 行。

### 3.2 对齐后三栋结构完全统一（干净故事）
| 建筑 | width | layers | modes | 谱截断保留 |
|---|---|---|---|---|
| OfficeSmall | 48 | 1 | 4 | 100% |
| OfficeMedium(对齐后) | 64 | 1 | 4 | **40%**(现 60%) |
| SchoolPrimary | 128 | 1 | 4 | 31% |
- 三栋 **L1 + modes4 统一**，width 按 state_dim 规律分化。**谱截断保留比例 100%→40%→31% 纯由 zone 驱动**，支柱 2 因果链干净。可讲成"统一最简结构(单层/modes4)，只让 width 随规模走"——有原则、好辩护。

### 3.3 对齐重跑的结论（✅ 245ep 完训坐实）
- **省幅变化**：原 L2/m6 省 2.6%（保留 60%）→ 对齐 L1/m4 省 2.2%（保留 40%）。能耗 7042±16 vs 7016±22 **std 重叠、统计无差异**（差 +26/+0.4%，噪声内）；违规 3.34 vs 3.37、comfort 均 0.61 均无差。→ **"持平"分支坐实。**
- **三条结论**（都进论文）：
  1. **统一无损**：简化到 L1/modes4（少一层 + 截断 60%→40%）性能零损失 → "统一最简结构 L1/modes4，width 随 state_dim"可辩护、非妥协；原版 L2/m6 是过度参数化的随手值。
  2. **洼地非架构造成**：对齐版仍是洼地（2.2%），洼地是 OfficeMedium 本身性质（耦合最强 10.65），非 L2/m6 偏差。
  3. **性能对 modes 数不敏感（支柱 2 机制点）**：在这栋楼上截断 60%→40%（更强）省幅无显著变化 → 佐证洼地与截断强度无关。⚠️ 勿过度解读 2.6%→2.2% 那 0.4%（噪声，不能说"截断更强省幅↓"）。
- ⚠️ **这与 §1 洼地混淆变量（MLP 基线调优不一致）不冲突、是两件事**：本节说"改 FNO 结构不改省幅"；§1 说"三栋 MLP 分母调优不一致使跨楼省幅曲线形状部分失真"。两者都要在支柱 2 诚实交代。
- 这是**同楼控变量**的结构对照，比跨楼比较干净得多，本身就是支柱 2 的强机制证据。

### 3.4 启动踩坑（每次重启必看，阶段三 §11.3）
1. **必须用 conda `dropt` 环境**：git-bash 里 `conda activate` 因 cygwin 路径失效，直接用绝对路径 `/c/Users/zouwei/anaconda3/envs/dropt/python.exe`。dropt = torch2.7.1+cu118/tianshou0.5.1/CUDA可用。
2. **必须 `PYTHONIOENCODING=utf-8`**：脚本打印含 ⚠️ emoji，经 tee 管道退回 gbk 编码会 UnicodeEncodeError 秒崩。
3. **GPU 容量（实测）**：8GB 卡，free ~5GB→显存上限 6-7 run；算力瓶颈是 FNO 6 步反扩散采样，**3 个 FNO 并行即 88-90% util 饱和**。本批 3 FNO 并行是上限、也最优。
4. **配置对齐验证**：启动后先 grep 打印的 config 逐项比对基准，确认只有目标参数(layers/modes)变、其余与主表一致，再放任跑。

---

## 4. 文章三支柱现状（阶段三 §5 提炼）

- **支柱 1｜单→三建筑，FNO 跨规模稳赢 MLP**：✅ **最硬、已锁死**。三栋 3-seed 主表齐口径（§1）。核心：FNO **在 OfficeSmall 参数少一个量级(6.8×)** 仍赢，且优势在大楼放大(School 省 52%)。⚠️ **措辞收口(07-15)**：参数比 6.8×/1.8×/**1.1×**(阶段三 §9.2)——"少一个量级"**只在 OfficeSmall 成立**，School 参数已追平 MLP(1.1×)。写作时说"参数效率优势在小楼最显著、大楼靠结构本身取胜"，勿讲成三栋都少一个量级。附骨干因果解耦（阶段三 §13.4：School 纯骨干效应 NoGuide vs MLP 省 38%/降 25%，独立于 guidance）。
- **支柱 2｜成分价值的场景依赖（FNO 截断 + guidance 双成分，何时起作用）**：🔸 **阶段四主攻**。会议版声称"谱截断=平滑先验"，但 OfficeSmall 零截断(modes4=rfft4)。改成研究问题。
  - 🔴 **【07-15 关键 reframe】叙事绝不赌"随规模单调增强"**——那是坑，重蹈 FNO 单调叙事覆辙。改赌**"成分价值场景依赖(可非单调)，OfficeMedium 是共同洼地"**。这个 claim 一个洼地破坏不了(只需证"价值非恒定")，且洼地怎么走都成立。
  - **两个成分都表现尺度/场景依赖**：①FNO 截断优势(主表 12%→2.6%→52%，洼地)；②guidance 增量(OfficeSmall ~3% 几乎不省能耗 → School 22%，阶段三 §13.4)。统一解释候选：OfficeMedium 耦合最强(10.65)、最难控，是两成分的共同洼地。
  - ⚠️ **两个成分的洼地是不同的量、吃不同混淆**：FNO 截断洼地在"FNO-vs-MLP 省幅"，混了"MLP 基线调优不一致"(§1 复盘新增，OffMed MLP 唯一手调)。**guidance 增量是同骨干同超参、只差 guidance 开关的对比，不含 MLP → 对该混淆免疫，比主卖点干净。**
  - 🔴 **决胜实验 = OfficeMedium NoGuide(现矩阵❌缺)**：它决定洼地是"FNO 独有"还是"FNO+guidance 共有"。共有→给出"OfficeMedium 一致异常、由最强耦合解释"的漂亮统一叙事(洼地从 bug 变 feature)；独有→两成分机制不同，需分述。**现在 guidance"场景依赖"只有 Small/School 两点外推、且缺的正是已知异常的中间点——必须补，见 §6.2。**
  - **坐实支柱 2 的关键**：对齐重跑(§3) + 大楼 m2/m4/m8 扫描(截断维度) + OfficeMedium NoGuide(guidance 维度)。
- **支柱 3｜修 axis mismatch**：🔸 半修，**建议降为 discussion/limitation 段、不硬撑成支柱**(§6.5)。FNO 滤分区轴、平滑证据测时间轴。阶段三 §10.3 有一条确证（FNO 压跨区空间差异方差~3×）+ 一条待验证（归一化谱形状 FNO 未必更低通，会议版因果链存疑）。

---

## 4bis. 【07-15 新增】稳妥版叙事（肯定能发的底线方案）

> 与 §4 的关系：§4 是"想发得好"的完整三支柱(含要赌 modes 扫描/洼地机制的部分)；本节是**剥掉一切还在赌的东西、只用已锁死资产**的底线版。**目标"自洽、能发"用本节即可满足；支柱 2/3 成不成立都不影响发表。**

### 一句话叙事（描述性，非机制性）
> **一个以 FNO 为动作轴去噪器的扩散策略，是跨建筑规模鲁棒、参数高效的 HVAC 控制器；通过三栋楼受控对比，严格厘清骨干与引导各自贡献，并诚实刻画每个成分何时有效。**
- ⚠️ 关键：是"我们展示了/厘清了"(描述性、数据在就成立)，**不是**"我们发现了结构先验何时起作用的规律"(机制性、要赌 modes 扫描+解洼地混淆)。这条分界就是稳妥与冒险的分水岭。

### 只站三块锁死资产
1. **跨规模鲁棒(支柱1)**：三栋 3-seed、std 极小、FNO 稳赢 MLP。最硬。
2. **骨干 vs 引导干净解耦**（**真正的智识增量**，不是"多测两栋"）：School NoGuide vs MLP 纯骨干省 38%/降 25%，独立于 guidance。别人没做、数据干净，把它推到台前当分析贡献，顶回"就是多测几栋"的印象。
3. **诚实的成分刻画（负面结果=加分）**：残差非 essential(NoRes 近 Pareto)、guidance 主要压违规、FNO 截断在小楼零发生。

### 洼地/guidance 的"不赌"处理 = 报告但不解释
- **洼地**：照实写"FNO 优势尺度/场景依赖、非单调，OfficeMedium 仅 2.6%"，**停在这里当诚实观察**。不建耦合结构因果模型(那要赌 modes 扫描+解混淆)。被问"为何中间低"→答"观察到但未完全解释，候选成因耦合强度差异，留待后续"。诚实的"不完全知道"比撑不住的机制模型安全。
- **guidance**：低姿态"已验证组件" + §6.2-5b 那句诚实说明。**不提"价值随规模变化"**(那要赌 OfficeMedium NoGuide)，只说"各楼贡献不同、主要压违规"。

### 稳妥版最小必做（比 §6 全量短很多）
1. ✅ ~~OfficeMedium 对齐 run 跑满 245ep~~ **【07-15 完成】** 7042±16，主表已替（§0/§1）。
2. **修 SAC + 补 School 基线**（§6.1，**现唯一硬成本**，任何版本都躲不掉）。← **下一步就是这个**
3. **写作卫生**：违规每区率、末段窗均值、补显著性、残差改口、guidance 诚实那句。
- **踢出稳妥版**：modes 扫描、Conv1d、OfficeMedium NoGuide、guidance Level-1——全是"想更强"的升级项，非发表门槛。

### 诚实定位（不哄）
稳妥版**肯定能发，但在"扎实应用向期刊"档**(建议 Energy and Buildings 类)，**不是强贡献**——苛刻审稿人可能说"方法是会议的、增量主要是更多楼"。防线=第 2 块资产(骨干/引导解耦)。**若想从"能发"升"发得好"，唯一最值得加的一件事是大楼 modes 扫描**(让支柱 2 从观察变机制)——但那是升级，非稳妥版的一部分。

---

## 5. 已封死的路（阶段一/二/三，别重走）

- guidance 做更花（自适应η/oracle投影/LS拟合/能耗解析梯度）四方向全否（`HANDOFF_journal_direction.md`§5）。
- Sinergym 转向（动作是 1-2 全局设定点，FNO 无处落脚）、时间轴 FNO（自拆会议地基）、双路门控（实测更差）——阶段三 §1 / `HANDOFF_sinergym_direction.md`。
- guidance 作为**方法**是拥挤赛道（QGPO/QGF/DAC 撞车），**方法层面必须降级**，不能当核心新机制。🔴 **【07-15 更正】但"降级方法"≠"文里静音"**——上一版把发现连方法一起降了，是把孩子跟洗澡水一起倒。guidance 增量(Full vs NoGuide)是**同骨干同超参单变量对比、不含 MLP，比主卖点还干净**，且其"场景依赖价值"与支柱 2 同轴 → **提上来并入支柱 2(§4)，不静音**。measure 见 §4 支柱 2 与 §6.2。
- 残差**非 essential**（阶段三 §10.5：Full/NoRes 能耗 std 重叠），期刊版须改口会议稿"残差 essential"说法。

---

## 6. 后续待办 / 补实验缺口（优先级）

> 07-15 全面盘点磁盘现有 run 后重排。判断依据：支柱 1 已锁死、可独立成篇（§4），故排序看"补哪些能让支柱 2/3 立住 + 堵审稿人必问"。**modes 扫描的优先级是条件性的**（见 §6.2），按你要不要认真讲支柱 2 决定。

### 6.0 现有实验矩阵（✅有 / ⚠️有但不合用 / ❌缺，均 07-15 核实）

| 实验 | OfficeSmall | OfficeMedium | School |
|---|---|---|---|
| 主表 Guided-FNO vs MLP 3-seed | ✅ | ✅对齐版7042±16(07-15完成) | ✅ |
| NoGuide（骨干解耦） | ✅3-seed | ❌ | ✅3-seed |
| NoRes（残差消融） | ✅3-seed | ❌ | ❌ |
| modes 扫描 m2/m4/m8 | ⚠️零截断退化* | ❌ | ❌ |
| guidance-scale η=0/.5/1/2 | ⚠️单seed未聚合 | ❌ | ❌ |
| SAC 基线 | ✅3-seed(差6.8×) | ✅单 | ❌ |
| SAC+MPC 基线 | ✅3-seed | ✅单 | ❌ |

*OfficeSmall rfft长=4，m2/m4=保留2/4、m8=m4(超不过4)→ 只有 m2 一个真截断点，且日志 m2 vs m4 低频比 0.867/0.868 几无差。要看"截断强度↔优势"必须去 rfft 更长的大楼。

### 6.1 第一档 · 必做（发表门槛，不补审稿人会卡）

1. ✅ **【已完成 07-15】OfficeMedium 对齐重跑**（§3）：3-seed 跑满 245ep，末段窗 **7042±16 / 3.34**，对齐无损坐实（vs 原版 7016±22 无显著差），主表已替换、三栋结构全统一 L1/modes4。**支柱 2 下一步主攻转向 §1 洼地混淆变量**：三栋 MLP 基线调优不一致（OffMed 手调 vs School/Small 默认档），须主动承认或把三栋 MLP 拉到同等档重比。
2. **修 SAC 基线**：OfficeSmall 5980(6.8×能耗)差到审稿人必问"是不是没调好"，要么重调到合理量级、要么文中明确解释 BEAR 上 SAC 的已知失效原因。**且 School 完全无 SAC/SAC+MPC**——最亮那栋(省52%)反缺基线、最扎眼，至少补 School 的 SAC+MPC(oracle上界) + 一个调好的 SAC。
   - **【07-16 进展】** 18 个 SAC/SAC+MPC 1M 重跑已启动（诊断+公平协议见 `HANDOFF_sac_baseline.md`）。⚠️ **medium 5 个 run(#6,7,13,14,15) 已 HOLD**：受 §9 统一决定影响，OfficeMedium 的 SAC 对齐目标从手调档(bc0.6/vp12)变默认档(bc0.1/vp10)，须等探针验证后改 launcher 再跑。Small/School 的 SAC 照常。队列真相源 `run_logs/SAC_QUEUE.md`。

### 6.2 第二档 · 强烈建议（把"应用"抬成"有机制的研究"，堵新颖性质疑）

3. **大楼 modes 统一扫描 m2/m4/m8**（OfficeMedium+School，OfficeSmall 只作对照）：支柱 2 的核心机制主张是"谱截断真起作用"，现只有三点跨楼相关支撑、且被 modes 手调不统一(4/6/4)+耦合强度双变量污染——**同楼控变量扫描才是"截断强度↔优势"的受控证据；没有它，支柱 2 机制主张只靠跨楼相关，站不住。** ⚠️ **优先级条件性**：只发支柱 1 则可放一放（洼地可用 §10.4 耦合结构假说口头解释，弱但够）；要认真立支柱 2，此项近乎必做。**成本控制**：先单 seed 筛信号，有信号的档再补 3-seed（承阶段三 §1.4 教训）。⚠️ 画图前先修 `scripts/paper_coupling_structure.py:39` 硬编码 `MODES=4`（OffMed 那几列错，阶段三 §10.4）。
4. **Conv1d 去噪器对照**（回答"凭什么 FNO 而非任意空间先验"）：现只有 FNO vs MLP，审稿人会问 FNO 赢靠**谱结构/全局混频**还是**任何空间归纳偏置**都行。加一个 Conv1d 去噪器(有空间局部性、无谱截断)：介于 MLP-FNO 间→"空间先验有用+FNO全局谱额外加分"；≈FNO→卖点其实是空间局部性、须改叙事。**⚠️ 需改代码**（新去噪器，把 `diffusion/model_fno.py` 的 SpectralConv1d 换成 nn.Conv1d），非纯跑 run，成本高于其它项，但这是防住核心新颖性(FNO-on-action-axis)最值的对照，比多跑 PPO/TD3 强。
5. 🔴 **OfficeMedium NoGuide —— 决胜实验(07-15 从"选做"升级)**：这是支柱 2 guidance 维度的关键点。现"guidance 价值场景依赖"只有 Small(~3%)/School(22%)两点外推，**缺的正是已知异常的中间点(洼地)**。它决定洼地是"FNO 截断独有"还是"FNO+guidance 共有"：共有→"OfficeMedium 一致异常、由最强耦合解释"的统一叙事(洼地变 feature)；独有→两成分分述。⚠️ 只差 guidance 一个开关、其余对齐 OfficeMedium 手调主表(§14.2 那套 w64/L2/m6/actor_lr5e-5…，g0.5→0.0)。**顺带补 OfficeMedium NoRes**(残差消融现只 OfficeSmall 有)。
5b. **guidance 有偏实现拆雷 —— 【07-15 已定：默认不改代码，只加一句话】**：现实现是 QGPO 明确批评的**朴素 point-estimate energy guidance**——对 x̂₀ 点求 `∂Q/∂x̂₀` 且 detach(`main_..._guided_bcfix_clean.py:279-288` + `diffusion.py:178-182`)，两处偏差：(a) Jensen(拿点上 Q 替 exp-Q 期望的 log)；(b) 更致命的 **critic off-support**——高噪声步 x̂₀ 在 Q 训练支撑外求值(stdout 刷"Critic梯度过大→裁剪20"即征兆)。**✅ 决定：代码保持现状，论文加一句**"采用 point-estimate energy guidance，是 QGPO 指出的有偏近似，作经验有效组件、非本文贡献"。**理由**：guidance 已定为非卖点，改 Level-1 会作废现有全部干净 3-seed 数据(要重跑)，不划算；风险不在"用了朴素版"而在"没说"→ 承认即拆雷、零成本。**前提**：全文 guidance 一律以"已验证组件"低姿态出现，**不用"我们提出 critic-guided…"当创新点**。**例外**：仅当你改主意要把 guidance 当亮点卖，才做 Level-1(DPS 式：去 detach + 梯度穿去噪器对 x_t + 噪声层加权~1/σ_t²，改 `p_mean_variance` 几行)。**永不做 Level-2(QGPO/CEP 新网络)**。
5c. **稳定器 vs 推理引导混淆(guidance 机制主张的前提)**：现 guidance 训练采样+测试都开(§9.1)，分不清 School 22% 是"推理期能量倾斜"还是"训练期稳定收敛"(§13.4 线索：NoGuide energy std=553 vs Full 18，大楼无 guidance 训练不稳)。要把"稳定器"写成**断言**须补对照(训练全程 guidance vs 只推理期)；不补则只能写成**开放机制线索**，别写死。

### 6.3 第三档 · 便宜/低风险（有余力就做）

6. **diffusion_steps 扫描 2/4/6/8/10**：推理期不重训、很便宜，论证"6 是拐点"进附录，堵一个必问。
7. **guidance-scale 扫描聚合成图**：η=0/.5/1/2 数据已在 event 里，纯抽数、零算力。
8. **分区轴空间谱补 seed**：支柱 3 "压跨区方差3×"现单seed单楼(阶段三 §10.3)，多补 seed 才敢写。
9. **（选）朴素参照 RBC/固定设定点**：给读者标尺，HVAC 论文习惯有，很便宜。

### 6.4 关于"是否跑其他模型"的判断

**别急堆 RL 基线(PPO/TD3)**——无底洞且不服务核心论点。优先级：先修已有 SAC（比新增基线性价比高）> 加 Conv1d 去噪器（防新颖性的关键对照）> 便宜的 RBC 标尺。MPC 已有(oracle 上界)且扩散赢它，是强对比，保留。

- **主叙事**：🔴 **【07-15 更正措辞】"跨规模鲁棒系统 + 成分价值的场景依赖"**。⚠️ **不写"随规模单调增强/放大"**(FNO 截断和 guidance 都有 OfficeMedium 洼地，单调必翻车)；写"价值场景依赖、非恒定，OfficeMedium 是共同洼地、由最强耦合解释"(§4 支柱 2)。参数效率"少一个量级"只对 OfficeSmall 说，别外推三栋(§4 支柱 1)。
- **guidance**：方法降级为已验证组件(拥挤赛道)，但**发现不静音**——场景依赖价值并入支柱 2；按楼分述不外推("几乎不省能耗"只 OfficeSmall 成立、School 省 22%)；写作前拆有偏实现的雷(§6.2 第 5b)。
- **残差**：改口(NoRes 近 Pareto，会议稿"essential"说法作废)；违规一律每区率、一律末段窗均值；补显著性。支柱 3(axis mismatch)降为 discussion/limitation 段。
- 🔴 **【07-15 新增·重新评估投稿靶】**：四份 handoff 一路默认 IEEE IoT-J，**但没人质疑过是否最优**。这是 HVAC 控制 + RL + 机制分析型论文，**建筑能源类(Energy and Buildings / Applied Energy)对"方法 + 多环境鲁棒 + 何时有效分析"这个 archetype 接受度更高、门槛更清晰**，比 IoT-J 更天然契合。定稿前认真比较两类靶,别惯性投 IoT-J。

---

## 7. 关键文件索引

- 训练脚本：`main_building_fno_guided_bcfix_clean.py`(FNO+guidance)、`main_building_bcfix_clean.py`(MLP)。base argparse 在 `main_building.py`（参数名全用连字符：`--fno-width/-layers/-modes --guidance-scale --actor-lr --critic-lr --batch-size --bc-weight[-final/-decay-steps] --update-per-step --violation-penalty`）。
- FNO 去噪器：`diffusion/model_fno.py`。guidance 注入：`diffusion/diffusion.py:178-182`。
- 抽数脚本：`scripts/school_tailmean.py`、`scripts/school_fno_vs_mlp.py`（改 glob 可复用于任意 3-seed 对比）。
- 主表数据源：三建筑 run 的 `paper_data/paper_metadata.pkl`(完整args, ground truth) + event。OfficeSmall 3-seed CSV：`paperfigure_bcfixclean_smalloffice_multiseed/compare_energy_violations_mapping.csv`。
- 阶段三完整推导：`docs/HANDOFF_option3_bear_journal.md`（历史档案，§编号本文沿用）。

---

## 8. 【07-14 追加】School MLP 高位失稳诊断 —— 对比已公平、无需调参（本轮亲抽 event 坐实）

> 起因：本轮复盘一度提出"School MLP 用小楼默认超参、可能没调好 → 需补调 School MLP 才公平"这个担忧。**本节把它查清并否掉：对比已公平（同配置），MLP 高位失稳是真实现象，不需调参、无需补实验。** 这是一条"防止下一轮误去调 School MLP 浪费机时"的记录，不涉及任何叙事/口径改动主张。

### 8.1 配置公平性（复核 §10.6 记录 + 本轮确认）
- School 的 **FNO 与 MLP 共用同一套训练超参**（脚本默认档，只覆盖 `--building-type`）。二者差异**恰好就是被测对象**：骨干（FNO vs MLP）+ guidance（FNO 0.5 / MLP 结构上不支持，已由 NoGuide 变体单独解耦）。
- ⚠️ 结论：这**已经是公平的受控对比**（同配置，只变骨干）。MLP 并未被区别对待。"需补调 School MLP 才公平"这个本轮一度提出的判断**被否**——真正被单独手调的是 OfficeMedium（FNO+MLP 都手调），School 不是。

### 8.2 本轮亲抽三 seed 能耗曲线（`test/avg_energy`，非引用旧结论）
从三个完训 MLP run（`school_mlp_1m_s{42,0,1}_..._20260710_14{4107,4450,4613}`；⚠️ 另有 `s42_143905` 是 333B 假启动，忽略）与三个 FNO run（`school_guided_1m_s{42,0,1}_20260708`）直抽：

| | 三 seed 后半程平台(mean) | 后半程 std | 形状判定 |
|---|---|---|---|
| **MLP** | 11039 / 14675 / 13011 | 1064 / 1459 / 1305 | 高位有界震荡 |
| **FNO(Full)** | 6448 / 6403 / 6423 | 64 / 66 / 54 | 紧钉低位 |

⚠️ **口径说明（防误判为"与主表矛盾"）**：本表是**后半程(约122点)per-seed 的 mean 与 run 内部 std**，用于看曲线形状；此处 std 是**单 run 内的时间震荡幅度**，不是 seed 间离散。**主表 §1 的权威数字(FNO 6418±18)是末8点窗口、seed 间 mean±std**，口径不同、并不冲突。此表只服务"形状诊断"，论文一律以 §1 为准。

- **不是发散、不是配置坏**：MLP 三 seed 都从 ~18000 学下来、在各自平台**有界震荡**（末段不爬升到无穷），是训练**不稳定**、非崩坏。
- **MLP 早期确触及好解但守不住（本轮验证非单点偶发）**：s0 在 epoch 2-16 **连续 16 点**停在 6000-6700、s42 连续 ~15 点在 6000-7000，随后失稳爬高不再回落（s1 较散）。→ 是"稳不住"，不是"表示不了"。
- ⚠️ **口径纪律**：MLP 早期 min（~6000）是训练早期低点，**论文一律用末段窗均值**（§10.5/§1.4），本节的"早期触及"只用于机制判定，不得当作 MLP 的成绩写进表。

### 8.3 结论（本节唯一定论：不需要动 School MLP）
- **无需补调、也无需补跑 School MLP**：同配置对比已公平，MLP 高位失稳是真实且可复现的现象（三 seed 一致）。这一条只用于**阻止下一轮误去调 School MLP**。
- **一个数据支持的补充观察（非必须、不替换任何现有数字）**：现有 52% 是真实数字、照常用。此外数据还额外支持"相同配置下 FNO 三 seed 收敛极紧（§1：6418±18，seed 间），MLP 高位震荡且守不住"这一鲁棒性观察，若写作时想用可用，但**这是加分项，不要求改动主表或头条口径**。
- ⚠️ **不要重复我这轮犯的越界**：本节仅是 School MLP 的诊断记录，**不构成对支柱框架、主表数字、或叙事方向的任何否定**。之前会话里那些"支柱2机制搭错/应重构叙事"之类的评论已确认是过度发挥（handoff 原文早已诚实标注对应 caveat），勿据此改方向。

---

## 9. 【07-16 决定】OfficeMedium 训练超参统一到默认档（正在验证）

> 本节记录 07-16 这轮的核心决定 + 已启动的探针。**直接影响 §1 主表 OfficeMedium 行、§6.1 SAC 计划、§6.2 决胜实验对齐目标。** 决定详情也存 memory `guided-difffno-officemedium-config-unify`。

### 9.1 决定：把 OfficeMedium 从"唯一手调异类"拉回默认档
- **动机**：§1 line 32 的混淆变量——三栋 MLP 调优不一致（OffMed 唯一手调），使跨楼省幅曲线(12%→2.2%→52%)混入"基线强弱"变量、不纯是耦合结构。用户选"把三栋拉到同等档重比"这条正解，而非只靠写作承认。
- **三栋 FNO 参数 ground truth**（本轮从各 paper_metadata.pkl 逐项读出）：**Small 与 School 完全相同=脚本默认档；OfficeMedium 8 个训练超参全不同**（单独手调的"稳定套装"）。
- **要改（OfficeMedium 手调→默认档，8 项）**：bc_weight 1.0→0.8、bc_weight_final **0.6→0.1**、bc_decay_steps 200000→150000、violation_penalty 12→10、actor_lr 5e-5→1e-4、critic_lr 5e-6→2e-5、batch_size 512→256、update_per_step 0.25→0.5。
- 🔴 **width 不统一（用户已定）**：三栋 48/64/128 按 state_dim 规律(width≥state_dim，阶段三§9.3)，是容量旋钮非训练超参。统一会违规律(School 77 瓶颈)+ 毁支柱1"参数少一量级"卖点。OfficeMedium 保持 64。
- **统一叙事边界**（论文口径）：统一训练协议 + 统一最简结构(L1/modes4) + width 随 state_dim 分化。比"三栋完全一样"更专业、且对 width 之问有明确规则可答。

### 9.2 风险 + 探针先行
- 🔴 **风险**：OfficeMedium 是最难楼(耦合最强 10.65、洼地)，那 8 个手调参**可能是必需的连体稳定套装**，全拉默认档可能整体退化甚至崩。故**不直接铺开重跑，探针先行**。
- **探针（07-16 已启动）**：`officemedium_fno_default_probe_s42`（log `run_logs/officemedium_fno_default_probe_s42.log`），config 已 grep 逐项核实全部落默认档（脚本默认恰=Small/School 档，只传 building-type/width64/L1/m4/g0.5）。启动时 GPU 5.4GB free/util 92%（3SAC+1FNO 共存，显存足）。
- **判据（末段窗均值 vs 手调版 7042±16）**：
  - **接近或更好** → 统一成立，铺开 OfficeMedium 全套默认档重跑：FNO-Full/MLP/SAC+MPC/纯SAC × 3-seed(≈12 run)。改 `scripts/_sac_launch.sh` medium 行为默认档、解冻 SAC_QUEUE #6,7,13,14,15。§1 主表 OfficeMedium 行替换为新数。
  - **崩/大跌** → **先试 width 64→96**(富余仅 8 最紧 + 默认档 actor_lr 更激进)排除 width 不够，再判死刑；仍崩则**统一这条路走不通**，退回手调 0.6、跨楼差异靠写作诚实交代（回到 line 32 的"承认混淆"退路）。
- **回退保险**：OfficeMedium SAC+MPC@0.6(SAC_QUEUE #2) 已完训(energy 7701/viol 4.93) **保留不用**——探针崩→它是 0.6 成品，探针成→作废(只损失一个)。

### 9.4 【07-17 上午·探针中途快照，未完训不定案】
> ⚠️ 探针 ep181/245（~74%），**未到末段窗、不满足 §9.2 判据、不改主表**。此节仅记趋势，防下轮重新抽数。血泪教训：没完训不写死结论。
- **能耗已在平台**：ep15 即收敛到 ~6900，之后 165 点稳在 6900-7100 震荡（非发散、非爬升）。当前末16点窗：能耗 **7001±47** / 违规 4.10±0.29（每区22.8%）/ comfort 0.674 / rew -3.23。
- **初判倾向"统一成立"侧（待完训坐实）**：核心量能耗 7001 vs 手调档 7042±16 **持平**（差在噪声内）——说明 8 个手调超参拉回默认档**没破坏能耗**，这是探针第一问的答案。
- ⚠️ **一个诚实退化点**：违规每区率 18.5%→22.8%（+4pp）、comfort 0.606→0.674。温和退化（非 School MLP 68% 那种崩），符合"去掉为最难楼调的 vp12/bc_final0.6 稳定套装后违规控制略松"的预期。
- **决策点留给用户**：§9.2 判据只写"能耗接近或更好→铺开"。能耗达标但违规轻升——铺开与否需用户定（能耗持平是否足够触发，还是介意违规 4pp）。倾向：能耗持平已拿到统一核心依据，违规轻升可在论文如实讲（本就是洼地/最难楼性质）。
- **下一步**：等 ep245 完训，重抽末8窗确认能耗不反弹（现末8点 6995），满足判据即按 §9.2 铺开 medium 默认档重跑 + 解冻 SAC #6,7,13,14,15。抽数用 `scripts/officemedium_aligned_tailmean.py` 改 RUN 指向 probe 目录。

### 9.3 对 §6.2 决胜实验(OfficeMedium NoGuide)的连带影响
- §6.2 第 5 项写的 NoGuide "对齐 OfficeMedium 手调主表(w64/L2/m6/actor_lr5e-5…)"——若统一成立，**对齐目标改为默认档(L1/m4/w64/actor1e-4…)**，且顺带 NoRes 也用默认档。等探针定案后再据新档设计。

