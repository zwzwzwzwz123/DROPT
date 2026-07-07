# Handoff：Guided-DiffFNO 期刊版方向决策

> 目的：把"梯度引导增量能否撑起期刊 + 第三创新点该怎么选"这轮分析的**全部结论、已核实事实、死路、和推荐路径**交接给下一个会话/协作者。
> 日期：2026-07-06。作者：分析会话。语言：中文。
> 用户目标定位（原话）：**不追求顶刊，只要内容自洽、故事完整、能发表即可。**

---

## 1. 项目状态

- **ICCC 投稿版**（`docs/DiffFNO_ICCC2026_submission.pdf`）：DiffFNO = 条件扩散策略 + **FNO 动作轴去噪器** + 门控残差通路。只在 BEAR OfficeSmall/Hot_Dry 上评测。核心贡献是 FNO-on-action-axis + 残差。
- **期刊初稿**（`docs/GDMOPT.pdf`，Guided-DiffFNO）：在 ICCC 基础上**新增推理期 critic 梯度引导**（把 Q 的梯度注入反向扩散）。另加训练动态、Q vs Monte-Carlo return 相关性（r=0.619）、跨区协调分析、更多消融。
- **本轮要评估的增量** = "critic 梯度引导"这一条，以及是否需要 / 如何加"第三个创新点"。

---

## 2. 已核实的关键代码事实（都读过源码，非推测）

- **引导实现**：`build_guidance_fn`（`main_building_fno_guided_bcfix_clean.py:273-290`）算 `q=min(Q1,Q2).mean()`、`grad=∂q/∂x_recon`；注入在 `diffusion/diffusion.py:175-182`，`x_recon ← x_recon + η·grad` 后裁剪。**是对重构干净动作 x̂0 的"朴素"能量引导**（QGPO 指出的有偏形式）。
- **FNO 谱截断在主基准上没起作用**：`diffusion/model_fno.py` 的 `SpectralConv1d` 沿**分区轴**做 FFT；`modes=4`，OfficeSmall=**6 分区** → `rfft(6)` 只有 4 个复模态 → `modes=4` **保留全部、零截断**。日志里 m2 vs m4 低频比 = 0.867 vs 0.868（几乎相同）证实了这点。**截断只在大建筑上才真正发生。**
- **机制轴 ≠ 证据轴**：FNO 的 FFT 在**分区（空间）轴**，但论文平滑性证据（mean|Δa|、Welch PSD）在**时间轴**。因果链没接上，需修正或澄清。
- **BEAR 真值动力学是线性时不变的**：`X_new = A_d·X + B_d·Y`（`BEAR/BEAR/Env/env_building.py:251`），`A_d,B_d` 在 reset 时用 `expm` 算一次、之后不变。下一步温度**对动作仿射**。`A_d/B_d` 直接挂在 env 上。
- **MPC 基线本来就读 `env.A_d/B_d`**（`BEAR/BEAR/Controller/MPC_Controller.py:14-15`）= 它就是"上帝视角模型控制器"，成绩 1004.9 kWh / 1.78 违规，**比扩散方法差**。
- **违规按下一步温度算**（`env_building.py:254` `error = X_new*acmap - target*acmap`）。
- **能耗是动作的解析已知函数**：`E(a)=(Pmax/1000)·Σ mᵢ|aᵢ|·(Δt/3600)`，`∂E/∂aᵢ=常数·mᵢ·sign(aᵢ)`（按功率加权的符号向量，方向恒指向归零）。
- **楼宇路径无 obs 归一化**（normalize 只在 datacenter/expert 路径）。`acmap` 掩码可控分区。

---

## 3. 磁盘上已有、但**未写进初稿**的实验（重大：很多"审稿人会要"的数据已存在）

- **第二栋建筑 OfficeMedium 全套已完成**，且结论对用户有利：Guided-DiffFNO 6987 kWh / 3.26 违规，**同时**赢 NoGuide(7052/3.78)、MLP、SAC+MPC。→ 直接把"单建筑"变"多建筑"。
  数据：`paperfigure_bcfixclean_officemedium_partial/`、`log_building/*OfficeMedium*`。
- **第三栋建筑 SchoolPrimary（更大，~25 区）在训**（截至交接 epoch ~30）：`log_building/*SchoolPrimary*`。分区多，FNO 截断/约束才"真正有事可做"。
- **引导标度扫描 η=0/0.5/1/2 已跑**：`log_building/diffusion_fno_guided_OfficeSmall_*guidancescale=*`。数值在 TensorBoard event 文件里，**未聚合成 CSV**。η=0.5 是当前最优。
- **FNO 模态扫描 m2/m4/m8 已跑**：`log_building/fno_guided_m2_*`、`m8_*`、`fno_modes_2_vs_4_psd_summary.json`。
- **多 seed（0/1/42）** 主变体齐全；`paperfigure_bcfixclean_smalloffice_multiseed*`。
- **rectified flow** 备选去噪：`log_building/rectified_flow_*`。
- **`mixture` / `mixture_reg` / `hybrid_scalar` 变体**（本周新跑，PDF 之后）：`log_building/diffusion_fno_mixture*`、`*hybrid_scalar*`。**这些不在 root 的 .py 里，可能由 `scripts/` 驱动，本轮未搞清是什么 —— 见开放问题。**
- 主表（单 seed）：`log_building/table_1m_metrics.csv`；消融：`log_building/ablation_summary.csv`。

---

## 4. 新颖性对标（本轮检索结论）

- **critic 梯度引导 = 拥挤赛道，非新机制**：
  - QGPO（ICML 2023，Lu et al.）：明确指出**直接对样本求 Q 梯度是"有偏近似"**，正确能量引导要对加噪能量分布建模。**用户当前 x̂0 梯度正是它批评的朴素版。**
  - QGF（arXiv 2606.11087, 2026-06）：flow-matching 版"测试期 critic 梯度引导去噪步"——**近乎撞车**。
  - DAC（ICLR 2025，已被引）、2025 一批 energy-weighted / energy-guided flow：全在这条线上。
- **结论**：把 guidance 当核心 ML 贡献 → 熟悉这块的审稿人会立刻点名 QGPO/QGF/DAC，且追问为什么用有偏形式。对**应用向期刊**，应把 guidance 降级为"框架里一个组件"，靠"系统组合 + 跨建筑落地"立论。

---

## 5. 第三创新点的决策历史 —— **四个方向已否，都否得对**

| 方向 | 死因（必读，避免重走） |
|---|---|
| **A. 自适应引导标度**（按 critic 分歧/步数调 η） | 用户**已试，效果差**。 |
| **B-oracle. 用真值 A_d/B_d 做约束投影** | **作弊**（部署读 `env.A_d`）；且**和 MPC 基线完全重复**（MPC 已经是 oracle 模型控制器且更差）；**自打脸**期刊反-MPC 动机（"MPC 依赖模型精度"）。 |
| **B-fit. 最小二乘学线性模型再投影** | 不作弊但**平凡**：BEAR 真值线性，LS 秒辨识到机器精度，"学模型"无难度，"模型鲁棒性"实验得靠人为注入误差硬凑。用户敏锐指出"很容易学出来"= 对。 |
| **能耗解析梯度引导** | **平凡**：`∂E/∂a` 是按功率加权的符号向量、方向恒指向归零，本质是推理期 L1/L2 正则；"部署期 Pareto 调权重"这层包装也有邻居（preference-conditioned / multi-objective diffusion policy）。 |

**四个死因收敛到同一句话**：在 BEAR（线性系统 + critic 已把活干完）上，"**再加一路采样期梯度**"这条路整体到顶了 —— 继续找只会不断撞"作弊 or 平凡"。**这是 benchmark 物理性质决定的，不是能力问题。**

---

## 6. Meta 结论 & 推荐的贡献轴（本轮最重要的产出）

**别再在"guidance 做更强/更花"里找第三点。** 换轴。两个更可能藏着诚实新意、且数据大半已有的方向：

### 轴一（稳、性价比最高）：跨建筑泛化 + FNO 机制诚实化
- 把 OfficeMedium 提进主表，SchoolPrimary 训完补上 → 单建筑变三建筑，解决最大送审风险。
- **把 FNO 讲诚实反而变加分**：明说 OfficeSmall（6 区→rfft 长度 6→4 模态，`modes=4` 零截断）上谱截断**没在工作**，真正截断只在 OfficeMedium/SchoolPrimary 发生。于是"多建筑"不再是凑数，而是**"验证结构先验何时真正起作用（随分区数/耦合强度）"**——一个诚实、有深度的机制问题。m2/m4/m8 扫描在大建筑上重画即可。
- ⚠️ 修 **axis mismatch**：FNO 滤分区轴，但论文平滑性证据（mean|Δa|、PSD）测的是时间轴。要么补分区轴的空间谱证据，要么把因果链讲清（时间平滑其实来自残差门控 + MPC-BC + guidance）。

### 轴二（若想要"新机制"）：`mixture` 变体可能才是真落点
- 用户本周自己在探 `mixture / mixture_reg / hybrid_scalar`——**本轮未搞清是什么**。若是"多模态动作分布 / mixture-of-experts 去噪"，那可能比 guidance 更有区分度（扩散策略的多模态在 HVAC 尚未被系统研究）。**下一轮第一件事就是搞清它。**

### 目标定位提醒
用户明确："不追顶刊，只要自洽完整能发。" → 轴一足够撑一篇诚实的应用向期刊；轴二是"如果还想要一个亮点"的可选加成。**不要为了新机制去换 EnergyPlus**（用户论文明确因开销/同步不稳而选 BEAR）。

---

## 7. 开放问题（下一轮待查）

1. **`mixture` / `mixture_reg` / `hybrid_scalar` 到底是什么？** 谁驱动（`scripts/`?）、去噪器结构、动机。→ 决定轴二是否成立。
2. η 扫描、m2/m4/m8 的**具体数值**还在 TensorBoard event 里，未聚合。要不要抽成 CSV/图。
3. Guided full 模型评估时**残余违规还有多少**？（决定任何"再压违规"方向是否还有空间。）
4. SchoolPrimary 训练是否收敛、最终指标。
5. 期刊版 vs ICCC 版的**残差叙事矛盾**（ICCC 说残差 essential；期刊 Table II 里 NoRes 能耗最低、近 Pareto 并列）——须统一口径。
6. SAC 基线疑似没调好（5980 kWh，差 6–7×）——重调或明确解释。

---

## 8. 建议的下一步（按优先级）

1. **搞清 `mixture` 变体是什么**（读 `scripts/` + 对应 `log_building/*mixture*` 的 config/结果）。这是叉路口。
2. **聚合已有数据**：从 TensorBoard 抽 η 扫描、m2/m4/m8、多建筑数值，生成可进论文的表/图。低成本、高回报。
3. 定轴：若 mixture 有料 → 轴二为主、轴一为底座；若无 → 轴一独立成篇（诚实的多建筑 + 机制研究 + guidance 作为已验证组件）。
4. 写作层（不急，用户明确暂缓）：重定位 contributions、修 axis mismatch、统一残差叙事、补失显著性/seed。

---

## 9. 关键文件索引

- 论文：`docs/DiffFNO_ICCC2026_submission.pdf`（旧/会议）、`docs/GDMOPT.pdf`（新/期刊初稿）。
- guidance 实现：`main_building_fno_guided_bcfix_clean.py:273-290`（`build_guidance_fn`）、`diffusion/diffusion.py:175-182`（注入点）。
- FNO 去噪器：`diffusion/model_fno.py`。
- 环境/动力学：`BEAR/BEAR/Env/env_building.py`（step:229-272、动力学:250-251、A_d/B_d:184-185、能耗项、datadriven 拟合:420-422）。
- MPC 专家（已 oracle 读模型）：`BEAR/BEAR/Controller/MPC_Controller.py:14-15,68-70`。
- 主表/消融：`log_building/table_1m_metrics.csv`、`log_building/ablation_summary.csv`。
- 图套件脚本：`scripts/paper_*.py`、`scripts/benchmark_inference_latency.py`。
