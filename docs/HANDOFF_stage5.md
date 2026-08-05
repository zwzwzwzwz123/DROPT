# HANDOFF 阶段五 —— 干净快照（DiffFNO 期刊版）

> 📑 全部 docs 的角色分类见 `docs/README.md`（活档/活引用/封存索引）。本文件是唯一"现状+待办"活档。
> 创建 2026-07-19。接续 `HANDOFF_stage4_alignment.md`（阶段四，结构对齐 + 训练超参统一决策，已闭环）。
> **本文件只讲当前状态 + 待办，不含推导史。** 阶段四的洼地/探针权衡/结构对齐/循环论证纠错等历史见旧文件（§5 封存指针），旧文件原地保留不删。
> 目标（用户原话）：不追顶刊，内容自洽、故事完整、能发表即可。投稿靶待定（建筑能源类 Energy and Buildings / Applied Energy vs IEEE IoT-J，定稿前比）。语言：中文。
> 会议版基线：DiffFNO ICCC 2026-07（单建筑 OfficeSmall、无 guidance），本项目相对增量见 `docs/INCREMENT_over_conference.md`。

---

## 0. 一句话现状

**阶段四核心决策全部闭环**：三建筑结构统一(L1/modes4)、训练超参统一默认档(选 a)、OfficeMedium 洼地消除→省幅曲线单调、残差重构为 guidance×residual 交互效应（非"改口成惰性"，见 §2 残差条）。
**当前叙事主承重 = 支柱1(FNO 跨规模稳赢) + 骨干/引导解耦**。次承重"单调增强"当观察不当定律。
**已完成**（07-22 更新）：✅ **OfficeMedium NoGuide 3-seed 全完训**（s42/s0/s1，默认档 8048.4±173.7 / 每区率 37.2%），guidance 解耦三点齐——见 §2 骨干解耦条（guidance 省幅单调 3%→13%→22%，骨干独立效应非单调）。**SAC 基线 18+1 run 全完训、6 组 3-seed 聚合已出**（含 medium mpc s42 补跑，见 §3 SAC 行）。
**✅ modes 扫描 School 全完（08-02）**：m2s1/m8s1 补种完训（07-31/08-01），3seed×3modes 表对称达成，**判决最终确认**。
**🟢 modes 扫描 School 3-seed 最终判决（08-02，决定性）**：m2/m4/m8 各 3seed 完整。**判决=边界 null**：
  - **违规率铁平**（每区率 m2/m4/m8 = 28.4/28.3/28.1%，跨全部9个种子稳）——**硬结果**。
  - **能耗弱单调掉进噪声**：组均 6414±43 / 6418±22 / 6449±10，m8−m2 仅 35kWh(0.55%)，差/std比 1.32（远<2），误差棒重叠。补种后效应进一步收窄（2-seed时0.67%→3-seed时0.55%），与预判完全一致。
  - **写作定论**：⚠️ **别写"截断=省能耗机制"**（0.55%+误差棒重叠=过度包装、会自伤）。能用的是**稳健性**——性能跨 15%→62% 保留率仅摆 0.55%，坐实 **modes=4 非调参刀尖值**（支柱1 防御点）。详见 memory `guided-difffno-modes-sweep-school`。
**🔴 07-24/25 OfficeLarge 已放弃(结论;max_power 教训见 §4 纪律+memory)**：尝试加 OfficeLarge(23区)作第四栋,但 8000W 全套崩。根因=`max_power=8000W` 对 OfficeLarge 欠配(真实动力学 oracle 确证:完美控制器在 8kW 都到 71%违规,School 8kW=0%)。**但 max_power 是 U 形非"越大越好"**(实训 20kW=48%、32kW=48%、64kW=60%——任何全局标量都卡~48%,到不了 School 水平)。唯一能到 ~21%(≈School)的是**逐区功率**,但需改共享 wrapper→**违反零侵入,否决**。**故放弃 OfficeLarge,支柱2 维持三栋(Small→Medium→School)**。全部 OfficeLarge 数据/脚本已删。
**支柱2 现状**：维持已确证三栋(省幅 12.4→19.2→52% 单调),仍是"仅3点、跨楼混杂"的观察(见 §2)。⚠️ **modes 扫描已出 null(违规扁平/能耗0.55%噪声，3-seed最终),无法从同楼因果加固支柱2**；可用价值=支柱1 稳健性防御(modes=4非调参)。支柱2 只能维持"观察"定调,不宣称严格单调律。
**待办首推**：无硬待办。modes 扫描 3×3 全满（08-02），科学结论最终确认（边界 null + 稳健性）。**主线正式转写作**（把 modes 稳健性写进支柱1 防御 + 违规扁平当硬结果）。✅ **08-02 出表数据已备齐并全量审计通过**：§1.1 全指标表（四指标×全方法×三栋，含 SAC reward/每区率、Medium NoGuide）可直接填 LaTeX；Medium NoGuide 已并入 master CSV（11 行全）；`Mean |Δa|` 已否决（§3 表）。可选实验：OffMed 跨楼 modes 验证 / 边界点(modes=1/满谱) / 补 Medium-School 的 NoRes 臂凑三楼 2×2——均非必需。

---

## 1. 【确证】三建筑主表（全 3-seed，1M，guidance0.5，末段 8 点窗均值 mean±std）

数据源（单一真相）：`paper_figures_v2/master_metrics_v2.csv`（OfficeSmall/School）+ `scripts/_extract_medium_default.py`（OfficeMedium 默认档）。违规跨楼比较用「每区违规率」=avg_violations/区数。

| 建筑 | 区数 | 结构 | Guided-DiffFNO | Diff-MLP | 能耗省 | 违规降 |
|---|---|---|---|---|---|---|
| OfficeSmall | 6 | L1/m4/w48 | 871±3 / 每区率8.3% | 994±35 / 23.8% | 12.4% | 65% |
| OfficeMedium | 18 | L1/m4/w64 | 6985±13 / 20.5% | 8645±430 / 38.0% | 19.2% | 46% |
| SchoolPrimary | 25 | L1/m4/w128 | 6418±22 / 28.3% | 13368±1345 / 68.4% | 52.0% | 59% |

> std 口径：全项目统一**样本 ddof=1（除 n-1）**（2026-07-20 定）。CSV 已重生成，旧 ddof=0 值（Small MLP ±28、School MLP ±1098 等）作废，换算 ×√(3/2)。均值不变。

- **三栋统一**：结构 L1+modes4；width 随 state_dim 分化 48/64/128（容量旋钮，非训练超参，不统一）；训练超参三栋全默认档、无逐楼调参。
- **省幅曲线 12.4%→19.2%→52.0% 单调递增**（默认档统一后，无洼地）。

### 1.1 论文出表用全指标表（08-02 补齐，四指标 × 全方法 × 三栋）

> 所有数字均 3-seed 末段 8 点窗 mean±std（ddof=1），已与 CSV 反算自洽（省幅/违规降/每区率全部核对通过）。
> **reward 跨楼不可比**（zone 数与 episode 长度不同致 scale 异），只在同楼内比较；出表时加脚注或仅在单楼消融表里放该列。

| 建筑 | 方法 | 能耗(kWh) | 违规(原始) | 每区率% | comfort | reward |
|---|---|---|---|---|---|---|
| Small(6) | **Guided-DiffFNO** | **870.8±3.1** | 0.496±0.059 | **8.3** | 0.460±0.016 | −0.487±0.035 |
| Small | NoRes（有引导） | 867.2±3.1 | 0.597±0.066 | 10.0 | 0.488±0.029 | −0.548±0.042 |
| Small | NoGuide | 900.4±26.1 | 0.865±0.096 | 14.4 | 0.557±0.021 | −0.726±0.067 |
| Small | NoRes+NoGuide | 868.2±5.7 | 1.076±0.417 | 17.9 | 0.614±0.090 | −0.840±0.254 |
| Small | Diff-MLP | 994.0±34.7 | 1.426±0.580 | 23.8 | 0.688±0.146 | −1.097±0.364 |
| Small | SAC+MPC | 2493.1±227.3 | 4.210±0.122 | 70.2 | 3.347±0.054 | −3.284±0.112 |
| Small | SAC | 5203.8±169.8 | 5.620±0.089 | 93.7 | 9.077±1.090 | −5.098±0.154 |
| Medium(18) | **Guided-DiffFNO** | **6984.6±13.3** | 3.687±0.241 | **20.5** | 0.636±0.021 | −2.977±0.144 |
| Medium | NoGuide | 8048.4±173.7 | 6.703±0.479 | 37.2 | 1.011±0.088 | −4.997±0.305 |
| Medium | Diff-MLP | 8644.6±430.1 | 6.842±0.940 | 38.0 | 1.141±0.195 | −5.163±0.640 |
| Medium | SAC+MPC | 13754.0±127.2 | 13.248±0.381 | 73.6 | 3.097±0.289 | −9.788±0.277 |
| Medium | SAC | 17355.5±208.6 | 15.837±0.814 | 88.0 | 6.761±1.448 | −12.193±0.660 |
| School(25) | **Guided-DiffFNO** | **6417.6±21.8** | 7.084±0.274 | **28.3** | 0.760±0.018 | −4.887±0.166 |
| School | NoGuide | 8267.6±677.9 | 12.748±0.245 | 51.0 | 1.562±0.282 | −8.611±0.259 |
| School | Diff-MLP | 13367.9±1344.8 | 17.110±0.829 | 68.4 | 3.064±0.671 | −12.030±0.740 |
| School | SAC+MPC | 21849.5±407.0 | 22.463±0.191 | 89.9 | 6.878±1.439 | −16.587±0.441 |
| School | SAC | 24861.9±222.1 | 23.563±0.239 | 94.3 | 11.062±1.086 | −18.264±0.303 |

- **消融臂完整度**：OfficeSmall 4 臂全（Full/NoRes/NoGuide/NoRes+NoGuide）；Medium/School 仅 Full+NoGuide+MLP（**缺 NoRes、NoRes+NoGuide**，各需补 2 run）。故 2×2 消融表只能是**单楼(OfficeSmall)版**。
- ⚠️ **NoRes 能耗 867.2 < Full 870.8**（低 0.4%，std 重叠）：写作时主动交代这是**能耗-舒适权衡**（残差以微弱能耗代价换 1.7pp 舒适），别让审稿人当异常。
- **SAC 数据源**：`scripts/_extract_sac_baselines.py`（每区率已内置）；**reward 列该脚本未抽**，08-02 用临时内联脚本补出（数字已录于本表，如需复现按 `test/reward` 末 8 窗抽即可）。
- **Medium NoGuide 数据源**：`officemedium_fno_noguide_default_1m_s{42,0,1}_*`，⚠️ s1 须用真身 `..._20260721_195310`（`..._095710` 是 0-eval stale）。✅ **08-02 已注册进 `extract_master_metrics.py` 的 REG 并重跑，Medium NoGuide 现已在 `master_metrics_v2.csv`（11 行全）**，单一真相源不再有缺口；独立脚本 `_extract_medium_default.py` 抽出的值与 CSV 完全一致（交叉验证通过）。
- ✅ **08-02 全量审计通过**：CSV 11 行完整（每行 n_seed=3）；handoff 所有派生百分比与 CSV 反算逐条自洽——省幅 12.4/19.2/52.0、违规降 65/46/59、骨干 9.4/6.9/38.2、guidance 省 3.3/13.2/22.4、guidance 违规降 42.7/45.0/44.4、School 骨干降违规 25.5、残差交互 3.5pp/1.7pp（精确复现）；§1.1 全指标表 11 行 × 5 列逐格与 CSV 比对无差。
- ⚠️ **违规口径**：`avg_violations`=每时刻超±1°C 容差的区数再对时间平均（被区数放大），跨楼必须归一化成每区率。禁与老表 `table_1m_metrics.csv` 的比率混用。

---

## 2. 支柱承重排序（写作时对着看，别把次承重当主论点）

### 主承重（硬，不依赖任何在跑实验，已够发一篇扎实应用向论文）
- **支柱1｜FNO 跨规模稳赢 MLP**：省 12.4/19.2/52.0%，3-seed std 极小。全文地基。⚠️「参数少一量级」只对 OfficeSmall(6.8×)成立，School 已追平，别外推三栋。
- **分析贡献｜骨干 vs 引导干净解耦**（真正的智识增量）：School NoGuide vs MLP 纯骨干省 38%/降违规 25%，独立于 guidance、同骨干同超参不含 MLP 混淆。顶回「就是多测几栋」。✅ **【07-22 三点齐】OfficeMedium NoGuide 3-seed 完训补齐**（8048.4±173.7 / 每区率 37.2%）。三栋成分分解（数据 INCREMENT §3.1/§3.2）：
  - **骨干独立省能耗非单调**：Small 9.4% → Medium **6.9%** → School 38.2%（Medium 骨干纯效应最弱）。
  - **guidance 能耗省幅单调**：Small **3.3%** → Medium **13.2%** → School **22.4%**（三点单调递增，正面观察；违规降幅 **42.7 / 45.0 / 44.4%**，三栋均 40%+ 但**非单调**，别写成"也随规模递增"）。
  - ⚠️ **诚实读法（写作要点）**：成分贡献**场景依赖**——Medium 总省 19.2% 里 guidance 主导（骨干~7%/引导~13%），School 总省 52% 里骨干主导（~38%）。**别把"纯骨干跨规模都强"写成通则**（Medium 骨干只 6.9%）；解耦贡献点仍成立（School 骨干 38% 干净漂亮），但要如实说成分谁主导随楼变。这与残差×引导交互的"成分场景依赖"同调。

### 次承重（有价值，写成「观察」不写成「定律」）
- **支柱2｜FNO 优势随区数单调增强**（12.4→19.2→52%）：叙事红利大，但软肋硬——① 单调部分由「大楼 MLP 崩得更狠」驱动、非全是 FNO 功劳（School MLP 违规 68%）；② 仅 3 点、「规模」是混杂变量（楼型/热质量都不同）；③ 缺同楼受控证据。**写法**："观察到随区数增大趋势，可能成因耦合复杂度；样本仅三栋、部分源于大楼 MLP 退化，不宣称严格单调律。" 防线=「三栋统一协议无逐楼调参」。要变重只能靠 §3 modes 扫描。

### 半支柱（建议降为 discussion/limitation）
- **支柱3｜谱结构物理机制（axis mismatch）**：会议版「谱截断=平滑先验」测的是时间轴、FNO 滤分区轴，因果链有裂缝。现有一条确证（FNO 压跨区空间方差~3×）+ 一条存疑（归一化谱未必更低通）。别硬撑成支柱。

### 诚实成分刻画（负面结果=加分，防过度包装）
- **残差 = guidance×residual 交互效应（非"惰性/趋零"）**：⚠️ **不要把会议稿"essential"改口成"残差没用"（上一版病根，过头）。** 正确定调 = 条件命题："*Guidance subsumes, rather than invalidates, the residual*"——会议版无引导时残差 essential；引入 critic guidance 后 guidance 部分吸收残差纠偏职能。硬数据：残差舒适收益 **3.5pp（无引导 14.4% vs 17.9%，精确复现会议数字）→ 1.7pp（有引导 8.3% vs 10.0%）**，被吸收一半、**非归零**。checkpoint 自压残差到 ~5% 贡献是**间接证据**（抑制靠残差权重近零），但别外推成"惰性"。⚠️ **别写"门控随 guidance 自衰减"**——跨 checkpoint 核实 gate 全程 sigmoid≈0.50–0.58、不随 guidance 变（Small 引导 0.508 vs 无引导 0.510；School 0.516 vs 0.576），该说法 07-21 证伪。memory `guided-difffno-residual-inert` 的旧"惰性"口径已被此交互叙事取代。
  - **写作策略（07-21 从 INCREMENT §6.3 移入，INCREMENT 现只留客观数据）**：
    - **Method**：residual 描述改中性 —— *a lightweight local-correction path whose contribution depends on whether value guidance is active*。
    - **Experiments**：设小节 *"Interaction between Critic Guidance and the Residual Pathway"*，放 2×2 消融表（数据见 INCREMENT §6.1）。⚠️ **别放门控 g 轨迹图，也别讲 gate 衰减**——① `residual_gate` 从未写进 TensorBoard，只有 checkpoint 值；② 更关键，跨 checkpoint 值本身就是平的（sigmoid 全程 0.50–0.58、不随 guidance 变），画出来也是平线、无故事。残差被压制的机制写"残差权重训练后近零（缩小10-20×）"，不要写成 gate。
    - **与会议版主动和解段**（防审稿人抓"与自己已发表结论矛盾"）：*In our conference version [X], where actions were sampled without value guidance, the residual pathway was essential for comfort compliance. The 2×2 ablation refines this picture: critic guidance subsumes most of the residual's corrective function, reducing its marginal comfort benefit from 3.5 to 1.7 percentage points. The two mechanisms are thus partially substitutable comfort regulators rather than independent modules.*
    - 定位：这是一个 interaction effect，作期刊版分析贡献点（非"负面结果"、非"推翻会议结论"）。
- **guidance**：主要压违规、不一定省能耗（**三栋实数**：能耗省 Small 3.3% / Medium 13.2% / School 22.4%；违规降 42.7 / 45.0 / 44.4%，均 40%+ 但非单调），按楼分述不外推；是 QGPO 批评的朴素 point-estimate 有偏实现→论文加一句诚实交代即可，不当创新点。
  - ⚠️ **guidance 的定位（08-02 澄清，别搞错）**：guidance 是**期刊版相对会议版最大的方法增量**（会议版完全没有），但**刻意不单独立成支柱**——若吹成核心创新，一句"这就是有偏 classifier guidance，QGPO 早批过"即可打穿。**正确写法 = guidance 的正面价值（能耗省幅单调 3.3%→13.2%→22.4%、违规降 40%+）通过「骨干/引导解耦」这根分析支柱来承载**：解耦既展示 guidance 有效，又用成分分解（谁主导随楼变）体现智识深度，比裸吹 guidance 安全得多。

---

## 3. 待办（按性价比）

| 优先级 | 项 | 作用 | 状态 |
|---|---|---|---|
| ✅ 完成 | **大楼 modes 扫描 m2/m4/m8**（School，同楼受控） | 把支柱2/3 从跨楼相关升**同楼机制因果**（谱轴受控） | ✅ **3seed×3modes 全满（08-02），判决=边界 null（决定性）**。配置核对+漂移检验：`main_building.py`02-13/主脚本03-23 均早于 anchor 跑期→走默认值核心超参无漂移→单变量成立；只改 --fno-modes。**违规铁平**（每区率 m2/m4/m8=28.4/28.3/28.1%，跨全9种子稳，硬结果）；**能耗弱单调掉进噪声**（组均 6414±43/6418±22/6449±10，m8−m2=35kWh 0.55% < m2 组内 std 43，差/std 比 1.32，误差棒重叠。补种后 2-seed 0.67%→3-seed 0.55% 进一步收窄）。**写作定论**：别写"截断=机制"（过度包装），用**稳健性**（跨15%→62%仅摆0.55%→modes=4非调参）当支柱1防御。判决/抽数脚本 `scripts/extract_modes_sweep.py`（08-02 由临时脚本转正留档，含判决摘要注释；画 3×3 稳健性图或应对审稿人追问 modes 选择依据时复用）。OffMed 跨楼/边界点(modes=1/满谱)可选非必需 |
| ❌ 放弃 | ~~OfficeLarge 新建筑（23区）~~ | 原为拆支柱2 楼型混杂变量(office 规模序列 + Large-vs-School 楼型对照) | **07-25 放弃**。8000W 崩、根因=max_power 欠配;但 max_power 是 U 形(实训 20/32kW 均~48%、64kW 更差),任何全局标量到不了 School 水平;唯一可行的逐区功率需改共享 wrapper 违反零侵入。**性价比与可辩护性都差,放弃,支柱2 维持三栋**。数据/脚本已删。教训见 §4 max_power 纪律 |
| ✅ 完成 | **OfficeMedium NoGuide 默认档 3-seed** | guidance 增量第三点（补解耦缺口），看 guidance 侧随规模是否单调 | 07-22 全完训（s42/s0/s1，245ep）。抽数（`_extract_medium_default.py` 已加 NoGuide 组）：**能耗 8048.4±173.7 / 每区率 37.2% / comfort 1.011**。已填 §2 骨干解耦条 + INCREMENT §3.1/§3.2。**结论**：guidance 能耗省幅单调（3%→13.2%→22%）；骨干独立效应非单调（9.4%→6.9%→38.2%，Medium 最弱、省幅靠 guidance 主导）。s1 真身目录 =`..._20260721_195310`（`..._095710` 是 0-eval stale） |
| ✅ 完成 | **SAC/SAC+MPC 基线 3-seed 聚合** | 发表门槛（尤其 School 基线曾缺） | 07-20 完成（含 medium mpc s42 补跑）。18+1 run 全完训，6 组均 n=3。三栋排序 Guided-FNO<Diff-MLP<SAC+MPC<纯SAC。SAC+MPC:small 2493±227/medium 13754±127/school 21850±407；纯SAC:small 5204±170/medium 17356±209/school 24862±222。RL 每区率70-95%守不住舒适。`scripts/_extract_sac_baselines.py` |
| ❌ 否决 | ~~Mean \|Δa\| 动作平滑度指标~~ | 会议版 Table II 有此列（撑"谱截断=平滑先验"） | **08-02 否决**（三条理由）：① **叙事依托已消失**——该列原为支柱3 服务，支柱3 已因 axis mismatch 降为 discussion；② **数字帮倒忙**——会议版 MPC \|Δa\|=0.0130 最平滑却能耗最高/违规最多，审稿人会反问"既平滑为何最差"，要花笔墨解释非论点内容；③ **成本回报不对称**——event 里**只有 `update/action/action_*`（训练期 batch 统计），无 test 轨迹动作差分**，需为全方法×三栋从 checkpoint 重跑专门评估脚本。**替代防线**：若审稿人问动作抖动，用 reward 列（已隐含稳定性）+ "扩散采样本身即正则化"一句话挡回 |
| 🥉 低 | Conv1d 去噪器对照 | 防新颖性质疑（FNO 赢靠谱结构还是任意空间先验） | 未开始，需改代码 |
| 🥈 中 | **guidance 训练期 vs 推理期贡献三点分解** | 答 stage4 §6.2-5c 开放问题(稳定器 vs 推理倾斜)，诚实刻画 guidance | 🔧 脚本已备 `scripts/_guidance_decompose.py`（reload checkpoint→复现test协议→NoGuide/Full-推理η0/Full-推理η0.5）。⚠️**朴素4点η扫描已否**(η>0.5=critic off-support污染、η=0≠NoGuide)。⚠️**event 里无多-η数据**(旧 handoff"数据在event"错，只有单次训练指标)，须靠 reload 推理脚本。含首跑验证闸门。等 GPU 空 |
| 🥉 低 | diffusion_steps 扫描 / 分区轴空间谱补 seed | 附录、堵必问 | 未做 |

- **别做**：PPO/TD3 等新 RL 基线（无底洞、不服务核心论点）；guidance Level-2 新网络（永不做）。

### 3.1 已生成论文图（`paper_figures_v2/`，均默认档新数据，07-19）
- **主管线** `scripts/gen_paper_figures_v2.py`（读 `master_metrics_v2.csv`）：fig1 三楼能耗、**fig2_saving_curve_monotonic**（12.4→19.2→52 单调，已换掉旧洼地图）、fig3 每区违规率、fig4 骨干/引导解耦、fig5 OfficeSmall 消融柱状、fig6 参数效率、fig7 comfort、fig8 能耗-违规散点、fig9 School 训练稳定性。
- **追加** `scripts/gen_extra_figures.py`（复用 REG 目录同源）：**figA 训练奖励曲线 FNO vs MLP×三楼**（3-seed 均值±std，FNO 全程稳、MLP 楼越大越崩，强图）；**figB OfficeSmall 消融 2×2 热力图**（残差×引导，能耗+违规，示"引导压违规、guidance 部分吸收残差纠偏：残差舒适收益 3.5pp→1.7pp"）。
- ⚠️ **数据源已更新**：`extract_master_metrics.py` 的 OfficeMedium Full/MLP 已指向默认档目录（旧手调档作废）。重画任一图前先跑 extract 再跑 gen。
- ⚠️ **现有 fig1-9/figA/figB 是 07-19 从旧 ddof=0 CSV 画的**（误差棒偏小）。CSV 已于 07-20 重生成为 ddof=1，**重画后误差棒会 ×√(3/2)≈1.22**。均值不变，不急重画，等图定稿(投稿靶定后)一并重跑。
- **待补图（数据未齐/未画）**：**SAC 基线对比（数据已齐可画：见 SAC_QUEUE 6 组 3-seed）**、**modes 扫描图（3seed×3modes 数据已全齐，可画 3×3 方阵稳健性图：能耗组均 6414/6418/6449、违规率 28.4/28.3/28.1%）**、消融热力图扩三楼（需 OffMed/School NoGuide+NoRes）。

---

## 4. 关键纪律 / 踩过的坑（压缩版）

- **07-21 新教训：gate 全程≈0.5、不随 guidance 衰减**——跨 checkpoint 核实证伪（Small 引导 0.508 vs 无引导 0.510，School 0.516 vs 0.576）。残差抑制靠**残差权重训练后近零（缩小 10-20×）**，非 gate 关闭。别把"门控随 guidance 自衰减"写进论文（已从 `gen_speech.py` + `INCREMENT_over_conference.md` 清除）。见 memory `guided-difffno-residual-inert`。
- **抽默认档数据认准 `officemedium_*_default_1m_*` 目录**。曾误用 `officemedium_aligned_tailmean.py`(RUNS 硬编码手调档目录)把手调档当探针=循环论证。用 `_extract_medium_default.py` / `_verify_probe_vs_aligned.py`（每 run 独立 EventAccumulator 防缓存串味）。
- **RAM 错峰**：16GB 机，FNO/MLP 1M buffer ~2.5GB/重进程、SAC 200k 轻。"安全上限约 6 重进程"是**低基线时**（07-17 测，3SAC+2FNO+1MLP=4GB空闲）。⚠️ **07-20 新教训：当前高基线（VS Code/浏览器等占 ~10GB）下，2 个满-buffer FNO 就装不下**——实测 2FNO→RAM 1838MB 逼近 thrash，已杀一个。**FNO 多 seed 必须串行**（一个完训再启下一个），或先关常用程序腾 RAM。启新 run 前必查 RAM、单 FNO 需 RAM>4-5GB。
- **完训判据**：run 总轮数 = **245 epoch**（非 122，那是 SAC）；末段窗取 8 点均值。
- **🔴 中断后重跑而非续训（07-28 定）**：run 被重启/杀断后，**从头重跑、别用 `--resume-path` 续**。续训丢 replay buffer/优化器动量/RNG 状态，训练轨迹与从空 buffer 一口气跑满的 anchor 不再同协议→失去单变量可比性、引入混杂。seed 固定时重跑确定性可复现，只亏已跑的算力。残档目录+日志先改名 `.interrupted_epN`（防按前缀抽数撞车、重演 OffMed 0-eval stale 坑），确认无用再删。**RAM 满载的 FNO 跑期间电脑重启会中断整链**，长跑前存好其他工作。
- **违规口径**：每区违规率，见 §1。
- **🔴 OfficeLarge max_power 纪律（07-24 定，血泪教训，最重要）**：`max_power`(单台/单区 HVAC 功率上限，BEAR 官方定义 utils_building.py:398)默认 **8000W 对 OfficeLarge 物理欠配**，是首批全套崩溃(comfort~9.5/每区率~100%/FNO塌零动作)的**唯一根因**。证据：全年8759时刻逐区稳态解析(达26°C所需动作×8000=物理需求功率)——**School 最费区仅7890W<8000(0/25饱和,故能训);OfficeLarge 11/23区超8000W**(BASEMENT 75225W=9.4倍/全年91%时刻不够、3个PLENUM各~30000W、DATACENTER_BASE 25767W、6外围区~10400W)。**机制**：功率不够→守舒适要满饱和大动作→BEAR能耗惩罚(∝原始动作)碾压→策略塌成零动作局部最优(ep5就崩)。**真实动力学 oracle 确证 8kW 欠配(非 lstsq 假象)**:1-步贪心最优控制器(完美模型、真实动力学、动作截断可行域)全年违规底——**School 8kW=0%(完美可控,故能训);OfficeLarge 8kW=70.9%(4.5区饱和)**。即便完美控制器在 8kW 都守不住。(早先 lstsq"BASEMENT 需75kW"是数值夸张,方向对。)
  - **🔴 max_power 是 U 形,不是"越大越好"(核心教训)**:两股力打架——功率↑→oracle结构底↓(8k:71%→20k:39%→32k:~15%→64k:~0.4%),但功率↑→训练策略噪声被放大↑(扩散策略采样随机,max_power放大其温度扰动→过冲)。**净效果实训 U 形:32kW=48%、64kW=60%(反而更差!)、带噪声oracle 20kW最低~36%**。教训:定 max_power **不能只看无噪声 oracle 结构底,必须计策略噪声被增益放大;别一路往上加**。有真实 HVAC 依据(设备过大→短循环/控制不稳)。
  - **更深根因=区异质性**:各区功率需求差~8倍(占用办公区~9-13kW,BASEMENT/3PLENUM/DATACENTER_BASE 这5个非办公服务区需33-48kW)。全局标量无论取多少都没法同时合身(小了地下室够不到,大了易控区被噪声抽抖)。
  - **逐区方案最优但已否决(零侵入)**:"只给最难5区按设计日负荷配、余18区保持8000"→带噪声oracle ~21%(≈School,最好且偏离最小)。**但实现需改 `env/building_env_wrapper.py:419` 能耗计算(数组max_power会崩)——该文件所有训练共享,改它违反零侵入,用户拍板否决**。
  - **最终结局=放弃 OfficeLarge(07-25)**:全局 20kW 实训 = 48.5% ≈ 32kW 的 48%(带噪声 oracle 预测的 36% 又落空——第三次单一分析代理误导)。**任何全局标量都卡~48%、到不了 School 水平;唯一可行的逐区功率违反零侵入**。硬凑违规48%+特殊功率的第四栋,性价比与可辩护性都差→放弃,支柱2 维持三栋。数据/脚本已删。
**非调参**：max_power 是 plant-sizing 物理选型;能耗可比性不破(energy=|action|×max_power,物理制热量固定→读数不随 max_power 变)。⚠️ **纠正旧误解**:卡住是**制热不够**(Tucson零动作漂15°C要升到26,地下室贴地基/回风腔散热快),**不是"DATACENTER高内热需制冷"**。⚠️ **零侵入原则**=不改共享文件(wrapper/config/主脚本),要改就新建脚本或只用CLI参数。memory `guided-difffno-officelarge-maxpower-rootcause`。
- **OfficeLarge 规格(已放弃,仅存档备查)**:BEAR 自带正牌 DOE 参考楼(`ASHRAE901_OfficeLarge_STD2019_Tucson`),roomnum=23/state_dim=71,width 本拟 96。**因 max_power 无全局解 + 逐区违反零侵入,07-25 放弃**(见上条 max_power 纪律)。数据/日志/launcher(`_large_launch.sh`/`_large_rolling.sh`)均已删。若日后重启:根因是区异质性(各区功率需求差8倍),唯一可行解是逐区功率(需先解决 wrapper 数组能耗计算的零侵入实现,如新建独立 wrapper 子类)。
- **rolling**：`scripts/_sac_rolling.sh`（MAX_PARALLEL=3，含 medium 队列）；停 = `touch run_logs/_rolling.STOP`。

---

## 5. 封存指针（历史推导，需要时查旧文件）

- **阶段四** `HANDOFF_stage4_alignment.md`：结构对齐(L2/m6→L1/m4 无损)、训练超参统一 (a)/(b) 决策全过程、洼地为何是假象、探针权衡纠错、循环论证事故复盘、支柱2 旧「不赌单调/共同洼地」reframe 推导史。
- **阶段三** `HANDOFF_option3_bear_journal.md`：三建筑推导史、耦合结构假说、guidance 发现史。
- **SAC 基线** `HANDOFF_sac_baseline.md`：诊断 + 公平协议。
- **增量对比** `docs/INCREMENT_over_conference.md`：相对会议论文的逐项增量 + 数据。
- **残差核查** memory `guided-difffno-residual-inert`；**OffMed 统一决策** memory `guided-difffno-officemedium-config-unify`。
- **队列真相源** `run_logs/SAC_QUEUE.md`。

