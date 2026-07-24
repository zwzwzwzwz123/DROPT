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
**在跑**：OfficeLarge full s42（Phase1 探针，**32000W 修复版**，07-24 14:18 发车，~15h）。
**🔴 07-24 重大发现·OfficeLarge 8000W 崩溃根因已定位并修复**：首批 8000W 全套(full s42/mlp s42/s0)**双双崩溃**(FNO/MLP comfort~9.5、每区违规率~100%、FNO 塌成零动作能耗1595)。**根因=`max_power=8000W`(单区功率上限)对 OfficeLarge 物理欠配**：全年8759时刻逐区解析显示 **11/23 区峰值需求超8000W**(BASEMENT 75225W=9.4倍/全年91%时刻不够、3个PLENUM ~30000W、DATACENTER_BASE 25767W),满功率都到不了26°C→任何策略必崩。**非楼不可控、非训练超参问题**。对照 School 最费区仅7890W<8000(0饱和)→故只此楼暴露。**Fix=`--max-power 32000`**(BEAR 原生支持逐楼设,是 plant-sizing 物理参数非调参;32000后所需动作mean0.21落回三栋同档、能耗可比性不破)。8000W 崩配旧run已删,32000W 探针已发并逐项核对配置对齐(见 §3 OfficeLarge 行 + §4 max_power 纪律 + memory `guided-difffno-officelarge-maxpower-rootcause`)。
**新增方向（07-22 定案）**：**加 OfficeLarge（23区）新建筑**，拆支柱2 的楼型混杂变量——office 内部受控序列 Small→Medium→Large + Large-vs-School 近规模楼型对照。（07-24 因 max_power bug 全套推倒重跑，见上条。）
**待办首推**：OfficeLarge 32000W 探针完训验证信号 → 铺开剩14 run；大楼 modes 扫描（同楼谱轴受控）。两者互补，都把支柱2/3 从跨楼相关升成受控证据。

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
- ⚠️ **违规口径**：`avg_violations`=每时刻超±1°C 容差的区数再对时间平均（被区数放大），跨楼必须归一化成每区率。禁与老表 `table_1m_metrics.csv` 的比率混用。

---

## 2. 支柱承重排序（写作时对着看，别把次承重当主论点）

### 主承重（硬，不依赖任何在跑实验，已够发一篇扎实应用向论文）
- **支柱1｜FNO 跨规模稳赢 MLP**：省 12.4/19.2/52.0%，3-seed std 极小。全文地基。⚠️「参数少一量级」只对 OfficeSmall(6.8×)成立，School 已追平，别外推三栋。
- **分析贡献｜骨干 vs 引导干净解耦**（真正的智识增量）：School NoGuide vs MLP 纯骨干省 38%/降违规 25%，独立于 guidance、同骨干同超参不含 MLP 混淆。顶回「就是多测几栋」。✅ **【07-22 三点齐】OfficeMedium NoGuide 3-seed 完训补齐**（8048.4±173.7 / 每区率 37.2%）。三栋成分分解（数据 INCREMENT §3.1/§3.2）：
  - **骨干独立省能耗非单调**：Small 9.4% → Medium **6.9%** → School 38.2%（Medium 骨干纯效应最弱）。
  - **guidance 能耗省幅单调**：Small ~3% → Medium **13.2%** → School ~22%（三点单调递增，正面观察）。
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
- **guidance**：主要压违规、不一定省能耗（Small ~3%/School 22%/OffMed 待测），按楼分述不外推；是 QGPO 批评的朴素 point-estimate 有偏实现→论文加一句诚实交代即可，不当创新点。

---

## 3. 待办（按性价比）

| 优先级 | 项 | 作用 | 状态 |
|---|---|---|---|
| 🥇 高 | **大楼 modes 扫描 m2/m4/m8**（OffMed/School，同楼受控） | 把支柱2/3 从跨楼相关升**同楼机制因果**（谱轴受控），是让次承重/半支柱变硬的关键杠杆之一 | 🔧 脚本已备 `scripts/_modes_sweep_launch.sh`（配置核对匹配主表 Full，只改 --fno-modes；rfft 越界 guard）。m4 已有 3-seed，只补 m2/m8。**发车方案**：先 School 单 seed 筛信号(`school 2 42`/`school 8 42`)，有信号再补 s0/s1+OffMed。等 RAM（每点整训 245ep） |
| 🥇 高 | **OfficeLarge 新建筑（23区）全套** | 拆支柱2 的**楼型混杂变量**：①office 内部受控规模序列 Small(6)→Medium(18)→Large(23)，建筑类型固定只变规模；②Large(23) vs School(25) **近规模楼型对照**，直接答"School 52% 是因为大还是因为是学校"（现缺这个隔离楼型的对照点） | 🏃 **32000W 修复版 full s42 探针在跑（07-24 14:18 发车）**。🔴 **07-24 推倒重来**：首批 8000W 全套(full s42/mlp s42 完训 + s0 中途)**全崩**——`max_power=8000` 对 OfficeLarge 物理欠配(11/23区峰值超8000W,详见 §4 max_power 纪律)。已杀 s0、删 8000W 崩配旧run、rolling 已 STOP。**launcher `scripts/_large_launch.sh` 已改 `--max-power 32000`**(OfficeLarge 专属,三栋不动),头部注释含根因。✅ **配置逐项核对**(2026-07-24)：探针 12 项训练超参(L1/modes4/g0.5/bc0.8/final0.1/decay150000/vp10/actor1e-4/critic2e-5/batch256/update0.5/buffer1M)与主表三栋**逐字节一致**,只有 building_type/width96/**max_power32000**/seed 这 4 项该不同的不同(actor_lr 靠 main_building.py:158 硬编码1e-4 兜底,building_config 那个 3e-4 常量未接上、无关)。✅ 冒烟已过(action_mean0.22 不塌零=陷阱解除)。**下一步**：探针完训抽末8窗验证 comfort 从9.5回落~1、省幅转真实节能 → 信号回正铺开剩 14 run(mlp/noguide/sacmpc/sac);不对再查。抽数 `officelarge_*_default_1m_*` 目录(认准新时间戳 `20260724_141849`,旧崩配已删) |
| ✅ 完成 | **OfficeMedium NoGuide 默认档 3-seed** | guidance 增量第三点（补解耦缺口），看 guidance 侧随规模是否单调 | 07-22 全完训（s42/s0/s1，245ep）。抽数（`_extract_medium_default.py` 已加 NoGuide 组）：**能耗 8048.4±173.7 / 每区率 37.2% / comfort 1.011**。已填 §2 骨干解耦条 + INCREMENT §3.1/§3.2。**结论**：guidance 能耗省幅单调（3%→13.2%→22%）；骨干独立效应非单调（9.4%→6.9%→38.2%，Medium 最弱、省幅靠 guidance 主导）。s1 真身目录 =`..._20260721_195310`（`..._095710` 是 0-eval stale） |
| ✅ 完成 | **SAC/SAC+MPC 基线 3-seed 聚合** | 发表门槛（尤其 School 基线曾缺） | 07-20 完成（含 medium mpc s42 补跑）。18+1 run 全完训，6 组均 n=3。三栋排序 Guided-FNO<Diff-MLP<SAC+MPC<纯SAC。SAC+MPC:small 2493±227/medium 13754±127/school 21850±407；纯SAC:small 5204±170/medium 17356±209/school 24862±222。RL 每区率70-95%守不住舒适。`scripts/_extract_sac_baselines.py` |
| 🥉 低 | Conv1d 去噪器对照 | 防新颖性质疑（FNO 赢靠谱结构还是任意空间先验） | 未开始，需改代码 |
| 🥈 中 | **guidance 训练期 vs 推理期贡献三点分解** | 答 stage4 §6.2-5c 开放问题(稳定器 vs 推理倾斜)，诚实刻画 guidance | 🔧 脚本已备 `scripts/_guidance_decompose.py`（reload checkpoint→复现test协议→NoGuide/Full-推理η0/Full-推理η0.5）。⚠️**朴素4点η扫描已否**(η>0.5=critic off-support污染、η=0≠NoGuide)。⚠️**event 里无多-η数据**(旧 handoff"数据在event"错，只有单次训练指标)，须靠 reload 推理脚本。含首跑验证闸门。等 GPU 空 |
| 🥉 低 | diffusion_steps 扫描 / 分区轴空间谱补 seed | 附录、堵必问 | 未做 |

- **别做**：PPO/TD3 等新 RL 基线（无底洞、不服务核心论点）；guidance Level-2 新网络（永不做）。

### 3.1 已生成论文图（`paper_figures_v2/`，均默认档新数据，07-19）
- **主管线** `scripts/gen_paper_figures_v2.py`（读 `master_metrics_v2.csv`）：fig1 三楼能耗、**fig2_saving_curve_monotonic**（12.4→19.2→52 单调，已换掉旧洼地图）、fig3 每区违规率、fig4 骨干/引导解耦、fig5 OfficeSmall 消融柱状、fig6 参数效率、fig7 comfort、fig8 能耗-违规散点、fig9 School 训练稳定性。
- **追加** `scripts/gen_extra_figures.py`（复用 REG 目录同源）：**figA 训练奖励曲线 FNO vs MLP×三楼**（3-seed 均值±std，FNO 全程稳、MLP 楼越大越崩，强图）；**figB OfficeSmall 消融 2×2 热力图**（残差×引导，能耗+违规，示"引导压违规、guidance 部分吸收残差纠偏：残差舒适收益 3.5pp→1.7pp"）。
- ⚠️ **数据源已更新**：`extract_master_metrics.py` 的 OfficeMedium Full/MLP 已指向默认档目录（旧手调档作废）。重画任一图前先跑 extract 再跑 gen。
- ⚠️ **现有 fig1-9/figA/figB 是 07-19 从旧 ddof=0 CSV 画的**（误差棒偏小）。CSV 已于 07-20 重生成为 ddof=1，**重画后误差棒会 ×√(3/2)≈1.22**。均值不变，不急重画，等图定稿(投稿靶定后)一并重跑。
- **待补图（数据未齐）**：**SAC 基线对比（数据已齐可画：见 SAC_QUEUE 6 组 3-seed）**、modes 扫描（未做）、消融热力图扩三楼（需 OffMed/School NoGuide+NoRes）。

---

## 4. 关键纪律 / 踩过的坑（压缩版）

- **07-21 新教训：gate 全程≈0.5、不随 guidance 衰减**——跨 checkpoint 核实证伪（Small 引导 0.508 vs 无引导 0.510，School 0.516 vs 0.576）。残差抑制靠**残差权重训练后近零（缩小 10-20×）**，非 gate 关闭。别把"门控随 guidance 自衰减"写进论文（已从 `gen_speech.py` + `INCREMENT_over_conference.md` 清除）。见 memory `guided-difffno-residual-inert`。
- **抽默认档数据认准 `officemedium_*_default_1m_*` 目录**。曾误用 `officemedium_aligned_tailmean.py`(RUNS 硬编码手调档目录)把手调档当探针=循环论证。用 `_extract_medium_default.py` / `_verify_probe_vs_aligned.py`（每 run 独立 EventAccumulator 防缓存串味）。
- **RAM 错峰**：16GB 机，FNO/MLP 1M buffer ~2.5GB/重进程、SAC 200k 轻。"安全上限约 6 重进程"是**低基线时**（07-17 测，3SAC+2FNO+1MLP=4GB空闲）。⚠️ **07-20 新教训：当前高基线（VS Code/浏览器等占 ~10GB）下，2 个满-buffer FNO 就装不下**——实测 2FNO→RAM 1838MB 逼近 thrash，已杀一个。**FNO 多 seed 必须串行**（一个完训再启下一个），或先关常用程序腾 RAM。启新 run 前必查 RAM、单 FNO 需 RAM>4-5GB。
- **完训判据**：run 总轮数 = **245 epoch**（非 122，那是 SAC）；末段窗取 8 点均值。
- **违规口径**：每区违规率，见 §1。
- **🔴 OfficeLarge max_power 纪律（07-24 定，血泪教训，最重要）**：`max_power`(单台/单区 HVAC 功率上限，BEAR 官方定义 utils_building.py:398)默认 **8000W 对 OfficeLarge 物理欠配**，是首批全套崩溃(comfort~9.5/每区率~100%/FNO塌零动作)的**唯一根因**。证据：全年8759时刻逐区稳态解析(达26°C所需动作×8000=物理需求功率)——**School 最费区仅7890W<8000(0/25饱和,故能训);OfficeLarge 11/23区超8000W**(BASEMENT 75225W=9.4倍/全年91%时刻不够、3个PLENUM各~30000W、DATACENTER_BASE 25767W、6外围区~10400W)。**机制**：功率不够→守舒适要满饱和大动作→BEAR能耗惩罚(∝原始动作)碾压→策略塌成零动作局部最优(ep5就崩)。**Fix=OfficeLarge 专用 `--max-power 32000`**(launcher 已改,三栋 Small/Medium/School 仍 8000 不动)。定 32000 依据：盖住除 BASEMENT 外全部区全年峰值+余量,典型动作 mean0.21 落回三栋(0.15/0.41/0.28)同档;BASEMENT 需 75kW 物理无解、写 limitation。**非调参**：max_power 是 plant-sizing 物理参数(BEAR 原生支持 np.ndarray 逐区设),大楼配大机组天经地义;能耗可比性不破(energy=|action|×max_power,物理制热量固定→读数不随 max_power 变)。⚠️ **纠正旧误解**：卡住的是**制热不够**(Tucson零动作漂13-17°C全楼要升温到26,地下室贴地基/回风腔散热快),**不是"DATACENTER高内热负荷需制冷"**(4个datacenter仅地下室那个吃力)。memory `guided-difffno-officelarge-maxpower-rootcause`。
- **OfficeLarge 规格（07-22 定，BEAR 解析器实测；07-24 补 max_power）**：roomnum=23, **state_dim=71**(=3×23+2), rfft长=12, modes4 谱保留 33%。**width=96**（地板 ≥71，富余25；序列 48/64/96/128 对 state_dim 20/56/71/77 单调）；结构照旧 L1/modes4；**max_power=32000**(专属,见上条)；其余训练超参全默认档（07-24 逐项核实与三栋 Full 逐字节一致：bc0.8/final0.1/decay150000/vp10/actor1e-4/critic2e-5/batch256/update0.5/buffer1M/g0.5，只 building_type/width/max_power/seed 该不同才不同）。launcher `scripts/_large_launch.sh`。**机时实测 ~15h/seed**。抽数复用 `_extract_medium_default.py` 加 OfficeLarge 组，认准 `officelarge_*_default_1m_*` 目录(**新时间戳 20260724_141849,8000W崩配旧run已删**)。
- **rolling**：`scripts/_sac_rolling.sh`（MAX_PARALLEL=3，含 medium 队列）；停 = `touch run_logs/_rolling.STOP`。

---

## 5. 封存指针（历史推导，需要时查旧文件）

- **阶段四** `HANDOFF_stage4_alignment.md`：结构对齐(L2/m6→L1/m4 无损)、训练超参统一 (a)/(b) 决策全过程、洼地为何是假象、探针权衡纠错、循环论证事故复盘、支柱2 旧「不赌单调/共同洼地」reframe 推导史。
- **阶段三** `HANDOFF_option3_bear_journal.md`：三建筑推导史、耦合结构假说、guidance 发现史。
- **SAC 基线** `HANDOFF_sac_baseline.md`：诊断 + 公平协议。
- **增量对比** `docs/INCREMENT_over_conference.md`：相对会议论文的逐项增量 + 数据。
- **残差核查** memory `guided-difffno-residual-inert`；**OffMed 统一决策** memory `guided-difffno-officemedium-config-unify`。
- **队列真相源** `run_logs/SAC_QUEUE.md`。

