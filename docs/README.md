# docs/ 索引 —— Guided-DiffFNO 期刊版

> 用途：一眼看清 `docs/` 下每份文档的**角色**(活档/活引用/封存)，避免新会话读错文件、或误把封存档当现状。
> 维护：末次更新 2026-07-30。**新会话从 🟢 START HERE 读起。**
> ℹ️ **OfficeLarge（第四栋）已放弃（07-25）**：尝试加 23 区大楼拆支柱2 楼型混杂,但 max_power 无全局解(实训全局标量都卡~48%)、唯一可行的逐区功率违反零侵入,故放弃。**支柱2 维持三栋(Small→Medium→School)**。教训见 stage5 §4 + memory `guided-difffno-officelarge-maxpower-rootcause`。
> ℹ️ **大楼 modes 扫描 2-seed 判决已出（07-29）**：违规铁平(硬结果) + 能耗弱单调掉进噪声(边界 null)。结论=稳健性(modes=4 非调参)当支柱1 防御，**别写"截断=机制"**。s1 链在跑仅补表对称、不改结论。详见 stage5 §0 判决条 + memory `guided-difffno-modes-sweep-school`。**主线可转写作**。

---

## 🟢 START HERE — 活档（先读这些）

| 文件 | 角色 | 一句话 |
|---|---|---|
| **HANDOFF_stage5.md** | **当前状态干净快照** | 唯一"现状+待办"活档。只讲当前状态，不含推导史。任何时候先读它。 |
| **INCREMENT_over_conference.md** | 增量清单（活伴档） | 相对会议论文(DiffFNO ICCC'26)的逐项增量 + 数据结果，持续维护。写作/对外汇报的素材源。 |
| **HANDOFF_sac_baseline.md** | 活引用（协议档） | SAC/SAC+MPC 公平重跑的诊断 + **防作弊红线**(BC 地板逐楼对齐、唯一允许 critic_lr 差异)。基线跑完前仍是纪律来源。 |

## 🔵 支线计划（未激活，主线优先）

| 文件 | 角色 | 一句话 |
|---|---|---|
| PLAN_unified_cross_building_fno.md | 支线计划 | 维度无关 FNO 跨多栋楼 + 零样本迁移（天花板方向）。主线优先，暂不动。 |

## 📦 封存档（推导史，需要查历史时才读，原地保留不删）

| 文件 | 封存日期 | 内容 |
|---|---|---|
| HANDOFF_stage4_alignment.md | 07-19 封存 | 阶段四推导史：结构对齐(L2/m6→L1/m4)、训练超参统一 (a)/(b) 决策全过程、洼地为何是假象、探针权衡纠错、循环论证事故复盘。 |
| HANDOFF_option3_bear_journal.md | — | 阶段三推导史：三建筑推导、耦合结构假说、guidance 发现史。 |
| HANDOFF_sinergym_direction.md | 07-07 死路 | Sinergym 转向被否（动作是全局设定点、FNO 无处落脚）。防重走。 |
| HANDOFF_journal_direction.md | 07-06 死路 | 最早方向决策：双路/门控被否、guidance 做花四方向全否。防重走。 |

---

## 阅读路径

- **接手当前工作** → stage5（现状）→ 需要数据引用则 INCREMENT → 碰 SAC 基线则 sac_baseline。
- **想知道"为什么当初否了某方向"** → 对应封存档（sinergym / journal_direction / stage4 §5 封死的路）。
- **想知道"某结论怎么推出来的"** → stage4（阶段四）/ option3_bear（阶段三）。

> ⚠️ 封存档里大量 🔴 作废标记均已在后续阶段定论——**查历史来这里，看现状去 stage5**。别把封存档的旧数字当现状用（尤其 OfficeMedium 手调档 7042/7202 已被默认档 6985/8645 取代）。
