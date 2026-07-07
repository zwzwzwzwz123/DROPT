# DiffFNO 期刊增量 — 部署期引导方案 交接文档

- 日期：2026-07-07
- 分支：331B
- 作者/接手：Wei Zou（项目负责人）

## 0. 一句话

在已投稿 ICCC 2026 的会议版 DiffFNO（`docs/DiffFNO_ICCC2026_submission.pdf`）基础上，做期刊增量（目标 IoT-J 一档应用期刊）：
**策略只训一次，部署时用一个"从观测数据学出来的温度预测小模型"做梯度引导，在扩散采样阶段把动作朝"更少舒适违规"推；引导强度是部署期可调旋钮，扫一遍就得到一整条能耗-舒适 Pareto 前沿。** 直接对症会议版"违规率降不够低"的痛点。

## 1. 为什么是这个方案（增量的定位）

- 会议版的核心：条件扩散策略 + FNO 动作去噪器 + 残差旁路 + 双 Critic 混合训练（BC+PG）。
- 原本设想的增量"Critic-Q 梯度引导采样"有**硬伤**：会议版 actor 训练目标本就是最大化同一个 Q（论文式 12 `L_PG=-E[Q]`），采样时再朝 Q 爬 = 冗余，审稿人会说"训练时就在做"。
- 新方案把引导目标从"训练用的 Q"换成"独立的舒适约束（温度预测模型）"：训练时"省电 vs 舒适"权重焊死，引导让这个权衡在**部署期在线可调、不重训**——这是训练目标做不到的事，堵上了冗余硬伤。
- IoT 叙事：边缘不重训、需求响应(DR)/电价响应、异构楼宇，都能顺会议版已铺的 sensing-actuation loop 讲下去。

## 2. 方案的技术构成

### 2.1 温度预测小模型（surrogate）
- 结构：普通 4 层 MLP（Linear+Mish），输入 `[state(20) + action(6)]=26` → 输出 6 维**温度变化量 ΔT**，加回当前温度得 `T_{t+1}`。故意做小，不是论文主角。
- 预测 ΔT 而非绝对温度（热惯性大，数值更稳）；输入输出做归一化，统计量随模型保存。
- 对 action 可微——这是它能当引导用的前提。

### 2.2 部署期引导（注入扩散采样）
- 复用 `diffusion.py` 已有的通用引导钩子：`x_recon <- x_recon - scale * guidance_fn(x_recon, s, t)`。
- 舒适引导（主力）：surrogate 预测 `T_{t+1}` → 平滑越界代价 `C(a)` → 回传 `∂C/∂a`，采样朝"少违规"滑。
- 能耗引导（备选，默认关）：闭式 `∂(Σ|a|·ac_map·max_power)/∂a`，单向省能旋钮。
- `combine_guidance` 可按权重合成多个引导项。

## 3. 交付文件清单

**原则：全部新增，不改训练脚本/核心模块（唯一例外见第 6 节的 bug 修复）。**

| 文件 | 作用 |
|---|---|
| `deploy_guidance/surrogate.py` | `TempSurrogate` 模型 + `build_comfort_guidance` + `build_energy_guidance` + `combine_guidance` |
| `deploy_guidance/policy_io.py` | `PolicySpec` + `build_policy`/`load_policy`，复现结构并加载已训 checkpoint（兼容裸 state_dict 和 `{"model":...}` 两种格式） |
| `scripts/collect_transitions.py` | 采集观测转移 `(s_t, a_t, T_{t+1})`，uniform/extreme/可选 on-policy 混合 |
| `scripts/train_comfort_surrogate.py` | 训练 surrogate，报物理量纲 RMSE(°C) |
| `scripts/eval_guided_pareto.py` | 加载策略+surrogate，扫引导强度，出 Pareto CSV |

## 4. 运行流程（尚未端到端跑过——见第 7 节阻塞项）

```
# 1) 采数据（纯随机，不需 checkpoint；建议带 --policy-checkpoint 混 on-policy 缓解分布偏移）
python scripts/collect_transitions.py --num-transitions 40000

# 2) 训温度模型（CPU 几分钟，看 val_RMSE 是否很低）
python scripts/train_comfort_surrogate.py

# 3) 扫引导强度出 Pareto（需要你训练好的策略 .pth）
python scripts/eval_guided_pareto.py --policy-checkpoint <你的.pth> \
    --backbone-variant residual --fno-width 48 --fno-modes 4 --fno-layers 1 --diffusion-steps 6
```

## 5. 合法性（反上帝视角）——已逐文件核实

- surrogate 只从 `next_obs[:roomnum]`（智能体 t+1 观测到的传感器读数）学 `(s,a)→T_{t+1}`，标准系统辨识。
- 能耗引导只用 `ac_map`（设备额定容量）+ `max_power`（铭牌功率），真实运营方本就知道。
- **全程不碰** BEAR 的 `A_d`/`B_d`/`X_new`/内部 reward 分解。全仓库扫描确认这些字段仅出现在注释里，无真实访问。

## 6. 复查中修复的问题
1. **既存 bug（改了 1 处现有文件）**：`diffusion/__init__.py` 第 2 行 `from .rectified_flow import RectifiedFlow` 是死导入（`rectified_flow.py` 在 `c28d5fd` 被删）。**这导致 331B 上任何 `from diffusion import ...` 崩溃、训练脚本跑不起来。** 已删该行（纯修复，全仓库无处真正使用 RectifiedFlow）。
2. 误导注释：collect 原写"取自 info['zone_temperature']"，实际永远走 fallback 取 `next_obs[:roomnum]`。已改为如实说明。
3. 动作域外推：surrogate 只在 `a∈[-1,1]` 训练，采样中间步 `x_recon` 可能越界。comfort 引导已加域内 clamp（域外梯度=0）。
4. no_grad 路径：eval 在 `torch.no_grad()` 下跑、引导靠内部 `enable_grad()` 覆盖，已补测试确认生效。

验证：合成张量冒烟测试全过（surrogate 前向+梯度、两种引导、combine、注入扩散改变输出、no_grad 路径、域外梯度=0）。临时测试文件已删。

## 7. 阻塞项 / 下一步
- **阻塞**：当前工作树无训练好的 `.pth`，无法端到端跑 eval。需负责人提供策略 checkpoint 路径 + 确认结构超参（现按 ablation 默认：residual / width48 / modes4 / layers1 / steps6）。
- 可先跑 `collect_transitions.py`（纯随机）验证与 BEAR 接通。

## 8. ⚠️ 全文最大审稿威胁（务必主动应对）

**已核实**：BEAR 转移对动作**精确线性/仿射**——`X_new = A_d·X + B_d·Y`，动作在 `Y` 里线性出现；唯一非线性项 Occupower 只依赖当前温度、与动作无关。即固定状态下 `T_{t+1} = c(s_t) + M·a`，`M` 为常数矩阵。

推论与应对：
- surrogate 会学得极好（RMSE 极低）——正常，甚至一个线性层够用。**不削弱方案**：可把"引导梯度 ∂T/∂a=M 为常数、方向稳定可信"写成方法的干净性质。
- **"违规率能到 0"的担心不成立**：动作饱和(极端天气物理不可达) + 单步贪心 + 能耗代价，三道墙保证 Pareto 前沿存在、压不到 0；能压很低反而是卖点。
- **真正的威胁**：审稿人会问"线性动力学下一步闭式 QP 就能精确求最优动作，你扩散+引导不就是重造 MPC 吗？"
  - **应对叙事**：引导是在"已从 MPC 专家学到长期/多模态行为的生成策略"之上做**部署期可调校正**，价值在"可调 + 继承长期行为"，**不是**"解这一步最优"。绝不把故事讲成"学动力学去优化动作"。
  - **必做实验（挡 QP 质疑的关键）**：加一个**单步贪心 QP/MPC 基线**，证明它单步违规低但**长期指标（整段能耗、平滑度、累计违规）差于 扩散策略+引导**（因纯单步无长期价值、动作抖、忽略惯性）。此对比不做或做输，则这条线危险。

## 9. 尚未落实的待办
- [ ] 拿到策略 checkpoint，端到端跑通三步流程。
- [ ] 写"单步贪心 QP/MPC"基线控制器（利用线性结构精确求解），量化长期差距。
- [ ] 采数据时混入 on-policy 缓解分布偏移（surrogate 主要在随机动作上训，eval 状态来自策略）。
- [ ] 舒适+能耗同开时，两者梯度量纲不同（~0.04 vs ~0.17），`--energy-weight` 需手调平衡。
- [ ] 画 Pareto 图的绘图脚本（可参考 `scripts/paper_compare_energy_violations.py` 风格）。
- [ ] 决定是否修 `diffusion/__init__.py` 之外还需回填一个空的 rectified_flow 兼容（当前已确认无需）。
