# Rectified Flow 技术整合报告（BEAR Building）

## 背景
- 目标：在现有 BEAR 建筑节能优化管线中，将 DDPM 替换为 Rectified Flow（RF）速度场模型，提升低步数采样效率与早期收敛稳定性。
- 入口脚本：`rectified_flow_building.py`（复用 `main_building` 参数，并追加 RF 专属参数）。

## 集成方式（组件关系）
- 环境层：`env/building_env_wrapper.py`
  - 构建 BEAR 环境并适配 state/action space。
  - 若专家控制器初始化成功（默认 MPC），每步在 `info` 写入 `expert_action`，供 BC/RF 使用。
- 模型层：
  - 速度网络 `diffusion/model.py::MLP`：输入 `x_t, t_scaled, state`，输出速度场。
  - RF actor `diffusion/rectified_flow.py::RectifiedFlow`：
    - 训练：恒定速度回归 `v = x0 - z0`，`loss_fn(pred_v, target_v)`，可选 reflow（默认关）。
    - 采样：Euler ODE（默认），steps=`rf_sample_N`（默认=diffusion_steps），`time_scale=10`。
    - 噪声：初始噪声 `noise_scale=0.5`，采样时 `sigma_var=0.05` 控制衰减噪声。
  - Critic `diffusion/model.py::DoubleCritic`：双 Q 估计。
- 策略层：`policy/DiffusionOPT`
  - 组装 RF actor + critic，Tianshou BasePolicy 接口。
  - BC 模式：`bc_coef=True` 时，调用 `actor.loss(expert_action, state)`；否则走策略梯度 `-Q`.
  - 目标网络软更新、n-step TD、可选探索噪声调度。
- 训练驱动：`dropt_utils.tianshou_compat.offpolicy_trainer`
  - Collector + `VectorReplayBuffer`/PER。
  - warmup_noise_steps=200,000：前期关闭 collector 探索噪声，优先模仿专家。
  - EnhancedTensorboardLogger 记录损失/Q/环境指标，周期保存 checkpoint、best。

## 相较原 DDPM 的主要变化
- 生成范式：由噪声/score 去噪（DDPM）改为恒定速度场 ODE（RF），采样可更确定、更平滑。
- 步数与时间缩放：默认 `diffusion_steps=6`（原默认≥10），`time_scale=10`（原 999），低步数加速训练/推理并降低时间嵌入高频振荡。
- 噪声策略：RF 仅初始噪声 + 轻量 `sigma_var`，反向不叠噪；动作方差更可控。
- 专家引导：默认 `expert_type=mpc` 且 `bc_coef=True`，早期直接回归专家动作，比 DDPM 的噪声回归更接近监督回归，收敛更快。
- 采样器：默认 Euler，避免 RK45 的 CPU 慢路径；`rf_sample_N` 与训练步数对齐。
- 日志前缀：默认 `rectified_flow_mpc`，与 DDPM 版区分。

## 默认超参（2026-01 更新）
- `diffusion_steps=6`（如用户未手动指定且原默认≥10）
- `rf_sample_N=diffusion_steps`
- `rf_time_scale=10`
- `rf_noise_scale=0.5`
- `rf_sigma_var=0.05`
- `rf_sampler=euler`
- `expert_type=mpc`（若未指定）
- `bc_coef=True`
- `warmup_noise_steps=200000`
- 日志前缀：`rectified_flow_mpc`
- reflow：默认关闭（需教师对 `(z0, x0)` 才有意义）

## 训练/推理流程
1) 参数解析：`rectified_flow_building.py::get_args` 合并 base + RF 配置，设置默认前缀、MPC、BC、步数压缩。
2) 环境构建：`make_building_env` 创建 train/test vector env；MPC 初始化成功后写入 `expert_action`。
3) 网络搭建：MLP(velocity) → RectifiedFlow actor；DoubleCritic。
4) 策略封装：DiffusionOPT 绑定 actor/critic/optimizer；BC/PG 混合；目标网络软更新。
5) 数据采集：Collector 写入 ReplayBuffer；warmup 前关闭探索噪声。
6) 训练循环：offpolicy_trainer 调用 `learn`，日志记录损失、Q 值、环境指标；周期保存 best/checkpoint。
7) 采样/评估：actor.sample 使用 Euler ODE，steps=rf_sample_N，输出裁剪到 [-1,1]。

## 重要注意
- 若 BC 损失为 0，通常是专家初始化失败或 `batch.info` 缺少 `expert_action`；需检查启动日志的“专家控制器初始化成功”提示。
- reflow 仍为可选项，需提供教师轨迹 `(z0, x0)` 或额外蒸馏数据；未满足条件时保持关闭更稳。
- 如需更快探索，可缩短 warmup；如需更平滑生成，可小幅上调 `diffusion_steps`（如 8）。

## 启动示例
```bash
python rectified_flow_building.py \
  --building-type OfficeSmall \
  --weather-type Hot_Dry \
  --logdir log_building \
  --device cuda:0
```
（直接 `python rectified_flow_building.py` 也可，使用上述默认值）

## 模型层与策略层关系补充
- 模型层（Rectified Flow actor + DoubleCritic）：只负责“如何生成动作/估计 Q”，是纯网络结构与前向/损失定义。
- 策略层（DiffusionOPT）：将模型层的 actor/critic 组合成可训练的 RL 策略，决定用什么损失（BC 或 PG）、何时软更新目标网络、如何从 buffer 取数据、如何处理探索噪声、以及如何向 Tianshou 暴露 `forward/learn` 接口。
- 简言之：模型层提供“函数”，策略层负责“训练和调度这些函数”，并连接采集器、经验回放和训练循环。
