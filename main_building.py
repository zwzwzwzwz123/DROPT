# ========================================
# BEAR 建筑环境优化训练主程序
# ========================================
# 基于 DROPT 框架，应用扩散模型+强化学习到建筑HVAC控制

import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
import warnings

# 导入建筑环境
from env.building_env_wrapper import make_building_env
# 导入日志格式化工具
from dropt_utils.logger_formatter import EnhancedTensorboardLogger
# 导入配置常量（修复：统一管理配置参数，避免硬编码）
from env.building_config import (
    DEFAULT_REWARD_SCALE,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_TEMP_WEIGHT,
    DEFAULT_VIOLATION_PENALTY,
    DEFAULT_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE,
    DEFAULT_MAX_POWER,
    DEFAULT_TIME_RESOLUTION,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_TRAINING_NUM,
    DEFAULT_TEST_NUM,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GAMMA,
    DEFAULT_N_STEP,
    DEFAULT_STEP_PER_EPOCH,
    DEFAULT_STEP_PER_COLLECT,
    DEFAULT_EPISODE_PER_TEST,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_BETA_SCHEDULE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ACTOR_LR,
    DEFAULT_CRITIC_LR,
    DEFAULT_EXPLORATION_NOISE,
    DEFAULT_LOG_DIR,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_MPC_PLANNING_STEPS,
)

# 导入 DROPT 核心组件
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic

warnings.filterwarnings('ignore')


def get_args():
    """
    解析命令行参数（针对建筑环境调整）
    """
    parser = argparse.ArgumentParser(description='BEAR 建筑环境 HVAC 优化训练程序')
    
    # ========== 环境参数（使用配置常量作为默认值） ==========
    parser.add_argument('--building-type', type=str, default='OfficeSmall',
                        help='建筑类型 (OfficeSmall, Hospital, SchoolPrimary等)')
    parser.add_argument('--weather-type', type=str, default='Hot_Dry',
                        help='气候类型 (Hot_Dry, Hot_Humid, Cold_Humid等)')
    parser.add_argument('--location', type=str, default='Tucson',
                        help='地理位置 (Tucson, Tampa, Rochester等)')
    parser.add_argument('--target-temp', type=float, default=DEFAULT_TARGET_TEMP,
                        help=f'目标温度 (°C，默认{DEFAULT_TARGET_TEMP})')
    parser.add_argument('--temp-tolerance', type=float, default=DEFAULT_TEMP_TOLERANCE,
                        help=f'温度容差 (°C，默认{DEFAULT_TEMP_TOLERANCE})')
    parser.add_argument('--max-power', type=int, default=DEFAULT_MAX_POWER,
                        help=f'HVAC 最大功率 (W，默认{DEFAULT_MAX_POWER})')
    parser.add_argument('--time-resolution', type=int, default=DEFAULT_TIME_RESOLUTION,
                        help=f'时间分辨率 (秒，默认{DEFAULT_TIME_RESOLUTION}=1小时)')
    parser.add_argument('--episode-length', type=int, default=DEFAULT_EPISODE_LENGTH,
                        help='Episode horizon in steps (default ~1 week)')
    parser.add_argument('--full-episode', action='store_true', default=False,
                        help='Override to use the full-year trajectory (sets episode-length=None)')
    parser.add_argument('--energy-weight', type=float, default=DEFAULT_ENERGY_WEIGHT,
                        help=f'能耗权重 α (默认{DEFAULT_ENERGY_WEIGHT})')
    parser.add_argument('--temp-weight', type=float, default=DEFAULT_TEMP_WEIGHT,
                        help=f'温度偏差权重 β (默认{DEFAULT_TEMP_WEIGHT})')
    parser.add_argument('--add-violation-penalty', dest='add_violation_penalty',
                        action='store_true', default=True,
                        help='启用温度越界惩罚（默认开启）')
    parser.add_argument('--no-add-violation-penalty', dest='add_violation_penalty',
                        action='store_false',
                        help='关闭温度越界惩罚')
    parser.add_argument('--violation-penalty', type=float, default=DEFAULT_VIOLATION_PENALTY,
                        help=f'温度越界惩罚系数 γ (默认{DEFAULT_VIOLATION_PENALTY})')

    # ========== 专家控制器参数 ==========
    parser.add_argument('--expert-type', type=str, default=None,
                        choices=['mpc', 'pid', 'rule', 'bangbang', None],
                        help='专家控制器类型（用于行为克隆）')
    parser.add_argument('--mpc-planning-steps', type=int, default=DEFAULT_MPC_PLANNING_STEPS,
                        help=f'MPC 专家规划步数 (默认{DEFAULT_MPC_PLANNING_STEPS})')
    parser.add_argument('--bc-coef', action='store_true', default=False,
                        help='是否使用行为克隆（BC）损失')
    parser.add_argument('--bc-weight', type=float, default=0.8,
                        help='行为克隆损失初始权重（默认0.8）')
    parser.add_argument('--bc-weight-final', type=float, default=0.1,
                        help='BC权重最终值（默认0.1，逐渐过渡到策略梯度）')
    parser.add_argument('--bc-weight-decay-steps', type=int, default=150000,
                        help='BC权重线性衰减步数（默认15万步，延长专家引导）')
    
    # ========== 基础训练参数（使用配置常量作为默认值） ==========
    parser.add_argument('--exploration-noise', type=float, default=DEFAULT_EXPLORATION_NOISE,
                        help=f'探索噪声标准差 (默认{DEFAULT_EXPLORATION_NOISE}，降低随机扰动)')
    parser.add_argument('--algorithm', type=str, default='diffusion_opt',
                        help='算法名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_SIZE,
                        help=f'经验回放缓冲区大小 (默认{DEFAULT_BUFFER_SIZE:,})')
    parser.add_argument('-e', '--epoch', type=int, default=20000,
                        help='总训练轮次 (默认 20000，便于阶段性评估)')
    parser.add_argument('--step-per-epoch', type=int, default=16384,
                        help='每个epoch采集的环境步数 (默认16384)')
    parser.add_argument('--step-per-collect', type=int, default=4096,
                        help='每次采集的步数 (默认4096，提升样本质量)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='批次大小 (默认256)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='权重衰减系数')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'折扣因子 (默认{DEFAULT_GAMMA})')
    parser.add_argument('--n-step', type=int, default=DEFAULT_N_STEP,
                        help=f'N步TD学习 (默认{DEFAULT_N_STEP})')
    parser.add_argument('--training-num', type=int, default=DEFAULT_TRAINING_NUM,
                        help=f'并行训练环境数量 (默认{DEFAULT_TRAINING_NUM})')
    parser.add_argument('--test-num', type=int, default=DEFAULT_TEST_NUM,
                        help=f'并行测试环境数量 (默认{DEFAULT_TEST_NUM})')
    parser.add_argument('--episode-per-test', type=int, default=DEFAULT_EPISODE_PER_TEST,
                        help='每次评估所跑的episode数量（减少可缩短每轮评估时间）')
    parser.add_argument('--prioritized-replay', action='store_true', default=False,
                        help='是否启用优先经验回放（PER）')
    parser.add_argument('--prior-alpha', type=float, default=0.6,
                        help='PER 采样分布平滑因子 alpha')
    parser.add_argument('--prior-beta', type=float, default=0.4,
                        help='PER 重要性采样修正 beta')

    # ========== 网络架构参数（使用配置常量作为默认值） ==========
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM,
                        help=f'MLP隐藏层维度 (默认{DEFAULT_HIDDEN_DIM})')
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='Actor学习率 (默认1e-4，降低梯度震荡)')
    parser.add_argument('--critic-lr', type=float, default=DEFAULT_CRITIC_LR,
                        help=f'Critic学习率 (默认{DEFAULT_CRITIC_LR})')
    parser.add_argument('--reward-scale', type=float, default=DEFAULT_REWARD_SCALE,
                        help='Scale raw rewards to stabilize critic/actor updates')
    parser.add_argument('--reward-normalization', dest='reward_normalization', action='store_true',
                        help='Enable Tianshou reward normalization (default)')
    parser.add_argument('--no-reward-normalization', dest='reward_normalization', action='store_false',
                        help='Disable reward normalization for legacy behavior')
    parser.set_defaults(reward_normalization=True)

    # ========== 扩散模型参数（使用配置常量作为默认值） ==========
    parser.add_argument('--diffusion-steps', type=int, default=DEFAULT_DIFFUSION_STEPS,
                        help=f'扩散步数 (默认{DEFAULT_DIFFUSION_STEPS})')
    parser.add_argument('--beta-schedule', type=str, default=DEFAULT_BETA_SCHEDULE,
                        help=f'噪声调度类型 (默认{DEFAULT_BETA_SCHEDULE})')

    # ========== 日志和设备参数（使用配置常量作为默认值） ==========
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOG_DIR,
                        help=f'日志保存目录 (默认{DEFAULT_LOG_DIR})')
    parser.add_argument('--log-prefix', type=str, default='default',
                        help='日志文件前缀')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='观察模式（不训练）')
    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help='是否使用学习率衰减')
    parser.add_argument('--save-interval', type=int, default=DEFAULT_SAVE_INTERVAL,
                        help=f'模型保存间隔（轮次，默认{DEFAULT_SAVE_INTERVAL}）')
    parser.add_argument('--vector-env-type', type=str, default='dummy',
                        choices=['dummy', 'subproc'],
                        help='向量环境实现 (dummy=单进程, subproc=多进程并行)')
    parser.add_argument('--log-update-interval', type=int, default=50,
                        help='记录梯度/优化指标到 TensorBoard 的间隔（梯度步）')
    parser.add_argument('--update-per-step', type=float, default=0.5,
                        help='每个环境步执行的参数更新次数 (默认0.5)')
    args = parser.parse_args()

    if args.full_episode:
        args.episode_length = None

    if args.reward_normalization and args.n_step > 1:
        print("⚠️  提示: n_step>1 与奖励归一化不兼容，已自动关闭 reward_normalization")
        args.reward_normalization = False

    if args.bc_weight_final is None:
        args.bc_weight_final = args.bc_weight

    return args


def main():
    """主训练函数"""
    # ========== 获取参数 ==========
    args = get_args()
    
    # 设置设备
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ========== 创建日志目录 ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f"{args.log_prefix}_{args.building_type}_{args.weather_type}_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    
    # 创建 TensorBoard writer 和增强的日志记录器
    writer = SummaryWriter(log_path)
    # logger will be created after envs to inject metrics_getter
    metrics_getter = None
    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,  # 每个epoch都输出（可改为10表示每10个epoch输出一次）
        verbose=True,  # True=详细格式，False=紧凑格式
        diffusion_steps=args.diffusion_steps,  # 扩散模型步数
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch,
        metrics_getter=None,  # placeholder, will be replaced later
        png_interval=5,
    )
    
    # 打印配置
    print("\n" + "=" * 60)
    print("  BEAR 建筑环境 HVAC 优化训练")
    print("=" * 60)
    print(f"\n配置参数:")
    pprint.pprint(vars(args))
    print()
    
    # ========== 创建环境 ==========
    print("正在创建环境...")
    expert_kwargs = None
    if args.expert_type:
        expert_kwargs = {}
        if args.expert_type == 'mpc':
            expert_kwargs['planning_steps'] = args.mpc_planning_steps

    env, train_envs, test_envs = make_building_env(
        building_type=args.building_type,
        weather_type=args.weather_type,
        location=args.location,
        target_temp=args.target_temp,
        temp_tolerance=args.temp_tolerance,
        max_power=args.max_power,
        time_resolution=args.time_resolution,
        energy_weight=args.energy_weight,
        temp_weight=args.temp_weight,
        episode_length=args.episode_length,
        add_violation_penalty=args.add_violation_penalty,
        violation_penalty=args.violation_penalty,
        reward_scale=args.reward_scale,  # 奖励缩放，降低Q值和损失的尺度
        expert_type=args.expert_type if args.bc_coef else None,
        expert_kwargs=expert_kwargs,
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type
    )
    
    print(f"✓ 环境创建成功")
    print(f"  建筑类型: {args.building_type}")
    print(f"  气候类型: {args.weather_type}")
    print(f"  地理位置: {args.location}")
    print(f"  房间数量: {env.roomnum}")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作维度: {env.action_dim}")
    print(f"  奖励缩放系数: {env.reward_scale}")
    if args.expert_type:
        print(f"  专家控制器: {args.expert_type}")
        if args.expert_type == 'mpc':
            print(f"    - MPC 规划步数: {args.mpc_planning_steps}")

    def _aggregate_metrics(vector_env):
        if vector_env is None:
            return None
        env_list = getattr(vector_env, "_env_list", None)
        if not env_list:
            return None
        values = [env_inst.consume_metrics() for env_inst in env_list]
        values = [m for m in values if m]
        if not values:
            return None
        result = {}
        for key in ('avg_energy', 'avg_comfort_mean', 'avg_violations', 'avg_pue'):
            nums = [m[key] for m in values if m.get(key) is not None]
            if nums:
                result[key] = float(np.mean(nums))
        return result if result else None

    def metrics_getter(mode: str):
        target_env = train_envs if mode == 'train' else test_envs
        return _aggregate_metrics(target_env)

    logger.training_logger.metrics_getter = metrics_getter

    # ========== 创建网络 ==========
    print("\n正在创建神经网络...")
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0
    
    # Actor (扩散模型)
    actor = MLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        t_dim=16
    ).to(args.device)

    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # Critic (双Q网络)
    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )
    
    print(f"✓ 网络创建成功")
    print(f"  Actor 参数量: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic 参数量: {sum(p.numel() for p in critic.parameters()):,}")
    
    # ========== 创建扩散模型 ==========
    diffusion = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor,
        max_action=max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
    ).to(args.device)
    
    # ========== 创建策略 ==========
    print("\n正在创建策略...")
    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion,  # 使用扩散模型作为 actor
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=args.device,
        tau=0.005,
        gamma=args.gamma,
        exploration_noise=args.exploration_noise,
        bc_coef=args.bc_coef,
        bc_weight=args.bc_weight,
        bc_weight_final=args.bc_weight_final,
        bc_weight_decay_steps=args.bc_weight_decay_steps,
        action_space=env.action_space,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        reward_normalization=args.reward_normalization,
    )
    
    print(f"✓ 策略创建成功")
    print(f"  算法: {args.algorithm}")
    print(f"  扩散步数: {args.diffusion_steps}")
    if args.bc_coef:
        print(f"  行为克隆权重: {args.bc_weight}")
    
    # ========== 创建收集器 ==========
    print("\n正在创建数据收集器...")
    buffer_num = max(1, args.training_num)
    if args.prioritized_replay:
        replay_buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=buffer_num,
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        replay_buffer = VectorReplayBuffer(args.buffer_size, buffer_num)

    train_collector = Collector(
        policy,
        train_envs,
        replay_buffer,
        exploration_noise=True
    )

    test_collector = Collector(policy, test_envs)

    # 仅在前 warmup_steps 关闭采集噪声，之后恢复默认噪声
    warmup_noise_steps = 250_000

    def train_fn(epoch: int, env_step: int):
        """动态切换采集噪声：前 warmup_steps 关闭，之后开启。"""
        if not hasattr(train_collector, "exploration_noise"):
            return
        enable_noise = env_step >= warmup_noise_steps
        # 仅在状态变化时更新，避免反复赋值
        if train_collector.exploration_noise != enable_noise:
            train_collector.exploration_noise = enable_noise
            status = "开启" if enable_noise else "关闭"
            print(f"[train_fn] env_step={env_step}: 已{status}采集噪声")
    
    print(f"✓ 收集器创建成功")
    print(f"  训练环境数: {args.training_num}")
    print(f"  测试环境数: {args.test_num}")
    buffer_type = "Prioritized" if args.prioritized_replay else "Uniform"
    print(f"  缓冲区大小: {args.buffer_size:,} ({buffer_type})")
    
    # ========== 开始训练 ==========
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"\n⚠️ 注意: 奖励已缩放 {args.reward_scale}x")
    print(f"\n💡 提示: 日志输出已优化，关键指标将清晰显示")
    print(f"  - 每个epoch都会显示详细的训练指标")
    print(f"  - 异常值会用 ⚠ 符号标记")
    print(f"  - 时间统计会自动估算剩余训练时间\n")

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        logger=logger,
        save_best_fn=lambda policy: torch.save(
            policy.state_dict(),
            os.path.join(log_path, 'policy_best.pth')
        ),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: torch.save(
            {
                'model': policy.state_dict(),
                'optim_actor': actor_optim.state_dict(),
                'optim_critic': critic_optim.state_dict(),
            },
            os.path.join(log_path, f'checkpoint_{epoch}.pth')
        ) if epoch % args.save_interval == 0 else None,
        train_fn=train_fn,
    )
    
    # ========== 训练完成 ==========
    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)
    pprint.pprint(result)
    
    # 保存最终模型
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy_final.pth'))
    print(f"\n✓ 模型已保存到: {log_path}")


if __name__ == '__main__':
    main()
