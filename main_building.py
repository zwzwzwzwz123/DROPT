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
from tianshou.trainer import offpolicy_trainer
import warnings

# 导入建筑环境
from env.building_env_wrapper import make_building_env
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
    DEFAULT_TRAINING_NUM,
    DEFAULT_TEST_NUM,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GAMMA,
    DEFAULT_N_STEP,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_BETA_SCHEDULE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ACTOR_LR,
    DEFAULT_CRITIC_LR,
    DEFAULT_EXPLORATION_NOISE,
    DEFAULT_LOG_DIR,
    DEFAULT_SAVE_INTERVAL,
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
    parser.add_argument('--episode-length', type=int, default=None,
                        help='回合长度（步数，None表示完整年度）')
    parser.add_argument('--energy-weight', type=float, default=DEFAULT_ENERGY_WEIGHT,
                        help=f'能耗权重 α (默认{DEFAULT_ENERGY_WEIGHT})')
    parser.add_argument('--temp-weight', type=float, default=DEFAULT_TEMP_WEIGHT,
                        help=f'温度偏差权重 β (默认{DEFAULT_TEMP_WEIGHT})')
    parser.add_argument('--add-violation-penalty', action='store_true', default=False,
                        help='是否添加温度越界惩罚')
    parser.add_argument('--violation-penalty', type=float, default=DEFAULT_VIOLATION_PENALTY,
                        help=f'温度越界惩罚系数 γ (默认{DEFAULT_VIOLATION_PENALTY})')

    # ========== 专家控制器参数 ==========
    parser.add_argument('--expert-type', type=str, default=None,
                        choices=['mpc', 'pid', 'rule', 'bangbang', None],
                        help='专家控制器类型（用于行为克隆）')
    parser.add_argument('--bc-coef', action='store_true', default=False,
                        help='是否使用行为克隆（BC）损失')
    parser.add_argument('--bc-weight', type=float, default=1.0,
                        help='行为克隆损失权重')
    
    # ========== 基础训练参数（使用配置常量作为默认值） ==========
    parser.add_argument('--exploration-noise', type=float, default=DEFAULT_EXPLORATION_NOISE,
                        help=f'探索噪声标准差 (默认{DEFAULT_EXPLORATION_NOISE})')
    parser.add_argument('--algorithm', type=str, default='diffusion_opt',
                        help='算法名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_SIZE,
                        help=f'经验回放缓冲区大小 (默认{DEFAULT_BUFFER_SIZE:,})')
    parser.add_argument('-e', '--epoch', type=int, default=50000,
                        help='总训练轮次')
    parser.add_argument('--step-per-epoch', type=int, default=1,
                        help='每个训练轮次的步数')
    parser.add_argument('--step-per-collect', type=int, default=1,
                        help='每次收集的步数')
    parser.add_argument('-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'批次大小 (默认{DEFAULT_BATCH_SIZE})')
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

    # ========== 网络架构参数（使用配置常量作为默认值） ==========
    parser.add_argument('--hidden-dim', type=int, default=DEFAULT_HIDDEN_DIM,
                        help=f'MLP隐藏层维度 (默认{DEFAULT_HIDDEN_DIM})')
    parser.add_argument('--actor-lr', type=float, default=DEFAULT_ACTOR_LR,
                        help=f'Actor学习率 (默认{DEFAULT_ACTOR_LR})')
    parser.add_argument('--critic-lr', type=float, default=DEFAULT_CRITIC_LR,
                        help=f'Critic学习率 (默认{DEFAULT_CRITIC_LR})')
    parser.add_argument('--reward-scale', type=float, default=DEFAULT_REWARD_SCALE,
                        help=f'奖励缩放系数（降低奖励尺度，稳定训练，默认{DEFAULT_REWARD_SCALE}）')

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
    
    args = parser.parse_args()
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
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    # 打印配置
    print("\n" + "=" * 60)
    print("  BEAR 建筑环境 HVAC 优化训练")
    print("=" * 60)
    print(f"\n配置参数:")
    pprint.pprint(vars(args))
    print()
    
    # ========== 创建环境 ==========
    print("正在创建环境...")
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
        training_num=args.training_num,
        test_num=args.test_num
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
        action_space=env.action_space,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        reward_normalization=False,  # Tianshou不支持N步回报+奖励归一化
    )
    
    print(f"✓ 策略创建成功")
    print(f"  算法: {args.algorithm}")
    print(f"  扩散步数: {args.diffusion_steps}")
    if args.bc_coef:
        print(f"  行为克隆权重: {args.bc_weight}")
    
    # ========== 创建收集器 ==========
    print("\n正在创建数据收集器...")
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    
    test_collector = Collector(policy, test_envs)
    
    print(f"✓ 收集器创建成功")
    print(f"  训练环境数: {args.training_num}")
    print(f"  测试环境数: {args.test_num}")
    print(f"  缓冲区大小: {args.buffer_size:,}")
    
    # ========== 开始训练 ==========
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"\n⚠️ 注意: 奖励已缩放 {args.reward_scale}x")
    print(f"  - 训练日志显示的是缩放后的奖励")
    print(f"  - 真实奖励 = 显示奖励 / {args.reward_scale}")
    print(f"  - 例如: test_reward=-20000 → 真实奖励=-200000\n")

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=1.0,
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

