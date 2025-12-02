# ========================================
# 数据中心空调优化训练主程序
# ========================================
# 基于DROPT框架改造，应用扩散模型+强化学习

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

# 导入数据中心环境
from env.datacenter_env import make_datacenter_env

# 导入DROPT核心组件（复用）
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic

# 导入日志格式化工具
from utils.logger_formatter import EnhancedTensorboardLogger

warnings.filterwarnings('ignore')


def get_args():
    """
    解析命令行参数（针对数据中心场景调整）
    """
    parser = argparse.ArgumentParser(description='数据中心空调优化训练程序')
    
    # ========== 环境参数 ==========
    parser.add_argument('--num-crac', type=int, default=4,
                        help='CRAC空调单元数量')
    parser.add_argument('--target-temp', type=float, default=24.0,
                        help='目标温度 (°C)')
    parser.add_argument('--temp-tolerance', type=float, default=2.0,
                        help='温度容差 (°C)')
    parser.add_argument('--episode-length', type=int, default=288,
                        help='回合长度（步数，默认24小时）')
    parser.add_argument('--energy-weight', type=float, default=1.0,
                        help='能耗权重 α')
    parser.add_argument('--temp-weight', type=float, default=10.0,
                        help='温度偏差权重 β')
    parser.add_argument('--violation-penalty', type=float, default=100.0,
                        help='温度越界惩罚 γ')

    # ========== 专家控制器参数 ==========
    parser.add_argument('--expert-type', type=str, default='pid',
                        choices=['pid', 'mpc', 'rule_based'],
                        help='专家控制器类型')
    
    # ========== 基础训练参数 ==========
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                        help='探索噪声标准差')
    parser.add_argument('--algorithm', type=str, default='diffusion_opt',
                        help='算法名称')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='经验回放缓冲区大小')
    parser.add_argument('-e', '--epoch', type=int, default=100000,
                        help='总训练轮次')
    parser.add_argument('--step-per-epoch', type=int, default=1,
                        help='每个训练轮次的步数')
    parser.add_argument('--step-per-collect', type=int, default=1,
                        help='每次收集的步数')
    parser.add_argument('--update-per-step', type=float, default=1.0,
                        help='每个环境步的参数更新次数 (≤1 可降低计算)')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='权重衰减系数')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子（数据中心场景建议0.99）')
    parser.add_argument('--n-step', type=int, default=3,
                        help='N步TD学习')
    parser.add_argument('--training-num', type=int, default=4,
                        help='并行训练环境数量')
    parser.add_argument('--test-num', type=int, default=2,
                        help='并行测试环境数量')
    parser.add_argument('--episode-per-test', type=int, default=1,
                        help='评估时运行的episode数量')
    parser.add_argument('--vector-env-type', type=str, default='dummy',
                        choices=['dummy', 'subproc'],
                        help='向量环境实现方式')
    
    # ========== 网络架构参数 ==========
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='MLP隐藏层维度')
    
    # ========== 日志和设备参数 ==========
    parser.add_argument('--logdir', type=str, default='log_datacenter',
                        help='日志保存目录')
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
    parser.add_argument('--log-update-interval', type=int, default=50,
                        help='TensorBoard写入梯度指标的间隔（梯度步）')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='用于日志展示的奖励缩放系数')
    
    # ========== 扩散模型参数 ==========
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                        help='Actor学习率')
    parser.add_argument('--critic-lr', type=float, default=3e-4,
                        help='Critic学习率')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='目标网络软更新系数')
    parser.add_argument('-t', '--n-timesteps', type=int, default=10,
                        help='扩散时间步数（建议10-15，从5增加到10以提升生成质量）')
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'],
                        help='噪声调度策略')
    
    # ========== 训练模式 ==========
    parser.add_argument('--bc-coef', action='store_true', default=False,
                        help='是否使用行为克隆（True=有专家数据）')
    parser.add_argument('--bc-weight', type=float, default=1.0,
                        help='行为克隆损失权重（0=忽略BC，1=纯BC）')
    parser.add_argument('--bc-weight-final', type=float, default=None,
                        help='BC权重最终值（默认与初始值相同）')
    parser.add_argument('--bc-weight-decay-steps', type=int, default=0,
                        help='BC权重线性衰减步数（0表示不衰减）')
    
    # ========== 优先经验回放 ==========
    parser.add_argument('--prioritized-replay', action='store_true', default=False,
                        help='是否使用优先经验回放')
    parser.add_argument('--prior-alpha', type=float, default=0.6,
                        help='优先级采样alpha')
    parser.add_argument('--prior-beta', type=float, default=0.4,
                        help='重要性采样beta')
    
    args = parser.parse_known_args()[0]
    if args.bc_weight_final is None:
        args.bc_weight_final = args.bc_weight
    return args


def main(args=None):
    """
    数据中心空调优化主训练函数
    """
    if args is None:
        args = get_args()
    
    print("=" * 70)
    print("数据中心空调优化 - 基于扩散模型的强化学习")
    print("=" * 70)

    # ========== 创建数据中心环境 ==========
    print("\n[1/6] 创建数据中心环境...")

    # 准备环境参数
    env_kwargs = {
        'num_crac_units': args.num_crac,
        'target_temp': args.target_temp,
        'temp_tolerance': args.temp_tolerance,
        'episode_length': args.episode_length,
        'energy_weight': args.energy_weight,
        'temp_weight': args.temp_weight,
        'violation_penalty': args.violation_penalty,
        'expert_type': args.expert_type,
    }

    env, train_envs, test_envs = make_datacenter_env(
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
        **env_kwargs
    )
    
    # 获取状态和动作维度
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.0
    args.exploration_noise = args.exploration_noise * args.max_action
    
    print(f"  ✓ 状态维度: {args.state_shape}")
    print(f"  ✓ 动作维度: {args.action_shape}")
    print(f"  ✓ CRAC单元数: {args.num_crac}")
    print(f"  ✓ 目标温度: {args.target_temp}°C ± {args.temp_tolerance}°C")
    
    # ========== 设置随机种子 ==========
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
    # ========== 创建Actor网络（扩散模型） ==========
    print("\n[2/6] 初始化Actor网络（扩散模型）...")
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_dim  # 隐藏层维度
    )
    
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        bc_coef=args.bc_coef
    ).to(args.device)
    
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )
    
    print(f"  ✓ 扩散步数: {args.n_timesteps}")
    print(f"  ✓ 噪声调度: {args.beta_schedule}")
    print(f"  ✓ 网络结构: {args.hidden_dim}")
    
    # ========== 创建Critic网络（双Q网络） ==========
    print("\n[3/6] 初始化Critic网络（双Q网络）...")
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_dim
    ).to(args.device)
    
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )
    
    print(f"  ✓ Critic学习率: {args.critic_lr}")
    
    # ========== 设置日志系统 ==========
    print("\n[4/6] 配置日志系统...")
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_name = f"datacenter_{args.expert_type}_crac{args.num_crac}_t{args.n_timesteps}"
    log_path = os.path.join(args.logdir, args.log_prefix, log_name, time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    # 创建增强的日志记录器（美化终端输出）
    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,  # 每个epoch都输出（可改为10表示每10个epoch输出一次）
        verbose=True,  # True=详细格式，False=紧凑格式
        diffusion_steps=args.n_timesteps,
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch
    )

    print(f"  ✓ 日志路径: {log_path}")
    print(f"  ✓ TensorBoard: tensorboard --logdir={log_path}")
    print(f"  ✓ 日志输出已优化，关键指标将清晰显示")
    
    # ========== 定义策略 ==========
    print("\n[5/6] 初始化DiffusionOPT策略...")
    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        bc_weight=args.bc_weight,
        bc_weight_final=args.bc_weight_final,
        bc_weight_decay_steps=args.bc_weight_decay_steps,
        action_space=env.action_space,
        exploration_noise=args.exploration_noise,
    )
    
    # 加载预训练模型
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print(f"  ✓ 已加载模型: {args.resume_path}")
    
    # ========== 设置经验回放缓冲区 ==========
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
        print(f"  ✓ 使用优先经验回放 (alpha={args.prior_alpha})")
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )
        print(f"  ✓ 使用普通经验回放")
    
    # ========== 设置数据收集器 ==========
    train_collector = Collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)
    
    def save_best_fn(policy):
        """保存最佳模型"""
        save_path = os.path.join(log_path, 'policy_best.pth')
        torch.save(policy.state_dict(), save_path)
        print(f"  ✓ 保存最佳模型: {save_path}")
    
    # ========== 开始训练 ==========
    print("\n[6/6] 开始训练...")
    print("=" * 70)
    print(f"训练配置:")
    print(f"  - 设备: {args.device}")
    print(f"  - 训练模式: {'行为克隆（有专家数据）' if args.bc_coef else '策略梯度（无专家数据）'}")
    print(f"  - 专家控制器: {args.expert_type.upper()}")
    print(f"  - 总轮次: {args.epoch}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 折扣因子: {args.gamma}")
    print("=" * 70)
    
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.episode_per_test,
            args.batch_size,
            update_per_step=args.update_per_step,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        pprint.pprint(result)
        
        # 保存最终模型
        final_path = os.path.join(log_path, 'policy_final.pth')
        torch.save(policy.state_dict(), final_path)
        print(f"\n最终模型已保存: {final_path}")
    
    # ========== 推理/观察模式 ==========
    if args.watch:
        print("\n进入观察模式...")
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=3)
        
        print("\n观察结果:")
        print(f"  平均奖励: {result['rews'].mean():.2f}")
        print(f"  平均回合长度: {result['lens'].mean():.0f}")
        
        # 详细信息
        for i, info in enumerate(result['infos']):
            print(f"\n回合 {i+1}:")
            print(f"  累积能耗: {info.get('episode_energy', 0):.2f} kWh")
            print(f"  温度越界次数: {info.get('episode_violations', 0)}")


if __name__ == '__main__':
    main(get_args())

