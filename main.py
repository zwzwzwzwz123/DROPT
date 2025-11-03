# ========================================
# 基于生成扩散模型的网络优化主程序
# ========================================
# 本程序实现了使用扩散模型解决无线网络功率分配问题
# 支持有/无专家数据两种训练模式

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
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env  # 环境创建函数
from policy import DiffusionOPT  # 扩散优化策略
from diffusion import Diffusion  # 扩散模型
from diffusion.model import MLP, DoubleCritic  # Actor和Critic网络
import warnings
from argparse import Namespace
# 忽略警告信息，保持输出清晰
warnings.filterwarnings('ignore')

def get_args():
    """
    解析命令行参数
    返回包含所有训练配置的参数对象
    """
    parser = argparse.ArgumentParser(description='基于扩散模型的网络优化训练程序')
    
    # ========== 基础训练参数 ==========
    parser.add_argument("--exploration-noise", type=float, default=0.1,
                        help='探索噪声的标准差')
    parser.add_argument('--algorithm', type=str, default='diffusion_opt',
                        help='算法名称')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--buffer-size', type=int, default=1e6,
                        help='经验回放缓冲区大小')
    parser.add_argument('-e', '--epoch', type=int, default=1e6,
                        help='总训练轮次')
    parser.add_argument('--step-per-epoch', type=int, default=1,
                        help='每个训练轮次的步数')
    parser.add_argument('--step-per-collect', type=int, default=1,
                        help='每次收集的步数')
    parser.add_argument('-b', '--batch-size', type=int, default=512,
                        help='批次大小')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='权重衰减系数（L2正则化）')
    parser.add_argument('--gamma', type=float, default=1,
                        help='折扣因子')
    parser.add_argument('--n-step', type=int, default=3,
                        help='N步TD学习的步数')
    parser.add_argument('--training-num', type=int, default=1,
                        help='并行训练环境数量')
    parser.add_argument('--test-num', type=int, default=1,
                        help='并行测试环境数量')
    
    # ========== 日志和设备参数 ==========
    parser.add_argument('--logdir', type=str, default='log',
                        help='日志保存目录')
    parser.add_argument('--log-prefix', type=str, default='default',
                        help='日志文件前缀')
    parser.add_argument('--render', type=float, default=0.1,
                        help='渲染参数')
    parser.add_argument('--rew-norm', type=int, default=0,
                        help='是否标准化奖励（0=否，1=是）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备（cuda:0/cpu）')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='观察模式（不训练，仅推理）')
    parser.add_argument('--lr-decay', action='store_true', default=False,
                        help='是否使用学习率衰减')
    parser.add_argument('--note', type=str, default='',
                        help='备注信息')

    # ========== 扩散模型专用参数 ==========
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='Actor网络学习率')
    parser.add_argument('--critic-lr', type=float, default=1e-4,
                        help='Critic网络学习率')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='目标网络软更新系数')
    parser.add_argument('-t', '--n-timesteps', type=int, default=6,
                        help='扩散时间步数（关键参数，可选：3/6/8/12）')
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'],
                        help='噪声调度策略：linear/cosine/vp（variance preserving）')

    # ========== 训练模式选择 ==========
    # bc-coef=True: 有专家数据模式（行为克隆）
    # bc-coef=False: 无专家数据模式（策略梯度）
    parser.add_argument('--bc-coef', default=False,
                        help='是否使用行为克隆损失（True=有专家数据，False=无专家数据）')

    # ========== 优先经验回放参数 ==========
    parser.add_argument('--prioritized-replay', action='store_true', default=False,
                        help='是否使用优先经验回放')
    parser.add_argument('--prior-alpha', type=float, default=0.4,
                        help='优先级采样的alpha参数')
    parser.add_argument('--prior-beta', type=float, default=0.4,
                        help='重要性采样的beta参数')

    # 解析并返回参数
    args = parser.parse_known_args()[0]
    return args


#def main(args=get_args()):

def main(args=None, update_output=None, should_stop_training=None, **kwargs):
    """主训练函数"""
    # GUI 传 dict，命令行传 None
    if args is None:
        args = get_args()
    elif isinstance(args, dict):
        args = Namespace(**args)

    # 简单防御：如果没传输出函数，默认用 print
    if update_output is None:
        update_output = print
    """
    主训练函数
    1. 创建环境
    2. 初始化Actor（扩散模型）和Critic（双Q网络）
    3. 配置训练器
    4. 执行训练或推理
    """
    # ========== 创建环境 ==========
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    
    # 获取环境的状态和动作维度
    args.state_shape = env.observation_space.shape[0]  # 状态维度（信道数+奖励）
    args.action_shape = env.action_space.n  # 动作维度（功率分配方案维度）
    args.max_action = 1.  # 动作的最大值（归一化）

    # 计算实际的探索噪声大小
    args.exploration_noise = args.exploration_noise * args.max_action
    
    # ========== 随机种子设置（已注释） ==========
    # 注：如果需要完全可复现的结果，取消注释以下代码
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # ========== 创建Actor网络（扩散模型） ==========
    # Actor使用MLP作为去噪网络
    actor_net = MLP(
        state_dim=args.state_shape,  # 输入：状态维度
        action_dim=args.action_shape  # 输出：动作维度
    )
    
    # 将MLP包装成扩散模型
    # 核心思想：通过多步去噪过程，将随机噪声转换为最优动作
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,  # 去噪网络
        max_action=args.max_action,  # 动作范围
        beta_schedule=args.beta_schedule,  # 噪声调度策略
        n_timesteps=args.n_timesteps,  # 扩散步数（越大越精确但越慢）
        bc_coef=args.bc_coef  # 是否使用行为克隆
    ).to(args.device)
    
    # Actor优化器（使用AdamW，带权重衰减）
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )

    # ========== 创建Critic网络（双Q网络） ==========
    # 使用双Q网络减少Q值过估计问题
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    
    # Critic优化器
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    # ========== 设置日志系统 ==========
    time_now = datetime.now().strftime('%b%d-%H%M%S')  # 当前时间戳
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)  # TensorBoard写入器
    writer.add_text("args", str(args))  # 记录所有参数
    logger = TensorboardLogger(writer)  # Tianshou日志记录器

    # ========== 定义策略 ==========
    # DiffusionOPT整合了Actor（扩散模型）和Critic（双Q网络）
    policy = DiffusionOPT(
        args.state_shape,  # 状态维度
        actor,  # Actor网络（扩散模型）
        actor_optim,  # Actor优化器
        args.action_shape,  # 动作维度
        critic,  # Critic网络（双Q网络）
        critic_optim,  # Critic优化器
        args.device,  # 计算设备
        tau=args.tau,  # 目标网络软更新系数
        gamma=args.gamma,  # 折扣因子
        estimation_step=args.n_step,  # N步TD估计
        lr_decay=args.lr_decay,  # 是否使用学习率衰减
        lr_maxt=args.epoch,  # 学习率衰减的最大步数
        bc_coef=args.bc_coef,  # 训练模式（True=行为克隆，False=策略梯度）
        action_space=env.action_space,  # 动作空间
        exploration_noise=args.exploration_noise,  # 探索噪声
    )

    # ========== 加载预训练模型（如果提供） ==========
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("已加载模型：", args.resume_path)

    # ========== 设置经验回放缓冲区 ==========
    if args.prioritized_replay:
        # 优先经验回放：优先采样重要的经验
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,  # 优先级指数
            beta=args.prior_beta,  # 重要性采样指数
        )
    else:
        # 普通经验回放：均匀采样
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # ========== 设置数据收集器 ==========
    # 训练收集器：从训练环境收集经验并存入缓冲区
    train_collector = Collector(policy, train_envs, buffer)
    # 测试收集器：从测试环境评估策略性能
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        """保存最佳模型的回调函数"""
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # ========== 开始训练 ==========
    if not args.watch:
        # 训练模式
        print("=" * 50)
        print("开始训练...")
        print(f"设备: {args.device}")
        print(f"扩散步数: {args.n_timesteps}")
        print(f"训练模式: {'行为克隆（有专家数据）' if args.bc_coef else '策略梯度（无专家数据）'}")
        print("=" * 50)
        
        result = offpolicy_trainer(
            policy,  # 策略对象
            train_collector,  # 训练数据收集器
            test_collector,  # 测试数据收集器
            args.epoch,  # 总训练轮次
            args.step_per_epoch,  # 每轮步数
            args.step_per_collect,  # 每次收集步数
            args.test_num,  # 测试回合数
            args.batch_size,  # 批次大小
            save_best_fn=save_best_fn,  # 保存最佳模型的回调
            logger=logger,  # 日志记录器
            test_in_train=False  # 训练时不测试
        )
        pprint.pprint(result)

    # ========== 推理/观察模式 ==========
    # 使用方法: python main.py --watch --resume-path log/default/diffusion/XXX/policy.pth
    if __name__ == '__main__':
        policy.eval()  # 设置为评估模式
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)  # 收集1个回合的数据
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"最终奖励: {rews.mean():.4f}, 回合长度: {lens.mean():.0f}")


if __name__ == '__main__':
    main(get_args())
