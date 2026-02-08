# ========================================
# SustainDC 集成入口
# ========================================
# 复用 BEAR + 数据中心脚本，并切换为 SustainDCEnvWrapper。

import argparse
import os
import pprint
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from env.building_config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_EPISODE_PER_TEST,
    DEFAULT_EXPLORATION_NOISE,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LOG_DIR,
    DEFAULT_N_STEP,
    DEFAULT_STEP_PER_COLLECT,
    DEFAULT_STEP_PER_EPOCH,
    DEFAULT_TEST_NUM,
    DEFAULT_TRAINING_NUM,
)
from env.sustaindc_config import (
    DEFAULT_DAYS_PER_EPISODE,
    DEFAULT_DATACENTER_CAPACITY_MW,
    DEFAULT_FLEXIBLE_LOAD_RATIO,
    DEFAULT_INDIVIDUAL_REWARD_WEIGHT,
    DEFAULT_LOCATION,
    DEFAULT_MAX_BAT_CAP_MW,
    DEFAULT_MONTH_INDEX,
    DEFAULT_REWARD_AGGREGATION,
    DEFAULT_TIMEZONE_SHIFT,
    DEFAULT_WRAPPER_CONFIG,
)
from env.sustaindc_env_wrapper import make_sustaindc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from dropt_utils.paper_logging import add_paper_logging_args, run_paper_logging


def get_args():
    parser = argparse.ArgumentParser(description="Train DiffusionOPT on SustainDC")

    # SustainDC 特定参数选项
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="天气/碳强度区域标识（如 ny、ca 等）。")
    parser.add_argument("--month", type=int, default=DEFAULT_MONTH_INDEX, help="用于采样初始天数的 0 基月份索引。")
    parser.add_argument("--days-per-episode", type=int, default=DEFAULT_DAYS_PER_EPISODE, help="单个 SustainDC rollout 的天数。")
    parser.add_argument("--timezone-shift", type=int, default=DEFAULT_TIMEZONE_SHIFT, help="时间序列管理器使用的时区平移量。")
    parser.add_argument("--datacenter-capacity-mw", type=float, default=DEFAULT_DATACENTER_CAPACITY_MW, help="PyE+ 数据中心的负载缩放系数。")
    parser.add_argument("--max-bat-cap-mw", type=float, default=DEFAULT_MAX_BAT_CAP_MW, help="电池装机容量（MW）。")
    parser.add_argument("--flexible-load", type=float, default=DEFAULT_FLEXIBLE_LOAD_RATIO, help="可移峰（可调度）负载占比。")
    parser.add_argument("--individual-reward-weight", type=float, default=DEFAULT_INDIVIDUAL_REWARD_WEIGHT, help="单个智能体奖励与全局奖励的混合权重。")
    parser.add_argument("--reward-aggregation", choices=["mean", "sum"], default=DEFAULT_REWARD_AGGREGATION, help="三个智能体奖励的聚合方式。")
    parser.add_argument("--cintensity-file", type=str, default=DEFAULT_WRAPPER_CONFIG["cintensity_file"], help="dc-rl-main/data/CarbonIntensity 下的碳强度 CSV 文件名。")
    parser.add_argument("--weather-file", type=str, default=DEFAULT_WRAPPER_CONFIG["weather_file"], help="data/Weather 目录下天气 EPW 文件名。")
    parser.add_argument("--workload-file", type=str, default=DEFAULT_WRAPPER_CONFIG["workload_file"], help="data/Workload 目录下的工作负载 CSV 文件名。")
    parser.add_argument("--dc-config-file", type=str, default=DEFAULT_WRAPPER_CONFIG["dc_config_file"], help="dc-rl-main/utils 下数据中心 JSON 配置文件。")
    parser.add_argument("--action-threshold", type=float, default=0.33, help="将扩散策略输出映射为 SustainDC 离散动作时使用的阈值。")

    # 训练超参数（与建筑环境默认值保持一致）
    parser.add_argument("--algorithm", type=str, default="diffusion_opt", help="用于日志标记的算法名称。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--training-num", type=int, default=DEFAULT_TRAINING_NUM, help="并行训练环境数量。")
    parser.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM, help="并行评估环境数量。")
    parser.add_argument("--vector-env-type", choices=["dummy", "subproc"], default="dummy", help="Tianshou 向量环境后端类型。")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="经验回放缓冲区容量。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="最小批量大小。")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="折扣因子。")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP, help="N-step TD 长度。")
    parser.add_argument("--epoch", type=int, default=20000, help="总训练轮次。")
    parser.add_argument("--step-per-epoch", type=int, default=DEFAULT_STEP_PER_EPOCH, help="每个 epoch 采集的环境步数。")
    parser.add_argument("--step-per-collect", type=int, default=DEFAULT_STEP_PER_COLLECT, help="每次 collect 调用采集的步数。")
    parser.add_argument("--episode-per-test", type=int, default=DEFAULT_EPISODE_PER_TEST, help="测试阶段评估的回合数量。")
    parser.add_argument("--update-per-step", type=float, default=0.5, help="每个环境步的梯度更新次数。")
    parser.add_argument("--exploration-noise", type=float, default=DEFAULT_EXPLORATION_NOISE, help="高斯探索噪声标准差。")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Actor/Critic 的隐藏层维度。")
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS, help="扩散步骤数量。")
    parser.add_argument("--beta-schedule", type=str, default="vp", choices=["vp", "linear", "cosine"], help="扩散噪声 beta 调度方式。")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor 学习率。")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Critic 学习率。")
    parser.add_argument("--wd", type=float, default=1e-4, help="优化器权重衰减系数。")
    parser.add_argument("--tau", type=float, default=0.005, help="Target 网络软更新系数。")
    parser.add_argument("--logdir", type=str, default="log_sustaindc", help="日志根目录。")
    parser.add_argument("--log-prefix", type=str, default="default", help="额外的日志目录前缀。")
    parser.add_argument("--log-update-interval", type=int, default=50, help="TensorBoard 记录间隔（梯度步数）。")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="仅用于日志展示的奖励缩放系数。")
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备（Torch）。")
    parser.add_argument("--resume-path", type=str, default=None, help="可选的断点恢复模型路径。")
    parser.add_argument("--watch", action="store_true", default=False, help="仅执行评估，跳过训练。")
    parser.add_argument("--lr-decay", action="store_true", default=False, help="启用余弦学习率衰减。")
    parser.add_argument("--save-interval", type=int, default=1000, help="每隔 N 个 epoch 保存一次检查点。")

    # 行为克隆与优先级重放开关
    parser.add_argument("--bc-coef", action="store_true", default=False, help="启用行为克隆损失。")
    parser.add_argument("--bc-weight", type=float, default=0.8, help="开启 BC 后的初始权重。")
    parser.add_argument("--bc-weight-final", type=float, default=None, help="衰减完成后的 BC 最终权重。")
    parser.add_argument("--bc-weight-decay-steps", type=int, default=50000, help="BC 权重的线性衰减步数。")
    parser.add_argument("--prioritized-replay", action="store_true", default=False, help="启用优先级经验回放。")
    parser.add_argument("--prior-alpha", type=float, default=0.6, help="PER 的 alpha 参数。")
    parser.add_argument("--prior-beta", type=float, default=0.4, help="PER 的 beta 参数。")

    parser.add_argument("--reward-normalization", dest="reward_normalization", action="store_true")
    parser.add_argument("--no-reward-normalization", dest="reward_normalization", action="store_false")
    parser.set_defaults(reward_normalization=True)

    add_paper_logging_args(parser)
    args = parser.parse_args()
    if args.bc_weight_final is None:
        args.bc_weight_final = args.bc_weight
    if args.reward_normalization and args.n_step > 1:
        print("⚠️  n_step>1 与奖励归一化不兼容，已关闭 reward_normalization")
        args.reward_normalization = False
    return args


def main(args=None):
    args = get_args() if args is None else args
    print("=" * 80)
    print(" SustainDC 训练启动 ".center(80, "="))
    print("=" * 80)

    # ========== 构建 SustainDC 环境配置 ==========
    # 该字典会传入 env wrapper，用于统一管理位置、月份等关键配置。
    env_cfg = {
        "location": args.location,
        "month": args.month,
        "days_per_episode": args.days_per_episode,
        "timezone_shift": args.timezone_shift,
        "datacenter_capacity_mw": args.datacenter_capacity_mw,
        "max_bat_cap_Mw": args.max_bat_cap_mw,
        "flexible_load": args.flexible_load,
        "individual_reward_weight": args.individual_reward_weight,
        "cintensity_file": args.cintensity_file,
        "weather_file": args.weather_file,
        "workload_file": args.workload_file,
        "dc_config_file": args.dc_config_file,
    }

    # ========== 创建向量化 SustainDC 环境 ==========
    env, train_envs, test_envs = make_sustaindc_env(
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
        env_config=env_cfg,
        reward_aggregation=args.reward_aggregation,
        action_threshold=args.action_threshold,
    )

    # ========== 记录环境张量维度信息 ==========
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"\n环境信息:")
    print(f"  Location: {args.location.upper()}, month index: {args.month}")
    print(f"  Observation dim: {state_dim}")
    print(f"  Action dim: {action_dim} (aggregated from 3 SustainDC agents)")
    print(f"  Reward aggregation: {args.reward_aggregation}")

    # 随机种子设定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # 神经网络结构
    print("\n创建神经网络...")
    actor_backbone = MLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        t_dim=16,
    ).to(args.device)
    actor_optim = torch.optim.AdamW(actor_backbone.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    print(f"  Actor params: {sum(p.numel() for p in actor_backbone.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # ========== 构建扩散式 Actor ==========
    # 仍沿用 DROPT 原生组件，通过 n_timesteps/beta_schedule 控制生成质量。
    diffusion = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_backbone,
        max_action=max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
        bc_coef=args.bc_coef,
    ).to(args.device)

    # ========== 组装 DiffusionOPT 策略 ==========
    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=args.device,
        tau=args.tau,
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

    # ========== 可选的断点续训 ==========
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print(f"  已加载模型: {args.resume_path}")

    # ========== 构建经验回放缓冲区 ==========
    buffer_num = max(1, args.training_num)
    if args.prioritized_replay:
        replay_buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=buffer_num,
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        replay_buffer = VectorReplayBuffer(args.buffer_size, buffer_num=buffer_num)

    # ========== 封装数据收集器 ==========
    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # ========== 初始化日志与监控 ==========
    time_now = datetime.now().strftime("%b%d-%H%M%S")
    log_name = f"sustaindc_{args.location}_m{args.month}"
    log_path = os.path.join(args.logdir, args.log_prefix, log_name, time_now)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,
        verbose=True,
        diffusion_steps=args.diffusion_steps,
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch,
        png_interval=5,
    )

    # ========== 训练辅助回调 ==========
    def save_best_fn(policy_inst):
        torch.save(policy_inst.state_dict(), os.path.join(log_path, "policy_best.pth"))

    last_paper_epoch = {"value": None}

    def save_checkpoint_fn(epoch, *_):
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim_actor": actor_optim.state_dict(),
                    "optim_critic": critic_optim.state_dict(),
                },
                os.path.join(log_path, f"checkpoint_{epoch}.pth"),
            )
        if args.paper_log and args.paper_log_interval > 0 and epoch % args.paper_log_interval == 0:
            try:
                print(f"\n[paper-log] Epoch {epoch}: collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=diffusion,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                last_paper_epoch["value"] = epoch
            except Exception as exc:
                print(f"[paper-log] Failed at epoch {epoch}: {exc}")
        return None

    # ========== 仅评估模式 ==========
    if args.watch:
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=args.episode_per_test)
        print("\n评估完成:")
        print(f"  平均奖励: {result['rews'].mean():.2f}")
        print(f"  平均回合长度: {result['lens'].mean():.1f}")
        return

    # ========== 启动训练 ==========
    print("\n" + "=" * 60)
    print("开始训练".center(60, "="))
    print("=" * 60)
    print(f"日志目录: {log_path}")

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
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    # ========== 汇总并保存模型 ==========
    print("\n" + "=" * 60)
    print("训练完成".center(60, "="))
    print("=" * 60)
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

    if args.paper_log:
        try:
            if args.paper_log_interval > 0 and last_paper_epoch["value"] == args.epoch:
                print("[paper-log] Skipped final logging (already captured at last epoch).")
            else:
                print("\n[paper-log] Collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=diffusion,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")
    print(f"Final policy saved to: {log_path}")


if __name__ == "__main__":
    main()
