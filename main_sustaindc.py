# ========================================
# SustainDC integration entry point
# ========================================
# Mirrors the BEAR + data center scripts but uses the SustainDCEnvWrapper.

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


def get_args():
    parser = argparse.ArgumentParser(description="Train DiffusionOPT on SustainDC")

    # SustainDC specific knobs
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Weather/CI location key (ny, ca, ...).")
    parser.add_argument("--month", type=int, default=DEFAULT_MONTH_INDEX, help="0-based month index used for init day sampling.")
    parser.add_argument("--days-per-episode", type=int, default=DEFAULT_DAYS_PER_EPISODE, help="Length of each SustainDC rollout (in days).")
    parser.add_argument("--timezone-shift", type=int, default=DEFAULT_TIMEZONE_SHIFT, help="Shift applied to the time-series managers.")
    parser.add_argument("--datacenter-capacity-mw", type=float, default=DEFAULT_DATACENTER_CAPACITY_MW, help="Scaling factor for the PyE+ data center.")
    parser.add_argument("--max-bat-cap-mw", type=float, default=DEFAULT_MAX_BAT_CAP_MW, help="Installed battery capacity (MW).")
    parser.add_argument("--flexible-load", type=float, default=DEFAULT_FLEXIBLE_LOAD_RATIO, help="Proportion of shiftable workload.")
    parser.add_argument("--individual-reward-weight", type=float, default=DEFAULT_INDIVIDUAL_REWARD_WEIGHT, help="Per-agent reward mixing coefficient.")
    parser.add_argument("--reward-aggregation", choices=["mean", "sum"], default=DEFAULT_REWARD_AGGREGATION, help="How to aggregate the three agent rewards.")
    parser.add_argument("--cintensity-file", type=str, default=DEFAULT_WRAPPER_CONFIG["cintensity_file"], help="Carbon intensity CSV located inside dc-rl-main/data/CarbonIntensity.")
    parser.add_argument("--weather-file", type=str, default=DEFAULT_WRAPPER_CONFIG["weather_file"], help="Weather EPW filename relative to data/Weather.")
    parser.add_argument("--workload-file", type=str, default=DEFAULT_WRAPPER_CONFIG["workload_file"], help="Workload CSV filename relative to data/Workload.")
    parser.add_argument("--dc-config-file", type=str, default=DEFAULT_WRAPPER_CONFIG["dc_config_file"], help="Data center json config relative to dc-rl-main/utils.")
    parser.add_argument("--action-threshold", type=float, default=0.33, help="Threshold used when mapping Diffusion outputs to SustainDC discrete actions.")

    # Training hyper-parameters (align with building defaults for consistency)
    parser.add_argument("--algorithm", type=str, default="diffusion_opt", help="Algorithm label used for logging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--training-num", type=int, default=DEFAULT_TRAINING_NUM, help="Number of parallel training envs.")
    parser.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM, help="Number of parallel evaluation envs.")
    parser.add_argument("--vector-env-type", choices=["dummy", "subproc"], default="dummy", help="Tianshou vector env backend.")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor.")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP, help="N-step TD length.")
    parser.add_argument("--epoch", type=int, default=20000, help="Total training epochs.")
    parser.add_argument("--step-per-epoch", type=int, default=DEFAULT_STEP_PER_EPOCH, help="Env steps collected per epoch.")
    parser.add_argument("--step-per-collect", type=int, default=DEFAULT_STEP_PER_COLLECT, help="Env steps per data collection call.")
    parser.add_argument("--episode-per-test", type=int, default=DEFAULT_EPISODE_PER_TEST, help="Episodes evaluated during testing.")
    parser.add_argument("--update-per-step", type=float, default=0.5, help="Gradient updates per environment step.")
    parser.add_argument("--exploration-noise", type=float, default=DEFAULT_EXPLORATION_NOISE, help="Std of Gaussian exploration noise.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension for actor/critic.")
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS, help="Number of diffusion steps.")
    parser.add_argument("--beta-schedule", type=str, default="vp", choices=["vp", "linear", "cosine"], help="Beta schedule for diffusion.")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Critic learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay for optimizers.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument("--logdir", type=str, default="log_sustaindc", help="Root logging directory.")
    parser.add_argument("--log-prefix", type=str, default="default", help="Extra folder prefix.")
    parser.add_argument("--log-update-interval", type=int, default=50, help="TensorBoard logging interval (gradient steps).")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Purely cosmetic scale used by the logger.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--resume-path", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--watch", action="store_true", default=False, help="Evaluation-only mode (skip training).")
    parser.add_argument("--lr-decay", action="store_true", default=False, help="Enable cosine LR decay.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Dump checkpoint every N epochs.")

    # Behaviour cloning and PER toggles
    parser.add_argument("--bc-coef", action="store_true", default=False, help="Enable behaviour cloning loss.")
    parser.add_argument("--bc-weight", type=float, default=0.8, help="Initial BC weight when bc-coef is true.")
    parser.add_argument("--bc-weight-final", type=float, default=None, help="Final BC weight after decay.")
    parser.add_argument("--bc-weight-decay-steps", type=int, default=50000, help="Linear decay schedule for BC weight.")
    parser.add_argument("--prioritized-replay", action="store_true", default=False, help="Use prioritized experience replay.")
    parser.add_argument("--prior-alpha", type=float, default=0.6, help="PER alpha.")
    parser.add_argument("--prior-beta", type=float, default=0.4, help="PER beta.")

    parser.add_argument("--reward-normalization", dest="reward_normalization", action="store_true")
    parser.add_argument("--no-reward-normalization", dest="reward_normalization", action="store_false")
    parser.set_defaults(reward_normalization=True)

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

    env, train_envs, test_envs = make_sustaindc_env(
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
        env_config=env_cfg,
        reward_aggregation=args.reward_aggregation,
        action_threshold=args.action_threshold,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"\n环境信息:")
    print(f"  Location: {args.location.upper()}, month index: {args.month}")
    print(f"  Observation dim: {state_dim}")
    print(f"  Action dim: {action_dim} (aggregated from 3 SustainDC agents)")
    print(f"  Reward aggregation: {args.reward_aggregation}")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Networks
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

    diffusion = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_backbone,
        max_action=max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
        bc_coef=args.bc_coef,
    ).to(args.device)

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

    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print(f"  ✅ 已加载模型: {args.resume_path}")

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

    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

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
    )

    def save_best_fn(policy_inst):
        torch.save(policy_inst.state_dict(), os.path.join(log_path, "policy_best.pth"))

    def save_checkpoint_fn(epoch, *_):
        if epoch % args.save_interval != 0:
            return None
        torch.save(
            {
                "model": policy.state_dict(),
                "optim_actor": actor_optim.state_dict(),
                "optim_critic": critic_optim.state_dict(),
            },
            os.path.join(log_path, f"checkpoint_{epoch}.pth"),
        )

    if args.watch:
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=args.episode_per_test)
        print("\n评估完成:")
        print(f"  平均奖励: {result['rews'].mean():.2f}")
        print(f"  平均回合长度: {result['lens'].mean():.1f}")
        return

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

    print("\n" + "=" * 60)
    print("训练完成".center(60, "="))
    print("=" * 60)
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))
    print(f"Final policy saved to: {log_path}")


if __name__ == "__main__":
    main()
