"""
BEAR building training entry that swaps the DDPM actor for a Rectified Flow actor.

This keeps the rest of the training stack intact (collectors, critic, logger)
while isolating the new flow-based policy in a separate script to avoid
impacting existing runs.
"""

import os
import pprint
import warnings
from datetime import datetime

import numpy as np
import torch
import sys
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# Reuse the original argument definitions from main_building.
from main_building import get_args as base_get_args

from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from env.building_env_wrapper import make_building_env
from policy import DiffusionOPT
from diffusion.model import DoubleCritic, MLP
from diffusion.rectified_flow import RectifiedFlow

warnings.filterwarnings("ignore")


def _parse_rf_args():
    """RF 专属参数（不修改原 main_building）"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    # 调低默认时间缩放 + 噪声，减少高频抖动；默认步数与 diffusion_steps 对齐以提速
    parser.add_argument("--rf-time-scale", type=float, default=10.0, help="时间缩放，论文默认 999")
    parser.add_argument("--rf-noise-scale", type=float, default=0.5, help="初始噪声尺度")
    parser.add_argument("--rf-sigma-var", type=float, default=0.05, help="sigma_t=(1-t)*sigma_var")
    parser.add_argument("--rf-sampler", type=str, default="euler", choices=["euler", "rk45"], help="采样器")
    parser.add_argument("--rf-sample-N", type=int, default=None, help="采样步数覆盖（None=diffusion_steps）")
    parser.add_argument("--rf-reflow", action="store_true", default=False, help="启用 reflow/蒸馏（需要教师数据）")
    parser.add_argument(
        "--rf-reflow-t-schedule",
        type=str,
        default="uniform",
        help="reflow 时间调度：t0/t1/uniform/整数k",
    )
    parser.add_argument(
        "--rf-reflow-loss",
        type=str,
        default="l2",
        choices=["l2", "lpips", "lpips+l2"],
        help="reflow 损失类型（LPIPS 需 pip install lpips）",
    )
    args, _ = parser.parse_known_args()
    return args, parser


def get_args():
    saved_argv = sys.argv
    rf_args, rf_parser = _parse_rf_args()
    # 先从 sys.argv 中提取 RF 参数，其余交给 base_get_args
    rf_parsed, remaining = rf_parser.parse_known_args(saved_argv[1:])
    sys.argv = [saved_argv[0]] + remaining
    args = base_get_args()
    sys.argv = saved_argv
    rf_args = rf_parsed
    # Make runs distinguishable from the DDPM-based pipeline.
    if args.log_prefix == "default":
        # 与 DDPM/其他版本区分的默认前缀
        args.log_prefix = "rectified_flow_mpc"
    # 默认使用 CPU；如需 GPU 请在命令行显式指定 --device cuda:0/1
    if args.device == "cuda:0":
        args.device = "cpu"
    args.algorithm = "rectified_flow_opt"
    # 默认开启专家模仿：若未指定则启用 MPC + BC 引导
    if args.expert_type is None:
        args.expert_type = "mpc"
    args.bc_coef = True
    # 训练/采样步数进一步压缩，提升速度；仅在用户未手动指定时生效
    if args.diffusion_steps >= 10:  # 默认 10
        args.diffusion_steps = 2
    # attach RF configs
    args.rf_time_scale = rf_args.rf_time_scale
    args.rf_noise_scale = rf_args.rf_noise_scale
    args.rf_sigma_var = rf_args.rf_sigma_var
    args.rf_sampler = rf_args.rf_sampler
    args.rf_sample_N = rf_args.rf_sample_N or args.diffusion_steps
    args.rf_reflow = rf_args.rf_reflow
    args.rf_reflow_t_schedule = (
        rf_args.rf_reflow_t_schedule if not rf_args.rf_reflow_t_schedule.isdigit() else int(rf_args.rf_reflow_t_schedule)
    )
    args.rf_reflow_loss = rf_args.rf_reflow_loss
    return args


def main():
    args = get_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.log_prefix}_{args.building_type}_{args.weather_type}_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)
    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,
        verbose=True,
        diffusion_steps=args.diffusion_steps,
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch,
        metrics_getter=None,
        png_interval=5,
    )

    print("\n" + "=" * 60)
    print("  BEAR Building + Rectified Flow")
    print("=" * 60)
    pprint.pprint(vars(args))
    print(f"采样器: {args.rf_sampler}, flow_steps={args.diffusion_steps}, time_scale={args.rf_time_scale}")
    print()

    # Environment ---------------------------------------------------------
    print("Creating BEAR environments ...")
    expert_kwargs = None
    if args.expert_type:
        expert_kwargs = {}
        if args.expert_type == "mpc":
            expert_kwargs["planning_steps"] = args.mpc_planning_steps

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
        reward_scale=args.reward_scale,
        expert_type=args.expert_type if args.bc_coef else None,
        expert_kwargs=expert_kwargs,
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
    )

    print("Environments ready")
    print(f"  state_dim={env.state_dim}, action_dim={env.action_dim}, rooms={env.roomnum}")

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
        for key in ("avg_energy", "avg_comfort_mean", "avg_violations", "avg_pue"):
            nums = [m[key] for m in values if m.get(key) is not None]
            if nums:
                result[key] = float(np.mean(nums))
        return result if result else None

    def metrics_getter(mode: str):
        target_env = train_envs if mode == "train" else test_envs
        return _aggregate_metrics(target_env)

    logger.training_logger.metrics_getter = metrics_getter

    # Networks ------------------------------------------------------------
    print("\nBuilding networks ...")
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

    velocity_net = MLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        t_dim=16,
    ).to(args.device)

    actor = RectifiedFlow(
        state_dim=state_dim,
        action_dim=action_dim,
        model=velocity_net,
        max_action=max_action,
        n_timesteps=args.diffusion_steps,
        loss_type="l2",
        time_scale=args.rf_time_scale,
        noise_scale=args.rf_noise_scale,
        use_ode_sampler=args.rf_sampler,
        sigma_var=args.rf_sigma_var,
        sample_N=args.rf_sample_N,
        reflow_flag=args.rf_reflow,
        reflow_t_schedule=args.rf_reflow_t_schedule,
        reflow_loss=args.rf_reflow_loss,
    ).to(args.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    print(f"  Velocity net params: {sum(p.numel() for p in velocity_net.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # Policy --------------------------------------------------------------
    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=actor,
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

    print("\nPolicy ready")
    print(f"  algorithm={args.algorithm}, flow_steps={args.diffusion_steps}, sampler={args.rf_sampler}")

    # Replay buffers & collectors ----------------------------------------
    print("\nPreparing collectors ...")
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

    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # 更长的 warmup：在专家引导阶段关闭额外探索噪声，稳定收敛
    warmup_noise_steps = 200_000

    def train_fn(epoch: int, env_step: int):
        if not hasattr(train_collector, "exploration_noise"):
            return
        enable_noise = env_step >= warmup_noise_steps
        if train_collector.exploration_noise != enable_noise:
            train_collector.exploration_noise = enable_noise
            status = "enabled" if enable_noise else "disabled"
            print(f"[train_fn] env_step={env_step}: exploration noise {status}")

    print(f"Replay buffer: {args.buffer_size:,} ({'PER' if args.prioritized_replay else 'uniform'})")

    # Training -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Start training (Rectified Flow)")
    print("=" * 60)

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
            os.path.join(log_path, "policy_best_rf.pth"),
        ),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: torch.save(
            {
                "model": policy.state_dict(),
                "optim_actor": actor_optim.state_dict(),
                "optim_critic": critic_optim.state_dict(),
            },
            os.path.join(log_path, f"checkpoint_{epoch}.pth"),
        )
        if epoch % args.save_interval == 0
        else None,
        train_fn=train_fn,
    )

    print("\nTraining finished")
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final_rf.pth"))
    print(f"Saved final model to: {log_path}")


if __name__ == "__main__":
    main()
