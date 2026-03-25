"""
BEAR building diffusion baseline (MLP backbone) with BC label alignment fix.

This script is intentionally independent from the existing training entrypoints:
- keeps the original `main_building.py` unchanged
- reuses the bcfix-clean environment wrapper only
- preserves the standard MLP diffusion baseline training flow
"""

import math
import os
import pprint
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from diffusion import Diffusion
from diffusion.model import DoubleCritic, MLP
from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.paper_logging import run_paper_logging
from dropt_utils.tianshou_compat import offpolicy_trainer
from main_building import get_args as base_get_args
from main_building_fno_guided_bcfix_clean import make_building_env_bcfix_clean
from policy import DiffusionOPT

warnings.filterwarnings("ignore")


def get_args():
    args = base_get_args()

    if args.log_prefix == "default":
        args.log_prefix = "diffusion_mlp_bcfix_clean"
    args.algorithm = "diffusion_mlp_bcfix_clean"

    # Align this baseline with the bcfix-clean comparison setting.
    if args.expert_type is None:
        args.expert_type = "mpc"
    args.bc_coef = True

    # Match the paper runs used by the FNO family for fair comparison.
    args.diffusion_steps = 6
    args.training_num = 1
    args.test_num = 1
    args.step_per_epoch = 4096
    args.step_per_collect = 1024
    args.log_update_interval = 200

    argv = sys.argv[1:]
    has_epoch_flag = any(arg in ("--epoch", "-e") for arg in argv)
    has_total_steps_flag = "--total-steps" in argv
    if not has_epoch_flag and not has_total_steps_flag:
        args.total_steps = 1_000_000
    if args.total_steps is not None and args.total_steps > 0:
        args.epoch = max(1, math.ceil(args.total_steps / args.step_per_epoch))

    return args


def main():
    args = get_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if getattr(args, "reward_normalization", False) and getattr(args, "n_step", 1) > 1:
        print("Warning: n_step>1 与奖励归一化不兼容，已自动关闭 reward_normalization")
        args.reward_normalization = False

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
    print("  BEAR Building + Diffusion MLP (bcfix clean)")
    print("=" * 60)
    pprint.pprint(vars(args))
    print()

    print("Creating BEAR environments ...")
    expert_kwargs = None
    if args.expert_type:
        expert_kwargs = {}
        if args.expert_type == "mpc":
            expert_kwargs["planning_steps"] = args.mpc_planning_steps

    env, train_envs, test_envs = make_building_env_bcfix_clean(
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

    print("\nBuilding MLP diffusion backbone ...")
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

    actor = MLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        t_dim=16,
    ).to(args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd,
    )

    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor,
        max_action=max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
    ).to(args.device)

    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd,
    )

    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion_actor,
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
    print(f"  algorithm={args.algorithm}, diffusion_steps={args.diffusion_steps}")

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

    warmup_noise_steps = 250_000

    def train_fn(epoch: int, env_step: int):
        if not hasattr(train_collector, "exploration_noise"):
            return
        enable_noise = env_step >= warmup_noise_steps
        if train_collector.exploration_noise != enable_noise:
            train_collector.exploration_noise = enable_noise
            status = "enabled" if enable_noise else "disabled"
            print(f"[train_fn] env_step={env_step}: exploration noise {status}")

    print(f"Replay buffer: {args.buffer_size:,} ({'PER' if args.prioritized_replay else 'uniform'})")

    print("\n" + "=" * 60)
    print("  Start training (Diffusion + MLP, bcfix clean)")
    print("=" * 60)

    last_paper_epoch = {"value": None}

    def save_checkpoint_fn(epoch, env_step, gradient_step):
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
                    actor=diffusion_actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                last_paper_epoch["value"] = epoch
            except Exception as exc:
                print(f"[paper-log] Failed at epoch {epoch}: {exc}")
        return None

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
            os.path.join(log_path, "policy_best.pth"),
        ),
        save_checkpoint_fn=save_checkpoint_fn,
        train_fn=train_fn,
    )

    print("\nTraining finished")
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
                    actor=diffusion_actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")

    print(f"Saved final model to: {log_path}")


if __name__ == "__main__":
    main()
