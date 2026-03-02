#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于建筑环境的 SAC 基线，可选启用专家模仿（例如 MPC）。

- 保持原有的 rl_baseline 训练流程不变。
- 增加轻量级 BC 项，将 actor 拉向环境提供的专家动作。
- 专家动作来自 env.building_expert_controller 中的内置控制器（MPC、PID、rule、bangbang）。
"""

import argparse
import math
import sys
import os
import pprint
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from dropt_utils.paper_logging import add_paper_logging_args, run_paper_logging
from env.building_env_wrapper import make_building_env
from env.building_config import (
    DEFAULT_ACTOR_LR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BUILDING_TYPE,
    DEFAULT_CRITIC_LR,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_EPISODE_PER_TEST,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LOCATION,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_POWER,
    DEFAULT_MPC_PLANNING_STEPS,
    DEFAULT_N_STEP,
    DEFAULT_REWARD_SCALE,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_STEP_PER_COLLECT,
    DEFAULT_STEP_PER_EPOCH,
    DEFAULT_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE,
    DEFAULT_TEMP_WEIGHT,
    DEFAULT_TEST_NUM,
    DEFAULT_TRAINING_NUM,
    DEFAULT_VIOLATION_PENALTY,
    DEFAULT_WEATHER_TYPE,
    DEFAULT_TIME_RESOLUTION,
)


class LinearScheduler:
    """用于 BC 权重衰减的简单线性调度器。"""

    def __init__(self, start: float, end: float, steps: int) -> None:
        self.start = start
        self.end = end
        self.steps = max(1, steps)
        self.current_step = 0
        self.current = start

    def step(self) -> float:
        if self.current_step >= self.steps:
            self.current = self.end
            return self.current
        self.current_step += 1
        ratio = min(1.0, self.current_step / self.steps)
        self.current = self.start + (self.end - self.start) * ratio
        return self.current

    def get(self) -> float:
        return self.current


class SACImitationPolicy(SACPolicy):
    """在 actor 更新中可混入 BC 损失的 SAC。"""

    def __init__(
        self,
        *args,
        bc_coef: bool = False,
        bc_weight: float = 0.8,
        bc_weight_final: Optional[float] = 0.1,
        bc_weight_decay_steps: int = 50000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bc_enabled = bc_coef
        self.bc_weight = float(bc_weight)
        target = self.bc_weight if bc_weight_final is None else float(bc_weight_final)
        if bc_weight_decay_steps > 0 and not np.isclose(self.bc_weight, target):
            self.bc_scheduler = LinearScheduler(self.bc_weight, target, bc_weight_decay_steps)
        else:
            self.bc_scheduler = None
        self.bc_weight_target = target

    def _get_expert_actions(self, batch) -> Optional[torch.Tensor]:
        """如果存在，从 batch.info 中提取专家动作。"""
        infos = getattr(batch, "info", None)
        if infos is None:
            return None
        actions = []
        for item in infos:
            expert = None
            if isinstance(item, dict):
                expert = item.get("expert_action")
            else:
                expert = getattr(item, "expert_action", None)
                if expert is None and hasattr(item, "get"):
                    try:
                        expert = item.get("expert_action")
                    except Exception:
                        expert = None
            if expert is None:
                return None
            actions.append(expert)
        if not actions:
            return None
        device = next(self.actor.parameters()).device
        expert_tensor = torch.as_tensor(np.stack(actions, axis=0), dtype=torch.float32, device=device)
        return torch.clamp(expert_tensor, -1.0, 1.0)

    def _compute_bc_loss(self, policy_actions: torch.Tensor, batch) -> Optional[torch.Tensor]:
        expert_actions = self._get_expert_actions(batch)
        if expert_actions is None:
            return None
        return F.mse_loss(policy_actions, expert_actions)

    def learn(self, batch, **kwargs: Any) -> Dict[str, float]:
        if not self.bc_enabled:
            return super().learn(batch, **kwargs)

        # 更新第 1、2 个 critic（与 SACPolicy 相同）
        td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0

        # actor 更新时混入 BC
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss_pg = (
            self._alpha * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()

        bc_loss = self._compute_bc_loss(act, batch)
        bc_weight = self.bc_weight
        if self.bc_scheduler is not None:
            bc_weight = self.bc_scheduler.step()

        if bc_loss is not None:
            actor_loss = bc_weight * bc_loss + (1.0 - bc_weight) * actor_loss_pg
        else:
            actor_loss = actor_loss_pg
            bc_weight = 0.0

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/bc": bc_loss.item() if bc_loss is not None else 0.0,
            "bc_weight": bc_weight,
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()  # type: ignore
            result["alpha"] = self._alpha.item()  # type: ignore
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAC baseline with optional MPC/PID expert imitation on the BEAR building environment."
    )

    # 环境
    parser.add_argument("--building-type", type=str, default=DEFAULT_BUILDING_TYPE, help="Building type.")
    parser.add_argument("--weather-type", type=str, default=DEFAULT_WEATHER_TYPE, help="Weather pattern.")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="Geographic location.")
    parser.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP, help="Target indoor temperature (C).")
    parser.add_argument("--temp-tolerance", type=float, default=DEFAULT_TEMP_TOLERANCE, help="Comfort band (+/- C).")
    parser.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER, help="Max HVAC power (W).")
    parser.add_argument("--time-resolution", type=int, default=DEFAULT_TIME_RESOLUTION, help="Timestep in seconds.")
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH, help="Episode length in steps.")
    parser.add_argument("--energy-weight", type=float, default=DEFAULT_ENERGY_WEIGHT, help="Reward weight on energy use.")
    parser.add_argument("--temp-weight", type=float, default=DEFAULT_TEMP_WEIGHT, help="Reward weight on comfort.")
    parser.add_argument("--violation-penalty", type=float, default=DEFAULT_VIOLATION_PENALTY, help="Penalty per comfort violation.")
    parser.add_argument("--reward-scale", type=float, default=DEFAULT_REWARD_SCALE, help="Reward scaling factor.")
    parser.add_argument(
        "--add-violation-penalty",
        action="store_true",
        default=True,
        help="Enable temperature violation penalty (default on).",
    )
    parser.add_argument(
        "--no-add-violation-penalty", dest="add_violation_penalty", action="store_false", help="Disable violation penalty."
    )

    # 训练
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epoch", type=int, default=20000, help="Number of training epochs.")
    parser.add_argument("--step-per-epoch", type=int, default=DEFAULT_STEP_PER_EPOCH, help="Env steps per epoch.")
    parser.add_argument('--total-steps', type=int, default=None,
                        help='Total environment steps budget (overrides epoch if set)')
    parser.add_argument("--step-per-collect", type=int, default=DEFAULT_STEP_PER_COLLECT, help="Steps per data collection.")
    parser.add_argument("--episode-per-test", type=int, default=DEFAULT_EPISODE_PER_TEST, help="Episodes per evaluation.")
    parser.add_argument("--training-num", type=int, default=DEFAULT_TRAINING_NUM, help="Parallel training envs.")
    parser.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM, help="Parallel test envs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="Replay buffer size.")
    parser.add_argument("--update-per-step", type=float, default=0.5, help="Gradient steps per env step.")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP, help="N-step return.")

    # SAC 超参数
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="MLP hidden units.")
    parser.add_argument("--actor-lr", type=float, default=DEFAULT_ACTOR_LR, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=DEFAULT_CRITIC_LR, help="Critic learning rate.")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="Entropy temperature learning rate.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument("--target-entropy-scale", type=float, default=1.0, help="Entropy target multiplier.")
    parser.add_argument(
        "--reward-normalization",
        action="store_true",
        default=True,
        help="Enable reward normalization (incompatible with n-step>1).",
    )
    parser.add_argument(
        "--no-reward-normalization", dest="reward_normalization", action="store_false", help="Disable reward normalization."
    )

    # 专家模仿
    parser.add_argument(
        "--bc-coef",
        action="store_true",
        default=False,
        help="Enable behavior cloning loss using expert actions.",
    )
    parser.add_argument("--bc-weight", type=float, default=0.8, help="Initial BC weight in actor loss.")
    parser.add_argument(
        "--bc-weight-final",
        type=float,
        default=0.1,
        help="Final BC weight after decay (set equal to bc-weight to keep constant).",
    )
    parser.add_argument(
        "--bc-weight-decay-steps",
        type=int,
        default=50000,
        help="Linear decay steps for BC weight (0 to disable decay).",
    )
    parser.add_argument(
        "--expert-type",
        type=str,
        default="mpc",
        choices=["mpc", "pid", "rule", "bangbang"],
        help="Expert controller to generate guidance actions.",
    )
    parser.add_argument(
        "--mpc-planning-steps",
        type=int,
        default=DEFAULT_MPC_PLANNING_STEPS,
        help="Planning horizon (steps) for the MPC expert.",
    )

    # 日志 / 设备
    parser.add_argument("--prioritized-replay", action="store_true", default=False, help="Use PER replay buffer.")
    parser.add_argument("--prior-alpha", type=float, default=0.6, help="PER alpha.")
    parser.add_argument("--prior-beta", type=float, default=0.4, help="PER beta.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOG_DIR, help="Root log directory.")
    parser.add_argument("--log-prefix", type=str, default="sac_baseline_mpc", help="Log name prefix.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--vector-env-type", choices=["dummy", "subproc"], default="dummy", help="Vector env implementation."
    )
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodes for final evaluation.")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to resume model checkpoint.")

    add_paper_logging_args(parser)
    args = parser.parse_args()
    argv = sys.argv[1:]
    has_epoch_flag = any(arg in ('--epoch', '-e') for arg in argv)
    has_total_steps_flag = '--total-steps' in argv
    if not has_epoch_flag and not has_total_steps_flag:
        args.total_steps = 1_000_000
    if args.total_steps is not None and args.total_steps > 0:
        args.epoch = max(1, math.ceil(args.total_steps / args.step_per_epoch))
    if args.bc_weight_final is None:
        args.bc_weight_final = args.bc_weight
    return args


def _aggregate_metrics(vector_env) -> Optional[Dict[str, float]]:
    if vector_env is None:
        return None
    env_list = getattr(vector_env, "_env_list", None)
    if not env_list:
        return None
    values = [env_inst.consume_metrics() for env_inst in env_list]
    values = [m for m in values if m]
    if not values:
        return None

    def _avg(key: str) -> Optional[float]:
        nums = [m[key] for m in values if m.get(key) is not None]
        return float(np.mean(nums)) if nums else None

    result = {
        "avg_energy": _avg("avg_energy"),
        "avg_comfort_mean": _avg("avg_comfort_mean"),
        "avg_violations": _avg("avg_violations"),
        "avg_pue": _avg("avg_pue"),
    }
    return {k: v for k, v in result.items() if v is not None} or None


def make_logger(args: argparse.Namespace, train_envs, test_envs, log_path: str):
    writer = SummaryWriter(log_path)

    def metrics_getter(mode: str) -> Optional[Dict[str, float]]:
        vector_env = train_envs if mode == "train" else test_envs
        return _aggregate_metrics(vector_env)

    return EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,
        verbose=True,
        diffusion_steps=None,
        update_log_interval=50,
        step_per_epoch=args.step_per_epoch,
        metrics_getter=metrics_getter,
        png_interval=5,
    )


def build_sac_policy(args: argparse.Namespace, env, device: torch.device) -> SACPolicy:
    state_shape = env.observation_space.shape or (env.state_dim,)
    action_shape = env.action_space.shape or (env.action_dim,)
    max_action = float(np.max(np.abs(env.action_space.high)))

    # 策略网络（actor）
    actor_backbone = Net(
        state_shape=state_shape,
        hidden_sizes=[args.hidden_dim, args.hidden_dim],
        device=device,
    )
    actor = ActorProb(
        actor_backbone,
        action_shape,
        max_action=max_action,
        device=device,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # 价值网络（critic）
    critic1_backbone = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[args.hidden_dim, args.hidden_dim],
        device=device,
        concat=True,
    )
    critic2_backbone = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[args.hidden_dim, args.hidden_dim],
        device=device,
        concat=True,
    )
    critic1 = Critic(critic1_backbone, device=device).to(device)
    critic2 = Critic(critic2_backbone, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # 温度参数（alpha）
    target_entropy = -np.prod(action_shape) * args.target_entropy_scale
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACImitationPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        reward_normalization=args.reward_normalization,
        estimation_step=args.n_step,
        action_space=env.action_space,
        bc_coef=args.bc_coef,
        bc_weight=args.bc_weight,
        bc_weight_final=args.bc_weight_final,
        bc_weight_decay_steps=args.bc_weight_decay_steps,
    )
    return policy


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if getattr(args, "reward_normalization", False) and getattr(args, "n_step", 1) > 1:
        print("Warning: reward_normalization is disabled because n_step>1 is not supported together.")
        args.reward_normalization = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    expert_kwargs = None
    expert_type = args.expert_type if args.bc_coef else None
    if expert_type:
        expert_kwargs = {}
        if expert_type == "mpc":
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
        expert_type=expert_type,
        expert_kwargs=expert_kwargs,
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.log_prefix}_{args.building_type}_{args.weather_type}_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    logger = make_logger(args, train_envs, test_envs, log_path)

    print("\n" + "=" * 70)
    print("  SAC baseline (expert imitation optional) - BEAR building env")
    print("=" * 70)
    pprint.pprint(vars(args))

    policy = build_sac_policy(args, env, device)
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=device)
        policy.load_state_dict(ckpt)
        print(f"Resumed policy from {args.resume_path}")

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

    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs)

    last_paper_epoch = {"value": None}

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if epoch % max(1, DEFAULT_SAVE_INTERVAL) == 0:
            torch.save(
                {"model": policy.state_dict()},
                os.path.join(log_path, f"checkpoint_{epoch}.pth"),
            )
        if args.paper_log and args.paper_log_interval > 0 and epoch % args.paper_log_interval == 0:
            try:
                print(f"\n[paper-log] Epoch {epoch}: collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=None,
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
        save_best_fn=lambda p: torch.save(p.state_dict(), os.path.join(log_path, "policy_best.pth")),
        save_checkpoint_fn=save_checkpoint_fn,
    )

    print("\nTraining finished.")
    pprint.pprint(result)

    final_path = os.path.join(log_path, "policy_final.pth")
    torch.save(policy.state_dict(), final_path)

    if args.paper_log:
        try:
            if args.paper_log_interval > 0 and last_paper_epoch["value"] == args.epoch:
                print("[paper-log] Skipped final logging (already captured at last epoch).")
            else:
                print("\n[paper-log] Collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=None,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")
    print(f"Saved final policy to {final_path}")

    policy.eval()
    test_collector.reset()
    eval_res = test_collector.collect(n_episode=args.eval_episodes)
    metrics = _aggregate_metrics(test_envs)

    def _extract_eval_stats(res):
        if isinstance(res, dict):
            rews = res.get("rews", res.get("returns", res.get("rew")))
            lens = res.get("lens", res.get("len"))
            return rews, lens
        rews = getattr(res, "rews", None)
        if rews is None:
            rews = getattr(res, "returns", None)
        lens = getattr(res, "lens", None)
        return rews, lens

    eval_rews, eval_lens = _extract_eval_stats(eval_res)
    print("\nEvaluation summary:")
    print(f"  Episodes: {args.eval_episodes}")
    if eval_rews is not None:
        print(f"  Mean reward: {np.mean(eval_rews):.4f}")
    if eval_lens is not None:
        print(f"  Mean length: {np.mean(eval_lens):.1f}")
    if metrics:
        print("  Env metrics | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    else:
        print("  Env metrics: none (run more episodes to collect).")


if __name__ == "__main__":
    main()
