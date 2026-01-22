"""
BEAR 建筑环境：扩散策略 + DiffFNO + Critic Guidance（安全/性能引导）。

在 main_building_fno 基础上：
- 增加 guidance_scale/类型参数，默认 0（不启用）。
- guidance_fn 默认使用 critic 的 Q 梯度作为引导（提升回报=减少违规）。
保持原训练逻辑，作为独立脚本，不影响现有流程。
"""

import os
import pprint
import sys
import warnings
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from main_building import get_args as base_get_args

from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from env.building_env_wrapper import make_building_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.model_fno import DiffFNO

warnings.filterwarnings("ignore")


def _parse_fno_args():
    """仅解析 FNO / guidance 特定参数，避免干扰原始 CLI。"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fno-modes", type=int, default=4, help="保留的低频数量")
    parser.add_argument("--fno-width", type=int, default=48, help="频域通道宽度")
    parser.add_argument("--fno-layers", type=int, default=1, help="谱卷积层数")
    parser.add_argument("--fno-activation", type=str, default="mish", choices=["mish", "relu"], help="激活函数")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="采样引导强度（0 关闭）")
    parser.add_argument(
        "--guidance-type", type=str, default="critic", choices=["critic"], help="引导类型，当前支持 critic 梯度"
    )
    return parser.parse_known_args()


def get_args():
    saved_argv = sys.argv
    fno_args, remaining = _parse_fno_args()
    # 去掉 FNO/引导 私有参数后复用原 parser
    sys.argv = [saved_argv[0]] + remaining
    args = base_get_args()
    sys.argv = saved_argv

    if args.log_prefix == "default":
        args.log_prefix = "diffusion_fno_guided"
    args.algorithm = "diffusion_fno_guided"

    # 默认开启专家模式 + BC（与 rectified_flow 脚本一致）
    if args.expert_type is None:
        args.expert_type = "mpc"
    args.bc_coef = True

    # 挂载 FNO / guidance 参数
    args.fno_modes = fno_args.fno_modes
    args.fno_width = fno_args.fno_width
    args.fno_layers = fno_args.fno_layers
    args.fno_activation = fno_args.fno_activation
    args.guidance_scale = fno_args.guidance_scale
    args.guidance_type = fno_args.guidance_type
    # 覆盖默认运行超参以加速实验
    args.diffusion_steps = 6
    args.training_num = 1
    args.test_num = 1
    args.step_per_epoch = 4096
    args.step_per_collect = 1024
    args.log_update_interval = 200
    return args


def build_guidance_fn(critic: DoubleCritic, device: torch.device) -> Callable:
    """
    使用 critic 的 Q 梯度做引导：沿着提升 Q 的方向调整动作。
    返回 grad 张量，与动作同形状。
    """

    def guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        # x_recon/state 已在采样设备上
        critic.eval()
        x_recon.requires_grad_(True)
        with torch.enable_grad():
            q1, q2 = critic(state, x_recon)
            q = torch.min(q1, q2).mean()
            grad = torch.autograd.grad(q, x_recon, retain_graph=False, create_graph=False)[0]
        # 取反是为了在 Diffusion 内的 x - scale * guidance 中实现“朝提升 Q 的方向走”
        return -grad.detach()

    return guidance


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
    print("  BEAR Building + DiffFNO + Guidance")
    print("=" * 60)
    pprint.pprint(vars(args))
    print()

    # 环境 ---------------------------------------------------------------
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

    # 网络 ---------------------------------------------------------------
    print("\nBuilding DiffFNO backbone ...")
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

    fno_backbone = DiffFNO(
        state_dim=state_dim,
        action_dim=action_dim,
        width=args.fno_width,
        modes=args.fno_modes,
        n_layers=args.fno_layers,
        t_dim=16,
        activation=args.fno_activation,
    ).to(args.device)

    actor_optim = torch.optim.Adam(
        fno_backbone.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd,
    )

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

    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=fno_backbone,
        max_action=max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        guidance_fn=None,  # 下面按需要注入
    ).to(args.device)

    # 配置 guidance（仅当 scale > 0）
    if args.guidance_scale > 0 and args.guidance_type == "critic":
        guidance_fn = build_guidance_fn(critic, args.device)
        diffusion_actor.set_guidance(guidance_fn, args.guidance_scale)
        print(f"Guidance enabled: type={args.guidance_type}, scale={args.guidance_scale}")
    else:
        print("Guidance disabled (scale <= 0)")

    print(f"  DiffFNO params: {sum(p.numel() for p in fno_backbone.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # 策略 ---------------------------------------------------------------
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
    print(
        f"  algorithm={args.algorithm}, diffusion_steps={args.diffusion_steps}, "
        f"fno_modes={args.fno_modes}, guidance_scale={args.guidance_scale}"
    )

    # 缓冲与采集器 -------------------------------------------------------
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

    # 训练 ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Start training (Diffusion + DiffFNO + Guidance)")
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
            os.path.join(log_path, "policy_best_fno_guided.pth"),
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
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final_fno_guided.pth"))
    print(f"Saved final model to: {log_path}")


if __name__ == "__main__":
    main()
