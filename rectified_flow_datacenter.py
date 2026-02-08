"""
数据中心场景的 Rectified Flow 入口（不改动原 main_datacenter）。

特性：
- 复用 main_datacenter 的参数和环境创建。
- 将 Actor 替换为 Rectified Flow（含时间缩放、ODE/Euler 采样、reflow 开关）。
- 日志前缀默认设为 rectified_flow，避免覆盖原运行。
"""

import os
import pprint
from datetime import datetime
import sys
import argparse

import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import main_datacenter as dc_main
from diffusion.model import DoubleCritic, MLP
from diffusion.rectified_flow import RectifiedFlow
from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from dropt_utils.paper_logging import run_paper_logging
from env.datacenter_env import make_datacenter_env
from policy import DiffusionOPT


def _parse_rf_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rf-time-scale", type=float, default=999.0, help="时间缩放（论文使用 999）")
    parser.add_argument("--rf-noise-scale", type=float, default=1.0, help="初始噪声尺度")
    parser.add_argument("--rf-sigma-var", type=float, default=0.0, help="sigma_t=(1-t)*sigma_var")
    parser.add_argument("--rf-sampler", type=str, default="euler", choices=["euler", "rk45"], help="采样器")
    parser.add_argument("--rf-sample-N", type=int, default=None, help="采样步数覆盖（None=使用 n_timesteps）")
    parser.add_argument(
        "--rf-reflow",
        action="store_true",
        default=False,
        help="启用 reflow/蒸馏（若使用，需要按论文准备教师数据）",
    )
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
        help="reflow 损失类型",
    )
    args, _ = parser.parse_known_args()
    return args, parser


def get_args():
    saved_argv = sys.argv
    rf_args, rf_parser = _parse_rf_args()
    rf_parsed, remaining = rf_parser.parse_known_args(saved_argv[1:])
    sys.argv = [saved_argv[0]] + remaining
    args = dc_main.get_args()
    sys.argv = saved_argv
    rf_args = rf_parsed
    # 附加 RF 配置
    args.rf_time_scale = rf_args.rf_time_scale
    args.rf_noise_scale = rf_args.rf_noise_scale
    args.rf_sigma_var = rf_args.rf_sigma_var
    args.rf_sampler = rf_args.rf_sampler
    args.rf_sample_N = rf_args.rf_sample_N
    args.rf_reflow = rf_args.rf_reflow
    args.rf_reflow_t_schedule = (
        rf_args.rf_reflow_t_schedule
        if not rf_args.rf_reflow_t_schedule.isdigit()
        else int(rf_args.rf_reflow_t_schedule)
    )
    args.rf_reflow_loss = rf_args.rf_reflow_loss
    # 区分日志前缀/算法名
    if args.log_prefix == "default":
        args.log_prefix = "rectified_flow"
    args.algorithm = "rectified_flow_opt_datacenter"
    return args


def main(args=None):
    if args is None:
        args = get_args()

    print("=" * 70)
    print("数据中心空调优化 - Rectified Flow")
    print("=" * 70)

    # 环境 ----------------------------------------------------------------
    env_kwargs = {
        "num_crac_units": args.num_crac,
        "target_temp": args.target_temp,
        "temp_tolerance": args.temp_tolerance,
        "episode_length": args.episode_length,
        "energy_weight": args.energy_weight,
        "temp_weight": args.temp_weight,
        "violation_penalty": args.violation_penalty,
        "expert_type": args.expert_type,
    }

    env, train_envs, test_envs = make_datacenter_env(
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
        **env_kwargs,
    )

    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.max_action = 1.0
    args.exploration_noise = args.exploration_noise * args.max_action

    print(f"  状态维度: {args.state_shape}")
    print(f"  动作维度: {args.action_shape}")
    print(f"  CRAC 数: {args.num_crac}")

    # 设备/随机种子 ------------------------------------------------------
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    # 日志 ---------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.log_prefix}_dc_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,
        verbose=True,
        diffusion_steps=args.n_timesteps,
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch,
        metrics_getter=None,
        png_interval=5,
    )

    # 指标聚合（保持与 main_datacenter 一致的可视化输出）
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
    velocity_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_dim,
        t_dim=16,
    ).to(args.device)

    actor = RectifiedFlow(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=velocity_net,
        max_action=args.max_action,
        n_timesteps=args.n_timesteps,
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
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    # 策略 ---------------------------------------------------------------
    policy = DiffusionOPT(
        state_dim=args.state_shape,
        actor=actor,
        actor_optim=actor_optim,
        action_dim=args.action_shape,
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
        reward_normalization=False,  # 保持与原始脚本一致
    )

    # 缓冲区/采集器 -------------------------------------------------------
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

    print(f"  缓冲区: {args.buffer_size:,} ({'PER' if args.prioritized_replay else 'Uniform'})")
    print(f"  采样器: {args.rf_sampler}, flow_steps={args.n_timesteps}, time_scale={args.rf_time_scale}")

    # 训练 ---------------------------------------------------------------
    last_paper_epoch = {"value": None}

    def _save_checkpoint(ep, env_step, grad_step):
        if ep % args.step_per_epoch == 0:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optim_actor": actor_optim.state_dict(),
                    "optim_critic": critic_optim.state_dict(),
                },
                os.path.join(log_path, f"checkpoint_{ep}.pth"),
            )
        if args.paper_log and args.paper_log_interval > 0 and ep % args.paper_log_interval == 0:
            try:
                print(f"\n[paper-log] Epoch {ep}: collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                last_paper_epoch["value"] = ep
            except Exception as exc:
                print(f"[paper-log] Failed at epoch {ep}: {exc}")
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
        save_best_fn=lambda p: torch.save(p.state_dict(), os.path.join(log_path, "policy_best_rf.pth")),
        save_checkpoint_fn=_save_checkpoint,
    )

    print("\n训练完成")
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final_rf.pth"))

    if args.paper_log:
        try:
            if args.paper_log_interval > 0 and last_paper_epoch["value"] == args.epoch:
                print("[paper-log] Skipped final logging (already captured at last epoch).")
            else:
                print("\n[paper-log] Collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")
    print(f"模型已保存至: {log_path}")


if __name__ == "__main__":
    main()
