#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL 基线（SAC）入口文件。

目标：
- 作为扩散策略的对照基线，不修改原有训练脚本。
- 复用 `make_building_env`、Tianshou 收集器与增强日志。
- 训练结束后自动评测并打印能耗/舒适度/越界指标。
"""

import argparse
import os
import pprint
import inspect
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数（全部中文描述，便于对齐基线配置）。"""
    parser = argparse.ArgumentParser(
        description="SAC 基线（BEAR 建筑环境），用于与扩散策略对照。"
    )

    # 环境配置
    parser.add_argument("--building-type", type=str, default=DEFAULT_BUILDING_TYPE, help="建筑类型")
    parser.add_argument("--weather-type", type=str, default=DEFAULT_WEATHER_TYPE, help="天气类型")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION, help="地理位置")
    parser.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP, help="目标温度(℃)")
    parser.add_argument("--temp-tolerance", type=float, default=DEFAULT_TEMP_TOLERANCE, help="温度容差(℃)")
    parser.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER, help="HVAC 最大功率(W)")
    parser.add_argument("--time-resolution", type=int, default=DEFAULT_TIME_RESOLUTION, help="时间分辨率(秒)")
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH, help="每回合步数(默认一周)")
    parser.add_argument("--energy-weight", type=float, default=DEFAULT_ENERGY_WEIGHT, help="能耗权重 α")
    parser.add_argument("--temp-weight", type=float, default=DEFAULT_TEMP_WEIGHT, help="温度偏差权重 β")
    parser.add_argument("--violation-penalty", type=float, default=DEFAULT_VIOLATION_PENALTY, help="越界惩罚 γ")
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=DEFAULT_REWARD_SCALE,
        help="奖励缩放，保持与扩散策略一致以便对比。",
    )
    parser.add_argument(
        "--add-violation-penalty",
        action="store_true",
        default=True,
        help="启用温度越界惩罚（默认开）",
    )
    parser.add_argument(
        "--no-add-violation-penalty",
        dest="add_violation_penalty",
        action="store_false",
        help="关闭温度越界惩罚",
    )

    # 训练配置
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epoch", type=int, default=20000, help="训练轮数（对齐扩散策略）")
    parser.add_argument("--step-per-epoch", type=int, default=16384, help="每轮环境步数（对齐扩散策略）")
    parser.add_argument("--step-per-collect", type=int, default=4096, help="每次收集步数（对齐扩散策略）")
    parser.add_argument("--episode-per-test", type=int, default=DEFAULT_EPISODE_PER_TEST, help="评测回合数")
    parser.add_argument("--training-num", type=int, default=DEFAULT_TRAINING_NUM, help="并行训练环境数")
    parser.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM, help="并行测试环境数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="经验池大小")
    parser.add_argument("--update-per-step", type=float, default=0.5, help="每步更新次数（对齐扩散策略）")
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP, help="n-step TD 步数")

    # SAC 超参数
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="MLP 隐层尺寸")
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="Actor 学习率（对齐扩散策略）")
    parser.add_argument("--critic-lr", type=float, default=DEFAULT_CRITIC_LR, help="Critic 学习率（对齐扩散策略）")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="温度参数学习率")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    parser.add_argument(
        "--target-entropy-scale",
        type=float,
        default=1.0,
        help="目标熵系数：缺省为 -|A|，此参数乘在 |A| 上。",
    )
    parser.add_argument(
        "--reward-normalization",
        action="store_true",
        default=True,
        help="是否开启奖励归一化（对齐扩散策略，默认开启）。",
    )

    # 回放/日志/设备
    parser.add_argument("--prioritized-replay", action="store_true", default=False, help="是否启用 PER")
    parser.add_argument("--prior-alpha", type=float, default=0.6, help="PER alpha")
    parser.add_argument("--prior-beta", type=float, default=0.4, help="PER beta")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOG_DIR, help="日志根目录（与扩散策略一致）")
    parser.add_argument("--log-prefix", type=str, default="sac_baseline", help="日志前缀")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--vector-env-type", choices=["dummy", "subproc"], default="dummy", help="向量环境实现")
    parser.add_argument("--eval-episodes", type=int, default=3, help="训练结束评测回合数")
    parser.add_argument("--resume-path", type=str, default=None, help="恢复训练的模型路径")

    add_paper_logging_args(parser)
    return parser.parse_args()


def _aggregate_metrics(vector_env) -> Optional[Dict[str, float]]:
    """聚合向量环境中每个实例的 episode 级指标。"""
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
    """构建 TensorBoard + 终端日志器，输出环境指标。"""
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
    """构建 SAC 策略（含自动温度调节的 alpha）。"""
    state_shape = env.observation_space.shape or (env.state_dim,)
    action_shape = env.action_space.shape or (env.action_dim,)
    max_action = float(np.max(np.abs(env.action_space.high)))

    # Actor：带条件方差
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

    # Critic 双头
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

    # 自动熵权重
    target_entropy = -np.prod(action_shape) * args.target_entropy_scale
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    alpha = (target_entropy, log_alpha, alpha_optim)

    sac_params = inspect.signature(SACPolicy.__init__).parameters
    if "critic1" in sac_params:  # tianshou <= 0.x / early 1.x
        policy = SACPolicy(
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
        )
    else:  # tianshou >= 1.2
        policy_kwargs = dict(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=args.tau,
            gamma=args.gamma,
            alpha=alpha,
            estimation_step=args.n_step,
            action_space=env.action_space,
        )
        if "reward_normalization" in sac_params:
            policy_kwargs["reward_normalization"] = args.reward_normalization
        policy = SACPolicy(**policy_kwargs)
    return policy


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # tianshou 目前不支持 n_step>1 时启用 reward_normalization（会触发 AssertionError）
    if getattr(args, "reward_normalization", False) and getattr(args, "n_step", 1) > 1:
        print("Warning: n_step>1 与 reward_normalization 不兼容，已自动关闭 reward_normalization")
        args.reward_normalization = False

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建环境
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
        training_num=args.training_num,
        test_num=args.test_num,
        vector_env_type=args.vector_env_type,
    )
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    # 日志与目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.log_prefix}_{args.building_type}_{args.weather_type}_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    logger = make_logger(args, train_envs, test_envs, log_path)

    print("\n" + "=" * 70)
    print("  RL 基线 (SAC) - BEAR 建筑环境")
    print("=" * 70)
    pprint.pprint(vars(args))

    # 策略与优化器
    policy = build_sac_policy(args, env, device)
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=device)
        policy.load_state_dict(ckpt)
        print(f"已从 {args.resume_path} 恢复。")

    # 回放缓冲区
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

    # 数据收集器
    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs)

    # 训练
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

    print("\n训练完成。")
    pprint.pprint(result)

    # 保存最终模型
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
    print(f"已保存最终模型: {final_path}")

    # 评测
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
    print("\n评测摘要：")
    print(f"  回合数: {args.eval_episodes}")
    if eval_rews is not None:
        print(f"  平均奖励: {np.mean(eval_rews):.4f}")
    if eval_lens is not None:
        print(f"  平均长度: {np.mean(eval_lens):.1f}")
    if metrics:
        print("  环境指标 | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    else:
        print("  环境指标: 暂无（可继续运行更多 episode 获取）")


if __name__ == "__main__":
    main()
