"""
BEAR 建筑环境：扩散策略 + DiffFNO + Critic Guidance（安全/性能引导）。

在 main_building_fno 基础上：
- 增加 guidance_scale/类型参数，默认 0（不启用）。
- guidance_fn 默认使用 critic 的 Q 梯度作为引导（提升回报=减少违规）。
保持原训练逻辑，作为独立脚本，不影响现有流程。
"""

import os
import pickle
import pprint
import sys
import warnings
import math
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from main_building import get_args as base_get_args

from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from env.building_env_wrapper import make_building_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.helpers import SinusoidalPosEmb
from diffusion.model_fno import SpectralConv1d

warnings.filterwarnings("ignore")


class DiffFNO_NoResidual(nn.Module):
    """
    无残差版本的 DiffFNO：移除 residual 线性旁路与 gate。
    其余结构与 diffusion.model_fno.DiffFNO 一致，保持对比公平。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        width: int = 64,
        modes: int = 8,
        n_layers: int = 2,
        t_dim: int = 16,
        activation: str = "mish",
    ):
        super().__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU

        # 状态 + 时间编码
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, width),
            _act(),
            nn.Linear(width, width),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.cond_proj = nn.Linear(width + t_dim, width)

        # 输入: [B, 1, L] -> [B, width, L]
        self.input_proj = nn.Conv1d(1, width, kernel_size=1)

        # 频域层
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes=modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.activation = _act()

        # 输出映射回动作维度
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            _act(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        device = x.device
        if time.device != device:
            time = time.to(device)
        if state.device != device:
            state = state.to(device)

        state_feat = self.state_mlp(state)  # [B, width]
        t_feat = self.time_mlp(time)        # [B, t_dim]
        cond = self.cond_proj(torch.cat([state_feat, t_feat], dim=-1)).unsqueeze(-1)  # [B, width, 1]

        # [B, width, L]
        y = self.input_proj(x.unsqueeze(1))
        y = y + cond

        for spec_conv, pw_conv in zip(self.spectral_layers, self.pointwise):
            freq_out = spec_conv(y)
            point_out = pw_conv(y)
            y = self.activation(freq_out + point_out + cond)

        out = self.out_proj(y).squeeze(1)  # [B, L]
        return out


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
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--paper-log",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable paper data logging and plots.",
        )
    else:
        parser.add_argument(
            "--paper-log",
            action="store_true",
            default=True,
            help="Enable paper data logging and plots.",
        )
    parser.add_argument(
        "--paper-log-episodes",
        type=int,
        default=3,
        help="Episodes to record for paper data.",
    )
    parser.add_argument(
        "--paper-log-max-steps",
        type=int,
        default=0,
        help="Max steps per episode for paper logging (0=full episode).",
    )
    parser.add_argument(
        "--paper-guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale used for comparison plot (scale>0).",
    )
    parser.add_argument(
        "--paper-guidance-seed",
        type=int,
        default=0,
        help="Seed offset for guidance comparison sampling.",
    )
    parser.add_argument(
        "--paper-log-interval",
        type=int,
        default=50,
        help="Epoch interval to generate paper plots (0=disable during training).",
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
        args.log_prefix = "diffusion_fno_guided_nores"
    args.algorithm = "diffusion_fno_guided_nores"

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
    args.paper_log = fno_args.paper_log
    args.paper_log_episodes = fno_args.paper_log_episodes
    args.paper_log_max_steps = fno_args.paper_log_max_steps
    args.paper_guidance_scale = fno_args.paper_guidance_scale
    args.paper_guidance_seed = fno_args.paper_guidance_seed
    args.paper_log_interval = fno_args.paper_log_interval
    # 覆盖默认运行超参以加速实验
    args.diffusion_steps = 6
    args.training_num = 1
    args.test_num = 1
    args.step_per_epoch = 4096
    args.step_per_collect = 1024
    args.log_update_interval = 200
    argv = sys.argv[1:]
    has_epoch_flag = any(arg in ('--epoch', '-e') for arg in argv)
    has_total_steps_flag = '--total-steps' in argv
    if not has_epoch_flag and not has_total_steps_flag:
        args.total_steps = 1_000_000
    if args.total_steps is not None and args.total_steps > 0:
        args.epoch = max(1, math.ceil(args.total_steps / args.step_per_epoch))
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


@contextmanager
def _preserve_training_and_rng(policy: DiffusionOPT, actor: Diffusion):
    policy_was_training = policy.training
    actor_was_training = getattr(actor, "training", False)
    critic = getattr(policy, "_critic", None)
    critic_was_training = critic.training if critic is not None else False

    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    try:
        policy.eval()
        if actor is not None:
            actor.eval()
        if critic is not None:
            critic.eval()
        yield
    finally:
        policy.train(policy_was_training)
        if actor is not None:
            actor.train(actor_was_training)
        if critic is not None:
            critic.train(critic_was_training)
        torch.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)


def _policy_action(policy: DiffusionOPT, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = policy._predict_action(obs_tensor, use_target=False)
    return action.squeeze(0).cpu().numpy()


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _collect_eval_trajectories(
    env,
    policy: DiffusionOPT,
    episodes: int,
    max_steps: int,
    device: torch.device,
    gamma: float,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray]:
    episode_states: List[np.ndarray] = []
    episode_actions: List[np.ndarray] = []
    episode_rewards: List[np.ndarray] = []
    episode_info_metrics: List[Dict[str, np.ndarray]] = []
    episode_returns: List[np.ndarray] = []
    best_risk_state = None
    best_risk_key = (-1, -1.0)

    policy.eval()
    if hasattr(policy, "_actor"):
        policy._actor.eval()
    if hasattr(policy, "_critic"):
        policy._critic.eval()

    for ep in range(episodes):
        reset_seed = (seed + ep) if seed is not None else None
        obs, _ = env.reset(seed=reset_seed)
        done = False
        steps = 0
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        comfort_mean: List[float] = []
        comfort_viol: List[float] = []

        while not done:
            action = _policy_action(policy, obs, device)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            states.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            comfort_mean.append(float(info.get("comfort_mean_abs_dev", 0.0)))
            comfort_viol.append(float(info.get("comfort_violations", 0)))

            risk_key = (int(info.get("comfort_violations", 0)), float(info.get("comfort_mean_abs_dev", 0.0)))
            if risk_key > best_risk_key:
                best_risk_key = risk_key
                best_risk_state = obs.copy()

            obs = next_obs
            steps += 1
            if max_steps and steps >= max_steps:
                break

        states_arr = np.asarray(states, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.float32)
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        returns_arr = _discounted_returns(rewards_arr, gamma)
        episode_states.append(states_arr)
        episode_actions.append(actions_arr)
        episode_rewards.append(rewards_arr)
        episode_info_metrics.append(
            {
                "comfort_mean_abs_dev": np.asarray(comfort_mean, dtype=np.float32),
                "comfort_violations": np.asarray(comfort_viol, dtype=np.float32),
            }
        )
        episode_returns.append(returns_arr)

    max_len = max((len(ep) for ep in episode_actions), default=0)
    action_dim = episode_actions[0].shape[1] if episode_actions else 0
    state_dim = episode_states[0].shape[1] if episode_states else 0
    padded_actions = np.full((episodes, max_len, action_dim), np.nan, dtype=np.float32)
    padded_states = np.full((episodes, max_len, state_dim), np.nan, dtype=np.float32)
    padded_rewards = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_comfort = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_violations = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_returns = np.full((episodes, max_len), np.nan, dtype=np.float32)
    lengths = np.zeros((episodes,), dtype=np.int32)

    for idx in range(episodes):
        length = len(episode_actions[idx])
        lengths[idx] = length
        if length == 0:
            continue
        padded_actions[idx, :length] = episode_actions[idx]
        padded_states[idx, :length] = episode_states[idx]
        padded_rewards[idx, :length] = episode_rewards[idx]
        padded_comfort[idx, :length] = episode_info_metrics[idx]["comfort_mean_abs_dev"]
        padded_violations[idx, :length] = episode_info_metrics[idx]["comfort_violations"]
        padded_returns[idx, :length] = episode_returns[idx]

    metrics = {
        "lengths": lengths,
        "comfort_mean_abs_dev": padded_comfort,
        "comfort_violations": padded_violations,
    }
    data = {
        "states": padded_states,
        "actions": padded_actions,
        "rewards": padded_rewards,
        "returns": padded_returns,
        "lengths": lengths,
    }
    if best_risk_state is None:
        best_risk_state = padded_states[0, 0]
    return data, metrics, best_risk_state


def _sample_guidance_trajectories(
    actor: Diffusion,
    state: np.ndarray,
    guidance_scale: float,
    guidance_fn: Optional[Callable],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    device = actor.betas.device
    state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    prev_scale = actor.guidance_scale
    prev_fn = actor.guidance_fn

    def _run(scale: float, fn: Optional[Callable]) -> np.ndarray:
        actor.set_guidance(fn, scale)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _, diffusion = actor.p_sample_loop(
            state_tensor, (1, actor.action_dim), return_diffusion=True
        )
        return diffusion.squeeze(0).detach().cpu().numpy()

    try:
        traj_no_guidance = _run(0.0, None)
        traj_guidance = _run(guidance_scale, guidance_fn)
    finally:
        actor.set_guidance(prev_fn, prev_scale)

    return traj_no_guidance, traj_guidance


def _plot_action_series(actions: np.ndarray, lengths: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[paper-log] matplotlib unavailable, skip plots: {exc}")
        return

    if actions.size == 0:
        return
    min_len = int(np.min(lengths)) if lengths.size else actions.shape[1]
    if min_len == 0:
        return
    series = actions[0, :min_len]
    mean_series = np.nanmean(series, axis=1)
    window = max(3, min(15, min_len // 10))
    window = min(window, min_len)
    smooth = np.array([], dtype=np.float32)
    if window >= 2:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smooth = np.convolve(mean_series, kernel, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(mean_series, label="action_mean", color="#1f77b4", linewidth=1.5)
    if smooth.size > 0:
        plt.plot(np.arange(window - 1, window - 1 + smooth.size), smooth, label="moving_avg", color="#ff7f0e")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.title("Control Actions (Mean over dims)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_action_fft(actions: np.ndarray, lengths: np.ndarray, time_resolution: float, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[paper-log] matplotlib unavailable, skip FFT plot: {exc}")
        return

    if actions.size == 0:
        return
    min_len = int(np.min(lengths)) if lengths.size else actions.shape[1]
    if min_len < 2:
        return
    trimmed = actions[:, :min_len, :]
    mean_series = np.nanmean(trimmed, axis=(0, 2))
    fft_vals = np.fft.rfft(mean_series)
    freqs = np.fft.rfftfreq(min_len, d=time_resolution)
    amp = np.abs(fft_vals)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, amp, color="#2ca02c", linewidth=1.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Action FFT (Mean over episodes/dims)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_guidance_compare(diff0: np.ndarray, diff1: np.ndarray, scale: float, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[paper-log] matplotlib unavailable, skip guidance plot: {exc}")
        return

    if diff0.size == 0 or diff1.size == 0:
        return
    series0 = diff0.mean(axis=1)
    series1 = diff1.mean(axis=1)
    steps = np.arange(series0.shape[0])

    plt.figure(figsize=(8, 4))
    plt.plot(steps, series0, label="scale=0", color="#1f77b4")
    plt.plot(steps, series1, label=f"scale={scale:g}", color="#d62728")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Action (mean over dims)")
    plt.title("Guidance Trajectory Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_q_vs_return(q_values: np.ndarray, returns: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[paper-log] matplotlib unavailable, skip Q-vs-Return plot: {exc}")
        return

    if q_values.size == 0 or returns.size == 0:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(q_values, returns, s=8, alpha=0.5, color="#9467bd")
    min_val = min(np.min(q_values), np.min(returns))
    max_val = max(np.max(q_values), np.max(returns))
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="#7f7f7f", linewidth=1)
    plt.xlabel("Critic Q(s,a)")
    plt.ylabel("Monte Carlo Return")
    plt.title("Critic Q vs. Return")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _run_paper_logging_impl(
    env,
    policy: DiffusionOPT,
    actor: Diffusion,
    guidance_fn: Optional[Callable],
    args,
    log_path: str,
) -> None:
    paper_dir = os.path.join(log_path, "paper_data")
    fig_dir = os.path.join(log_path, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    episodes = max(1, int(args.paper_log_episodes))
    max_steps = int(args.paper_log_max_steps)
    data, metrics, risk_state = _collect_eval_trajectories(
        env=env,
        policy=policy,
        episodes=episodes,
        max_steps=max_steps,
        device=args.device,
        gamma=args.gamma,
        seed=args.seed,
    )

    np.savez_compressed(
        os.path.join(paper_dir, "trajectories.npz"),
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        returns=data["returns"],
        lengths=data["lengths"],
        comfort_mean_abs_dev=metrics["comfort_mean_abs_dev"],
        comfort_violations=metrics["comfort_violations"],
    )

    guidance_scale = float(args.paper_guidance_scale)
    seed = int(args.seed) + int(args.paper_guidance_seed)
    diff0, diff1 = _sample_guidance_trajectories(
        actor=actor,
        state=risk_state,
        guidance_scale=guidance_scale,
        guidance_fn=guidance_fn,
        seed=seed,
    )
    np.savez_compressed(
        os.path.join(paper_dir, "guidance_trajectories.npz"),
        state=risk_state,
        diffusion_scale0=diff0,
        diffusion_scaleN=diff1,
        guidance_scale=guidance_scale,
    )

    if data["actions"].size > 0:
        _plot_action_series(
            data["actions"],
            data["lengths"],
            os.path.join(fig_dir, "actions_timeseries.png"),
        )
        _plot_action_fft(
            data["actions"],
            data["lengths"],
            float(args.time_resolution),
            os.path.join(fig_dir, "actions_fft.png"),
        )

    if diff0.size > 0 and diff1.size > 0:
        _plot_guidance_compare(
            diff0,
            diff1,
            guidance_scale,
            os.path.join(fig_dir, "guidance_compare.png"),
        )

    traj = np.load(os.path.join(paper_dir, "trajectories.npz"))
    lengths = traj["lengths"]
    states = traj["states"]
    actions = traj["actions"]
    returns = traj["returns"]
    q_all = np.array([], dtype=np.float32)
    returns_all = np.array([], dtype=np.float32)
    for ep in range(states.shape[0]):
        length = int(lengths[ep])
        if length <= 0:
            continue
        ep_states = states[ep, :length]
        ep_actions = actions[ep, :length]
        ep_returns = returns[ep, :length]
        with torch.no_grad():
            s_tensor = torch.as_tensor(ep_states, device=args.device, dtype=torch.float32)
            a_tensor = torch.as_tensor(ep_actions, device=args.device, dtype=torch.float32)
            q1, q2 = policy._critic(s_tensor, a_tensor)
            q_min = torch.min(q1, q2).squeeze(-1).cpu().numpy()
        q_all = np.concatenate([q_all, q_min], axis=0)
        returns_all = np.concatenate([returns_all, ep_returns], axis=0)

    np.savez_compressed(
        os.path.join(paper_dir, "critic_q_vs_return.npz"),
        q_values=q_all,
        mc_returns=returns_all,
    )
    _plot_q_vs_return(
        q_all,
        returns_all,
        os.path.join(fig_dir, "critic_q_vs_return.png"),
    )

    with open(os.path.join(paper_dir, "paper_metadata.pkl"), "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "timestamp": datetime.now().isoformat(),
                "episodes": episodes,
                "max_steps": max_steps,
            },
            f,
        )


def run_paper_logging(
    env,
    policy: DiffusionOPT,
    actor: Diffusion,
    guidance_fn: Optional[Callable],
    args,
    log_path: str,
) -> None:
    with _preserve_training_and_rng(policy, actor):
        _run_paper_logging_impl(
            env=env,
            policy=policy,
            actor=actor,
            guidance_fn=guidance_fn,
            args=args,
            log_path=log_path,
        )


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
    print("  BEAR Building + DiffFNO(NoRes) + Guidance")
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
    print("\nBuilding DiffFNO(NoRes) backbone ...")
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1.0

    fno_backbone = DiffFNO_NoResidual(
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
    guidance_fn = build_guidance_fn(critic, args.device) if args.guidance_type == "critic" else None
    if args.guidance_scale > 0 and guidance_fn is not None:
        diffusion_actor.set_guidance(guidance_fn, args.guidance_scale)
        print(f"Guidance enabled: type={args.guidance_type}, scale={args.guidance_scale}")
    else:
        print("Guidance disabled (scale <= 0)")

    print(f"  DiffFNO(NoRes) params: {sum(p.numel() for p in fno_backbone.parameters()):,}")
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
    print("  Start training (Diffusion + DiffFNO(NoRes) + Guidance)")
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
                    guidance_fn=guidance_fn,
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
            os.path.join(log_path, "policy_best_fno_guided.pth"),
        ),
        save_checkpoint_fn=save_checkpoint_fn,
        train_fn=train_fn,
    )

    print("\nTraining finished")
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final_fno_guided.pth"))
    print(f"Saved final model to: {log_path}")

    if getattr(args, "paper_log", True):
        try:
            if args.paper_log_interval > 0 and last_paper_epoch["value"] == args.epoch:
                print("[paper-log] Skipped final logging (already captured at last epoch).")
            else:
                print("\n[paper-log] Collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=diffusion_actor,
                    guidance_fn=guidance_fn,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")


if __name__ == "__main__":
    main()
