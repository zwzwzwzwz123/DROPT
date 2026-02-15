import os
import pickle
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tianshou.data import Batch


def add_paper_logging_args(parser, default_interval: int = 50) -> None:
    import argparse

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
        default=default_interval,
        help="Epoch interval to generate paper plots (0=disable during training).",
    )


@contextmanager
def preserve_training_and_rng(policy, actor=None):
    policy_was_training = getattr(policy, "training", False)
    actor_was_training = getattr(actor, "training", False) if actor is not None else False
    critic = getattr(policy, "_critic", None)
    critic_was_training = getattr(critic, "training", False) if critic is not None else False

    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    try:
        if hasattr(policy, "eval"):
            policy.eval()
        if actor is not None and hasattr(actor, "eval"):
            actor.eval()
        if critic is not None and hasattr(critic, "eval"):
            critic.eval()
        yield
    finally:
        if hasattr(policy, "train"):
            policy.train(policy_was_training)
        if actor is not None and hasattr(actor, "train"):
            actor.train(actor_was_training)
        if critic is not None and hasattr(critic, "train"):
            critic.train(critic_was_training)
        torch.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)


def _policy_action(policy, obs: np.ndarray, device: torch.device) -> np.ndarray:
    if hasattr(policy, "_predict_action"):
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy._predict_action(obs_tensor, use_target=False)
        return action.squeeze(0).cpu().numpy()

    batch = Batch(obs=obs[None, ...])
    try:
        batch.info = [{}]
    except Exception:
        pass
    with torch.no_grad():
        try:
            result = policy.forward(batch, deterministic=True)
        except TypeError:
            result = policy.forward(batch)
    act = result.act if hasattr(result, "act") else result["act"]
    if torch.is_tensor(act):
        act = act.cpu().numpy()
    return np.asarray(act).squeeze(0)


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _collect_eval_trajectories(
    env,
    policy,
    episodes: int,
    max_steps: int,
    device: torch.device,
    gamma: float,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], np.ndarray]:
    episode_states: List[np.ndarray] = []
    episode_actions: List[np.ndarray] = []
    episode_rewards: List[np.ndarray] = []
    episode_returns: List[np.ndarray] = []
    episode_info_metrics: List[Dict[str, np.ndarray]] = []
    best_risk_state = None
    best_risk_key: Tuple[Any, ...] = (float("-inf"),)

    if hasattr(policy, "eval"):
        policy.eval()

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
            comfort_mean.append(float(info.get("comfort_mean_abs_dev", np.nan)))
            comfort_viol.append(float(info.get("comfort_violations", np.nan)))

            if not np.isnan(comfort_viol[-1]) or not np.isnan(comfort_mean[-1]):
                risk_key = (comfort_viol[-1], comfort_mean[-1])
            else:
                risk_key = (-float(reward),)
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
        episode_returns.append(returns_arr)
        episode_info_metrics.append(
            {
                "comfort_mean_abs_dev": np.asarray(comfort_mean, dtype=np.float32),
                "comfort_violations": np.asarray(comfort_viol, dtype=np.float32),
            }
        )

    max_len = max((len(ep) for ep in episode_actions), default=0)
    action_dim = episode_actions[0].shape[1] if episode_actions else 0
    state_dim = episode_states[0].shape[1] if episode_states else 0
    padded_actions = np.full((episodes, max_len, action_dim), np.nan, dtype=np.float32)
    padded_states = np.full((episodes, max_len, state_dim), np.nan, dtype=np.float32)
    padded_rewards = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_returns = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_comfort = np.full((episodes, max_len), np.nan, dtype=np.float32)
    padded_violations = np.full((episodes, max_len), np.nan, dtype=np.float32)
    lengths = np.zeros((episodes,), dtype=np.int32)

    for idx in range(episodes):
        length = len(episode_actions[idx])
        lengths[idx] = length
        if length == 0:
            continue
        padded_actions[idx, :length] = episode_actions[idx]
        padded_states[idx, :length] = episode_states[idx]
        padded_rewards[idx, :length] = episode_rewards[idx]
        padded_returns[idx, :length] = episode_returns[idx]
        padded_comfort[idx, :length] = episode_info_metrics[idx]["comfort_mean_abs_dev"]
        padded_violations[idx, :length] = episode_info_metrics[idx]["comfort_violations"]

    if best_risk_state is None:
        best_risk_state = padded_states[0, 0] if max_len > 0 else np.zeros((state_dim,), dtype=np.float32)

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
    return data, metrics, best_risk_state


def _sample_guidance_trajectories(
    actor,
    state: np.ndarray,
    guidance_scale: float,
    guidance_fn: Optional[Callable],
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if actor is None or guidance_fn is None or guidance_scale <= 0:
        return None
    if not hasattr(actor, "p_sample_loop"):
        return None

    device = actor.betas.device if hasattr(actor, "betas") else torch.device("cpu")
    state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    prev_scale = getattr(actor, "guidance_scale", 0.0)
    prev_fn = getattr(actor, "guidance_fn", None)

    def _run(scale: float, fn: Optional[Callable]) -> np.ndarray:
        if hasattr(actor, "set_guidance"):
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
        if hasattr(actor, "set_guidance"):
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


def _compute_q_values(policy, states: np.ndarray, actions: np.ndarray, device: torch.device) -> Optional[np.ndarray]:
    if states.size == 0 or actions.size == 0:
        return None
    s_tensor = torch.as_tensor(states, device=device, dtype=torch.float32)
    a_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
    with torch.no_grad():
        if hasattr(policy, "_critic"):
            q1, q2 = policy._critic(s_tensor, a_tensor)
            q_min = torch.min(q1, q2).squeeze(-1)
            return q_min.cpu().numpy()
        if hasattr(policy, "critic1") and hasattr(policy, "critic2"):
            q1 = policy.critic1(s_tensor, a_tensor)
            q2 = policy.critic2(s_tensor, a_tensor)
            q_min = torch.min(q1, q2).squeeze(-1)
            return q_min.cpu().numpy()
        if hasattr(policy, "critic") and hasattr(policy, "critic2"):
            q1 = policy.critic(s_tensor, a_tensor)
            q2 = policy.critic2(s_tensor, a_tensor)
            q_min = torch.min(q1, q2).squeeze(-1)
            return q_min.cpu().numpy()
        if hasattr(policy, "critic"):
            q = policy.critic(s_tensor, a_tensor).squeeze(-1)
            return q.cpu().numpy()
    return None


def run_paper_logging(
    env,
    policy,
    args,
    log_path: str,
    actor=None,
    guidance_fn: Optional[Callable] = None,
) -> None:
    device = getattr(args, "device", torch.device("cpu"))
    if isinstance(device, str):
        if "cuda" in device and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(device)

    with preserve_training_and_rng(policy, actor):
        paper_dir = os.path.join(log_path, "paper_data")
        fig_dir = os.path.join(log_path, "paper_figures")
        os.makedirs(paper_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        episodes = max(1, int(getattr(args, "paper_log_episodes", 3)))
        max_steps = int(getattr(args, "paper_log_max_steps", 0))
        data, metrics, risk_state = _collect_eval_trajectories(
            env=env,
            policy=policy,
            episodes=episodes,
            max_steps=max_steps,
            device=device,
            gamma=float(getattr(args, "gamma", 1.0)),
            seed=getattr(args, "seed", None),
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

        time_resolution = float(getattr(args, "time_resolution", 1.0))
        if data["actions"].size > 0:
            _plot_action_series(
                data["actions"],
                data["lengths"],
                os.path.join(fig_dir, "actions_timeseries.png"),
            )
            _plot_action_fft(
                data["actions"],
                data["lengths"],
                time_resolution,
                os.path.join(fig_dir, "actions_fft.png"),
            )

        guidance_scale = float(getattr(args, "paper_guidance_scale", 0.0))
        seed = int(getattr(args, "seed", 0)) + int(getattr(args, "paper_guidance_seed", 0))
        guidance_traj = _sample_guidance_trajectories(
            actor=actor,
            state=risk_state,
            guidance_scale=guidance_scale,
            guidance_fn=guidance_fn,
            seed=seed,
        )
        if guidance_traj is not None:
            diff0, diff1 = guidance_traj
            np.savez_compressed(
                os.path.join(paper_dir, "guidance_trajectories.npz"),
                state=risk_state,
                diffusion_scale0=diff0,
                diffusion_scaleN=diff1,
                guidance_scale=guidance_scale,
            )
            _plot_guidance_compare(
                diff0,
                diff1,
                guidance_scale,
                os.path.join(fig_dir, "guidance_compare.png"),
            )

        lengths = data["lengths"]
        states = data["states"]
        actions = data["actions"]
        returns = data["returns"]
        q_all = np.array([], dtype=np.float32)
        returns_all = np.array([], dtype=np.float32)
        for ep in range(states.shape[0]):
            length = int(lengths[ep])
            if length <= 0:
                continue
            ep_states = states[ep, :length]
            ep_actions = actions[ep, :length]
            ep_returns = returns[ep, :length]
            q_vals = _compute_q_values(policy, ep_states, ep_actions, device)
            if q_vals is None:
                q_all = np.array([], dtype=np.float32)
                returns_all = np.array([], dtype=np.float32)
                break
            q_all = np.concatenate([q_all, q_vals], axis=0)
            returns_all = np.concatenate([returns_all, ep_returns], axis=0)

        if q_all.size > 0:
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
