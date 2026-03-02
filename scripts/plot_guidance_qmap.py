#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot a 2D Q-value contour with guided vs. unguided diffusion trajectories.

Default log dir is the provided run:
  log_building/diffusion_fno_guided_OfficeSmall_Hot_Dry_20260209_112739——100万步

Notes:
- Uses paper_data/guidance_trajectories.npz if present.
- action_dim > 2 is handled by choosing two dims to plot and fixing others to
  the last guided action in the diffusion trajectory.
"""

import argparse
import os
import pickle
import sys
from typing import Dict, Any, Tuple

import numpy as np
import torch

# Ensure project root is on sys.path when running from scripts/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.model_fno import DiffFNO
from policy.diffusion_opt import DiffusionOPT


DEFAULT_LOG_KEY = "diffusion_fno_guided_OfficeSmall_Hot_Dry_20260209_112739"
DEFAULT_LOG_ROOT = "log_building"


def find_log_dir(log_root: str, log_key: str) -> str:
    if os.path.isdir(log_key):
        return log_key
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"log_root not found: {log_root}")
    for name in os.listdir(log_root):
        if log_key in name:
            return os.path.join(log_root, name)
    raise FileNotFoundError(f"No log dir contains key: {log_key}")


def load_meta_args(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta.get("args", {}) if isinstance(meta, dict) else {}


def build_models(args: Dict[str, Any], state_dim: int, action_dim: int, device: torch.device) -> DiffusionOPT:
    fno_backbone = DiffFNO(
        state_dim=state_dim,
        action_dim=action_dim,
        width=int(args.get("fno_width", 48)),
        modes=int(args.get("fno_modes", 4)),
        n_layers=int(args.get("fno_layers", 1)),
        t_dim=16,
        activation=str(args.get("fno_activation", "mish")),
    ).to(device)

    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(args.get("hidden_dim", 256)),
    ).to(device)

    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=fno_backbone,
        max_action=1.0,
        beta_schedule=str(args.get("beta_schedule", "vp")),
        n_timesteps=int(args.get("diffusion_steps", 6)),
        bc_coef=bool(args.get("bc_coef", True)),
        guidance_scale=float(args.get("guidance_scale", 0.0)),
        guidance_fn=None,
    ).to(device)

    actor_optim = torch.optim.Adam(fno_backbone.parameters(), lr=1e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion_actor,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=device,
        gamma=float(args.get("gamma", 1.0)),
        reward_normalization=bool(args.get("reward_normalization", True)),
        estimation_step=int(args.get("n_step", 1)),
        bc_coef=bool(args.get("bc_coef", True)),
        bc_weight=float(args.get("bc_weight", 1.0)),
        bc_weight_final=args.get("bc_weight_final", None),
        bc_weight_decay_steps=int(args.get("bc_weight_decay_steps", 0)),
        exploration_noise=float(args.get("exploration_noise", 0.1)),
        exploration_decay=bool(args.get("exploration_decay", False)),
    ).to(device)

    return policy


def load_policy(policy: DiffusionOPT, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        policy.load_state_dict(state, strict=True)
    except Exception as exc:
        print(f"[warn] strict load failed: {exc}. Retrying with strict=False.")
        policy.load_state_dict(state, strict=False)


def compute_q_grid(
    critic: DoubleCritic,
    state: np.ndarray,
    base_action: np.ndarray,
    dim_x: int,
    dim_y: int,
    xs: np.ndarray,
    ys: np.ndarray,
    device: torch.device,
    batch: int = 4096,
) -> np.ndarray:
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    actions = np.repeat(base_action[None, :], grid.shape[0], axis=0)
    actions[:, dim_x] = grid[:, 0]
    actions[:, dim_y] = grid[:, 1]
    state_batch = np.repeat(state[None, :], grid.shape[0], axis=0)

    q_vals = np.empty((grid.shape[0],), dtype=np.float32)
    critic.eval()
    with torch.no_grad():
        for i in range(0, grid.shape[0], batch):
            sl = slice(i, min(i + batch, grid.shape[0]))
            s = torch.as_tensor(state_batch[sl], device=device, dtype=torch.float32)
            a = torch.as_tensor(actions[sl], device=device, dtype=torch.float32)
            q = critic.q_min(s, a).squeeze(-1)
            q_vals[sl] = q.detach().cpu().numpy()
    return q_vals.reshape(len(ys), len(xs))


def compute_q_along_traj(
    critic: DoubleCritic,
    state: np.ndarray,
    traj: np.ndarray,
    device: torch.device,
    batch: int = 1024,
) -> np.ndarray:
    critic.eval()
    q_vals = np.empty((traj.shape[0],), dtype=np.float32)
    state_batch = np.repeat(state[None, :], traj.shape[0], axis=0)
    with torch.no_grad():
        for i in range(0, traj.shape[0], batch):
            sl = slice(i, min(i + batch, traj.shape[0]))
            s = torch.as_tensor(state_batch[sl], device=device, dtype=torch.float32)
            a = torch.as_tensor(traj[sl], device=device, dtype=torch.float32)
            q = critic.q_min(s, a).squeeze(-1)
            q_vals[sl] = q.detach().cpu().numpy()
    return q_vals


def pick_plot_range(values: np.ndarray, margin: float = 0.1) -> Tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    span = vmax - vmin
    vmin -= margin * span
    vmax += margin * span
    vmin = max(-1.0, vmin)
    vmax = min(1.0, vmax)
    return vmin, vmax


def load_guidance_trajectories(guidance_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(guidance_path)
    state = data["state"].astype(np.float32)
    traj0 = data["diffusion_scale0"].astype(np.float32)
    trajN = data["diffusion_scaleN"].astype(np.float32)
    scale = float(data["guidance_scale"])
    return state, traj0, trajN, scale


def load_states_for_avg(trajectories_path: str, num_states: int, mode: str, seed: int) -> np.ndarray:
    data = np.load(trajectories_path)
    states = data["states"]  # [E, T, state_dim]
    lengths = data["lengths"]
    comfort = data.get("comfort_mean_abs_dev", None)
    violations = data.get("comfort_violations", None)

    valid_states = []
    scores = []
    for ep in range(states.shape[0]):
        length = int(lengths[ep]) if lengths is not None else states.shape[1]
        if length <= 0:
            continue
        ep_states = states[ep, :length]
        valid_states.append(ep_states)
        if comfort is not None and violations is not None:
            ep_scores = violations[ep, :length] + comfort[ep, :length]
            scores.append(ep_scores)
        else:
            scores.append(np.zeros((length,), dtype=np.float32))

    if not valid_states:
        raise ValueError("No valid states found in trajectories.")

    all_states = np.concatenate(valid_states, axis=0)
    all_scores = np.concatenate(scores, axis=0)

    num_states = min(num_states, all_states.shape[0])
    if num_states <= 0:
        raise ValueError("num_states must be > 0")

    if mode == "risk":
        idx = np.argsort(-all_scores)[:num_states]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(all_states.shape[0], size=num_states, replace=False)

    return all_states[idx]


def build_guidance_fn(critic: DoubleCritic) -> Any:
    def _guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor):
        critic.eval()
        x_recon.requires_grad_(True)
        q1, q2 = critic(state, x_recon)
        q = torch.min(q1, q2).mean()
        grad = torch.autograd.grad(q, x_recon, retain_graph=False, create_graph=False)[0]
        return -grad.detach()

    return _guidance


def sample_guidance_trajectories(
    actor: Diffusion,
    state: np.ndarray,
    guidance_scale: float,
    guidance_fn,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    state_tensor = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    prev_scale = getattr(actor, "guidance_scale", 0.0)
    prev_fn = getattr(actor, "guidance_fn", None)

    def _run(scale: float, fn) -> np.ndarray:
        if hasattr(actor, "set_guidance"):
            actor.set_guidance(fn, scale)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _, diffusion = actor.p_sample_loop(state_tensor, (1, actor.action_dim), return_diffusion=True)
        return diffusion.squeeze(0).detach().cpu().numpy()

    try:
        traj0 = _run(0.0, None)
        trajN = _run(guidance_scale, guidance_fn)
    finally:
        if hasattr(actor, "set_guidance"):
            actor.set_guidance(prev_fn, prev_scale)

    return traj0, trajN


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot guided diffusion trajectory over Q-value contour.")
    parser.add_argument("--log-root", type=str, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--log-key", type=str, default=DEFAULT_LOG_KEY)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dim-x", type=int, default=0)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument(
        "--pair-with",
        type=int,
        default=None,
        help="If set, plot this dim against all other dims (overrides dim-x/dim-y).",
    )
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override guidance scale (recompute trajectories).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path override (default: policy_final_fno_guided.pth if present, else policy_best_fno_guided.pth).",
    )
    parser.add_argument(
        "--rescale-q",
        action="store_true",
        help="Rescale Q values by 1/reward_scale for readability.",
    )
    parser.add_argument(
        "--avg-states",
        type=int,
        default=0,
        help="If >0, compute mean Q curve over N states with 95% CI.",
    )
    parser.add_argument(
        "--avg-mode",
        type=str,
        default="random",
        choices=["random", "risk"],
        help="State sampling mode for avg curve (random or risk).",
    )
    args = parser.parse_args()

    log_dir = find_log_dir(args.log_root, args.log_key)
    paper_dir = os.path.join(log_dir, "paper_data")
    fig_dir = os.path.join(log_dir, "paper_figures")
    os.makedirs(fig_dir, exist_ok=True)

    meta_path = os.path.join(paper_dir, "paper_metadata.pkl")
    guidance_path = os.path.join(paper_dir, "guidance_trajectories.npz")
    traj_path = os.path.join(paper_dir, "trajectories.npz")
    default_best = os.path.join(log_dir, "policy_best_fno_guided.pth")
    default_final = os.path.join(log_dir, "policy_final_fno_guided.pth")
    parser_ckpt = getattr(args, "ckpt", None)
    if parser_ckpt:
        ckpt_path = parser_ckpt
    else:
        # Prefer best checkpoint by default
        ckpt_path = default_best if os.path.isfile(default_best) else default_final

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"missing meta: {meta_path}")
    if not os.path.isfile(guidance_path):
        raise FileNotFoundError(f"missing guidance trajectories: {guidance_path}")
    if args.avg_states > 0 and not os.path.isfile(traj_path):
        raise FileNotFoundError(f"missing trajectories for avg curve: {traj_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    args_dict = load_meta_args(meta_path)
    state, traj0, trajN, scale = load_guidance_trajectories(guidance_path)

    action_dim = trajN.shape[1]
    state_dim = state.shape[0]
    if args.pair_with is not None:
        if args.pair_with < 0 or args.pair_with >= action_dim:
            raise ValueError(f"pair-with out of range: {args.pair_with} (action_dim={action_dim})")
        pairs = [(args.pair_with, j) for j in range(action_dim) if j != args.pair_with]
    else:
        if args.dim_x < 0 or args.dim_x >= action_dim:
            raise ValueError(f"dim-x out of range: {args.dim_x} (action_dim={action_dim})")
        if args.dim_y < 0 or args.dim_y >= action_dim:
            raise ValueError(f"dim-y out of range: {args.dim_y} (action_dim={action_dim})")
        if args.dim_x == args.dim_y:
            raise ValueError("dim-x and dim-y must be different")
        pairs = [(args.dim_x, args.dim_y)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = build_models(args_dict, state_dim, action_dim, device)
    load_policy(policy, ckpt_path)

    critic = policy._critic
    critic.eval()

    if args.guidance_scale is not None:
        seed_base = int(args_dict.get("seed", 0))
        seed_offset = int(args_dict.get("paper_guidance_seed", 0))
        guidance_seed = seed_base + seed_offset
        guidance_fn = build_guidance_fn(critic)
        traj0, trajN = sample_guidance_trajectories(
            policy._actor, state, float(args.guidance_scale), guidance_fn, guidance_seed, device
        )
        scale = float(args.guidance_scale)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib required for plotting: {exc}")

    # Ensure Chinese text renders correctly on Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    base_action = trajN[-1].copy()

    for dim_x, dim_y in pairs:
        proj0 = traj0[:, [dim_x, dim_y]]
        projN = trajN[:, [dim_x, dim_y]]
        xs_range = pick_plot_range(np.concatenate([proj0[:, 0], projN[:, 0]]))
        ys_range = pick_plot_range(np.concatenate([proj0[:, 1], projN[:, 1]]))

        xs = np.linspace(xs_range[0], xs_range[1], args.grid_size)
        ys = np.linspace(ys_range[0], ys_range[1], args.grid_size)
        q_grid = compute_q_grid(critic, state, base_action, dim_x, dim_y, xs, ys, device)

        X, Y = np.meshgrid(xs, ys)
        plt.figure(figsize=(8, 6))
        cs = plt.contourf(X, Y, q_grid, levels=30, cmap="RdYlGn")
        plt.contour(X, Y, q_grid, levels=12, colors="k", linewidths=0.4, alpha=0.35)
        plt.colorbar(cs, label="Q(s, a)")

        # Unguided trajectory
        plt.plot(proj0[:, 0], proj0[:, 1], "--", color="#666666", linewidth=2, label="Unguided")
        plt.scatter(proj0[0, 0], proj0[0, 1], color="#666666", s=40, zorder=3)

        # Guided trajectory
        plt.plot(projN[:, 0], projN[:, 1], "-", color="#1f77b4", linewidth=2.5, label=f"Guided (scale={scale:g})")
        plt.scatter(projN[0, 0], projN[0, 1], color="#1f77b4", s=40, zorder=3)

        # Direction arrows
        def _arrows(traj, color):
            dx = np.diff(traj[:, 0])
            dy = np.diff(traj[:, 1])
            plt.quiver(
                traj[:-1, 0],
                traj[:-1, 1],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.003,
                alpha=0.9,
            )

        _arrows(proj0, "#666666")
        _arrows(projN, "#1f77b4")

        plt.xlabel(f"Action dim {dim_x}")
        plt.ylabel(f"Action dim {dim_y}")
        plt.title("Guided vs. Unguided Diffusion on Q-Contour")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(fig_dir, f"guidance_qmap_dim{dim_x}{dim_y}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"saved: {out_path}")

    # Q-value along trajectory plot (single state)
    q0 = compute_q_along_traj(critic, state, traj0, device)
    qN = compute_q_along_traj(critic, state, trajN, device)

    # Optional rescale for readability (only valid when reward_normalization=False)
    q_scale = float(args_dict.get("reward_scale", 1.0) or 1.0)
    if args.rescale_q:
        if bool(args_dict.get("reward_normalization", False)):
            print("[warn] reward_normalization=True, rescale-q may be misleading.")
        if np.isclose(q_scale, 0.0):
            print("[warn] reward_scale is 0, skip rescale.")
        else:
            q0 = q0 / q_scale
            qN = qN / q_scale
    plt.figure(figsize=(8, 4))
    plt.plot(q0, "--", color="#666666", linewidth=2, label="Unguided Q")
    plt.plot(qN, "-", color="#1f77b4", linewidth=2.5, label=f"Guided Q (scale={scale:g})")
    plt.xlabel("Diffusion step")
    ylabel = "Q(s, a_t)"
    title = "Q Values Along Diffusion Trajectory"
    if args.rescale_q and not np.isclose(q_scale, 0.0):
        ylabel = f"Q(s, a_t) / {q_scale:g}"
        title = f"Q Values Along Diffusion Trajectory (rescaled by 1/{q_scale:g})"
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    qplot_path = os.path.join(fig_dir, "guidance_q_along_traj.png")
    plt.savefig(qplot_path, dpi=300)
    plt.close()
    print(f"saved: {qplot_path}")

    
    # Numeric differences (final action + Q)
    delta_action = float(np.linalg.norm(trajN[-1] - traj0[-1]))
    delta_q_final = float(qN[-1] - q0[-1])
    delta_q_mean = float(np.mean(qN) - np.mean(q0))
    print(f"[stats] action_dim={action_dim}")
    print(f"[stats] ||a_guided - a_unguided|| (final) = {delta_action:.6f}")
    print(f"[stats] DeltaQ_final = {delta_q_final:.6f}")
    print(f"[stats] DeltaQ_mean  = {delta_q_mean:.6f}")

    # Mean + CI across multiple states
    if args.avg_states > 0:
        seed_base = int(args_dict.get("seed", 0))
        seed_offset = int(args_dict.get("paper_guidance_seed", 0))
        guidance_seed = seed_base + seed_offset
        guidance_fn = build_guidance_fn(critic)

        states = load_states_for_avg(traj_path, args.avg_states, args.avg_mode, seed_base)
        q0_list = []
        qN_list = []
        for idx, s in enumerate(states):
            traj0_i, trajN_i = sample_guidance_trajectories(
                policy._actor, s, float(scale), guidance_fn, guidance_seed + idx, device
            )
            q0_i = compute_q_along_traj(critic, s, traj0_i, device)
            qN_i = compute_q_along_traj(critic, s, trajN_i, device)
            q0_list.append(q0_i)
            qN_list.append(qN_i)

        q0_arr = np.stack(q0_list, axis=0)
        qN_arr = np.stack(qN_list, axis=0)

        if args.rescale_q and not np.isclose(q_scale, 0.0):
            q0_arr = q0_arr / q_scale
            qN_arr = qN_arr / q_scale

        mean0 = q0_arr.mean(axis=0)
        meanN = qN_arr.mean(axis=0)
        std0 = q0_arr.std(axis=0, ddof=1) if q0_arr.shape[0] > 1 else np.zeros_like(mean0)
        stdN = qN_arr.std(axis=0, ddof=1) if qN_arr.shape[0] > 1 else np.zeros_like(meanN)
        n = q0_arr.shape[0]
        ci0 = 1.96 * std0 / np.sqrt(n) if n > 1 else np.zeros_like(mean0)
        ciN = 1.96 * stdN / np.sqrt(n) if n > 1 else np.zeros_like(meanN)

        plt.figure(figsize=(8, 4))
        steps = np.arange(mean0.shape[0])
        plt.plot(steps, mean0, "--", color="#666666", linewidth=2, label="Unguided Q (mean)")
        plt.fill_between(steps, mean0 - ci0, mean0 + ci0, color="#666666", alpha=0.2, linewidth=0)
        plt.plot(steps, meanN, "-", color="#1f77b4", linewidth=2.5, label=f"Guided Q (mean, scale={scale:g})")
        plt.fill_between(steps, meanN - ciN, meanN + ciN, color="#1f77b4", alpha=0.2, linewidth=0)
        plt.xlabel("Diffusion step")
        plt.ylabel(ylabel)
        plt.title(f"Mean Q Along Trajectory (N={n}, 95% CI)")
        plt.legend()
        plt.tight_layout()
        out_avg = os.path.join(fig_dir, "guidance_q_along_traj_mean_ci.png")
        plt.savefig(out_avg, dpi=300)
        plt.close()
        print(f"saved: {out_avg}")

        delta_q_final_mean = float(meanN[-1] - mean0[-1])
        delta_q_mean_mean = float(meanN.mean() - mean0.mean())
        print(f"[avg] N={n}, mode={args.avg_mode}")
        print(f"[avg] mean DeltaQ_final = {delta_q_final_mean:.6f}")
        print(f"[avg] mean DeltaQ_mean  = {delta_q_mean_mean:.6f}")


if __name__ == "__main__":

    main()
