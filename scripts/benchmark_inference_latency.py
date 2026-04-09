#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark Guided-DiffFNO inference latency from a saved paper checkpoint.

This script loads:
- paper metadata for hyperparameters
- logged trajectory states for representative inputs
- the saved policy checkpoint

It then measures action-generation latency for batch size 1, with optional
guided and unguided passes, and reports both per-control-step and per-denoising-
step latency.
"""

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.model_fno import DiffFNO
from policy.diffusion_opt import DiffusionOPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark diffusion-policy inference latency.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=r"log_building\diffusion_fno_guided_bcfix_clean_OfficeSmall_Hot_Dry_20260408_145213",
        help="Run directory containing checkpoint and paper_data.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint override. Defaults to policy_best_fno_guided.pth inside log-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for benchmarking.",
    )
    parser.add_argument("--warmup", type=int, default=80, help="Warmup iterations before timing.")
    parser.add_argument("--num-samples", type=int, default=400, help="Measured iterations.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility of state ordering and diffusion noise.",
    )
    parser.add_argument(
        "--measure-unguided",
        action="store_true",
        help="Also benchmark the same checkpoint with guidance disabled for comparison.",
    )
    return parser.parse_args()


def load_meta_args(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if isinstance(meta, dict):
        return meta.get("args", {})
    return {}


def load_logged_states(trajectories_path: str) -> np.ndarray:
    data = np.load(trajectories_path)
    states = data["states"].astype(np.float32)
    lengths = data["lengths"].astype(np.int32)

    flat_states = []
    for ep in range(states.shape[0]):
        length = int(lengths[ep])
        if length > 0:
            flat_states.append(states[ep, :length])
    if not flat_states:
        raise ValueError(f"No valid states found in {trajectories_path}")
    return np.concatenate(flat_states, axis=0)


def build_policy(args_dict: Dict[str, Any], state_dim: int, action_dim: int, device: torch.device) -> DiffusionOPT:
    fno_backbone = DiffFNO(
        state_dim=state_dim,
        action_dim=action_dim,
        width=int(args_dict.get("fno_width", 48)),
        modes=int(args_dict.get("fno_modes", 4)),
        n_layers=int(args_dict.get("fno_layers", 1)),
        t_dim=16,
        activation=str(args_dict.get("fno_activation", "mish")),
    ).to(device)

    critic = DoubleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(args_dict.get("hidden_dim", 256)),
    ).to(device)

    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=fno_backbone,
        max_action=1.0,
        beta_schedule=str(args_dict.get("beta_schedule", "vp")),
        n_timesteps=int(args_dict.get("diffusion_steps", 6)),
        bc_coef=bool(args_dict.get("bc_coef", True)),
        guidance_scale=float(args_dict.get("guidance_scale", 0.0)),
        guidance_fn=None,
    ).to(device)

    actor_optim = torch.optim.Adam(fno_backbone.parameters(), lr=float(args_dict.get("actor_lr", 1e-4)))
    critic_optim = torch.optim.Adam(critic.parameters(), lr=float(args_dict.get("critic_lr", 1e-3)))

    policy = DiffusionOPT(
        state_dim=state_dim,
        actor=diffusion_actor,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=device,
        gamma=float(args_dict.get("gamma", 1.0)),
        reward_normalization=bool(args_dict.get("reward_normalization", False)),
        estimation_step=int(args_dict.get("n_step", 1)),
        bc_coef=bool(args_dict.get("bc_coef", True)),
        bc_weight=float(args_dict.get("bc_weight", 1.0)),
        bc_weight_final=args_dict.get("bc_weight_final", None),
        bc_weight_decay_steps=int(args_dict.get("bc_weight_decay_steps", 0)),
        exploration_noise=float(args_dict.get("exploration_noise", 0.1)),
        exploration_decay=False,
    ).to(device)
    return policy


def load_policy(policy: DiffusionOPT, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state, strict=True)


def build_guidance_fn(critic: DoubleCritic):
    def _guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor):
        critic.eval()
        x_recon.requires_grad_(True)
        with torch.enable_grad():
            q1, q2 = critic(state, x_recon)
            q = torch.min(q1, q2).mean()
            grad = torch.autograd.grad(q, x_recon, retain_graph=False, create_graph=False)[0]
        return -grad.detach()

    return _guidance


def set_guidance(policy: DiffusionOPT, enabled: bool, guidance_scale: float) -> None:
    actor = policy._actor
    if enabled and guidance_scale > 0:
        actor.set_guidance(build_guidance_fn(policy._critic), guidance_scale)
    else:
        actor.set_guidance(None, 0.0)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    policy: DiffusionOPT,
    states: np.ndarray,
    device: torch.device,
    warmup: int,
    num_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    policy.eval()
    policy._actor.eval()
    policy._critic.eval()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(states))
    if len(indices) < warmup + num_samples:
        repeats = int(np.ceil((warmup + num_samples) / len(indices)))
        indices = np.tile(indices, repeats)
    indices = indices[: warmup + num_samples]
    picked_states = states[indices]

    actions = []
    times_ms = []

    with torch.no_grad():
        for i in range(warmup + num_samples):
            obs = torch.as_tensor(picked_states[i], device=device, dtype=torch.float32).unsqueeze(0)
            torch.manual_seed(seed + i)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + i)
            synchronize_if_needed(device)
            start = time.perf_counter()
            action = policy._predict_action(obs, use_target=False)
            synchronize_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                actions.append(action.squeeze(0).detach().cpu().numpy())
                times_ms.append(elapsed_ms)

    return np.asarray(times_ms, dtype=np.float64), np.asarray(actions, dtype=np.float32)


def summarize(name: str, times_ms: np.ndarray, diffusion_steps: int) -> None:
    print(f"\n[{name}]")
    print(f"mean per control step: {times_ms.mean():.3f} ms")
    print(f"std  per control step: {times_ms.std(ddof=0):.3f} ms")
    print(f"p50  per control step: {np.percentile(times_ms, 50):.3f} ms")
    print(f"p95  per control step: {np.percentile(times_ms, 95):.3f} ms")
    print(f"mean per denoising step: {times_ms.mean() / max(diffusion_steps, 1):.3f} ms")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    log_dir = os.path.abspath(args.log_dir)
    checkpoint_path = (
        os.path.abspath(args.checkpoint)
        if args.checkpoint
        else os.path.join(log_dir, "policy_best_fno_guided.pth")
    )
    meta_path = os.path.join(log_dir, "paper_data", "paper_metadata.pkl")
    trajectories_path = os.path.join(log_dir, "paper_data", "trajectories.npz")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not os.path.isfile(trajectories_path):
        raise FileNotFoundError(f"Trajectory states not found: {trajectories_path}")

    device = torch.device(args.device)
    meta_args = load_meta_args(meta_path)
    states = load_logged_states(trajectories_path)
    state_dim = int(states.shape[-1])
    action_dim = int(np.load(trajectories_path)["actions"].shape[-1])
    diffusion_steps = int(meta_args.get("diffusion_steps", 6))
    guidance_scale = float(meta_args.get("guidance_scale", 0.0))

    policy = build_policy(meta_args, state_dim=state_dim, action_dim=action_dim, device=device)
    load_policy(policy, checkpoint_path)

    print(f"log_dir: {log_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"states used: {len(states)}")
    print(f"state_dim: {state_dim}, action_dim: {action_dim}, diffusion_steps: {diffusion_steps}")
    print(f"warmup: {args.warmup}, measured samples: {args.num_samples}")

    set_guidance(policy, enabled=True, guidance_scale=guidance_scale)
    guided_times, _ = benchmark(
        policy=policy,
        states=states,
        device=device,
        warmup=args.warmup,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    summarize("guided", guided_times, diffusion_steps)

    if args.measure_unguided:
        set_guidance(policy, enabled=False, guidance_scale=0.0)
        unguided_times, _ = benchmark(
            policy=policy,
            states=states,
            device=device,
            warmup=args.warmup,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        summarize("unguided", unguided_times, diffusion_steps)
        print(f"\nextra guided overhead: {guided_times.mean() - unguided_times.mean():.3f} ms per control step")


if __name__ == "__main__":
    main()
