#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
采集观测转移数据，用于训练舒适度温度代理模型（surrogate）。

合法性
------
只与环境的公开接口交互（reset/step），记录 (state_t, action_t, T_{t+1})，其中
T_{t+1} 取自 step() 返回的下一步观测 next_obs 的前 roomnum 维（即各区温度）——这正是
真实楼宇中下一时刻传感器测到的读数，是智能体本就能观测的量。全程不读取 BEAR 的
A_d/B_d 或任何仿真器内部状态，不开上帝视角。

采样策略
--------
代理模型将在"被引导扰动过的动作"上被查询，因此训练数据的动作必须覆盖较广的动作空间，
而不能只覆盖某个策略的窄分布。默认用多种动作分布混合采集：
  - uniform : 在 [-1, 1]^N 上均匀采样（主力，保证覆盖）
  - extreme : 每步整体偏向 ±1 或 0（覆盖满负荷/关机等边界）
可选叠加已训练策略的 on-policy 动作 + 探索噪声（--policy-checkpoint）。

输出
----
一个 .npz：states[N, state_dim], actions[N, action_dim], next_temp[N, roomnum]，
以及 roomnum/target_temp/tolerance 等元信息。不覆盖任何现有文件（默认写到 data/）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.building_env_wrapper import make_building_env  # noqa: E402
from env.building_config import (  # noqa: E402
    DEFAULT_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE,
    DEFAULT_MAX_POWER,
    DEFAULT_TIME_RESOLUTION,
)

def _sample_action(rng: np.random.Generator, action_dim: int, mode: str) -> np.ndarray:
    """按给定分布采一个动作，范围裁到 [-1, 1]。"""
    if mode == "extreme":
        # 每区独立在 {-1, 0, +1} 附近抖动，覆盖满负荷/关机边界
        base = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), size=action_dim)
        noise = rng.normal(0.0, 0.15, size=action_dim).astype(np.float32)
        return np.clip(base + noise, -1.0, 1.0).astype(np.float32)
    # uniform（默认）
    return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)


def _load_policy_action_fn(checkpoint: str, spec_kwargs: dict, device_str: str):
    """可选：加载已训练策略，返回 (obs)->action 的确定性动作函数。"""
    import torch
    from deploy_guidance.policy_io import PolicySpec, load_policy

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    spec = PolicySpec(**spec_kwargs)
    policy = load_policy(checkpoint, spec, device)

    def _action(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act = policy._predict_action(obs_t, use_target=False)
        return act.squeeze(0).cpu().numpy()

    return _action


def collect(args) -> None:
    rng = np.random.default_rng(args.seed)

    env, _, _ = make_building_env(
        building_type=args.building_type,
        weather_type=args.weather_type,
        location=args.location,
        target_temp=args.target_temp,
        temp_tolerance=args.tolerance,
        max_power=args.max_power,
        time_resolution=args.time_resolution,
        episode_length=args.episode_length,
        expert_type=None,          # 不需要专家，纯粹采转移
        training_num=1,
        test_num=1,
    )
    roomnum = env.roomnum
    state_dim = env.state_dim
    action_dim = env.action_dim
    print(f"env ready: state_dim={state_dim}, action_dim={action_dim}, rooms={roomnum}")

    policy_action_fn = None
    if args.policy_checkpoint:
        spec_kwargs = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            backbone_variant=args.backbone_variant,
            fno_width=args.fno_width,
            fno_modes=args.fno_modes,
            fno_layers=args.fno_layers,
            diffusion_steps=args.diffusion_steps,
        )
        policy_action_fn = _load_policy_action_fn(
            args.policy_checkpoint, spec_kwargs, args.device
        )
        print(f"on-policy actions enabled from: {args.policy_checkpoint}")

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_temps: List[np.ndarray] = []

    modes = [m.strip() for m in args.action_modes.split(",") if m.strip()]
    if not modes:
        modes = ["uniform"]

    collected = 0
    ep = 0
    while collected < args.num_transitions:
        obs, _ = env.reset(seed=int(args.seed + ep))
        done = False
        while not done and collected < args.num_transitions:
            mode = modes[rng.integers(0, len(modes))]
            if policy_action_fn is not None and rng.random() < args.policy_fraction:
                action = policy_action_fn(obs)
                action = np.clip(
                    action + rng.normal(0.0, args.policy_noise, size=action_dim),
                    -1.0, 1.0,
                ).astype(np.float32)
            else:
                action = _sample_action(rng, action_dim, mode)

            next_obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # 下一步各区温度 = 下一步观测的前 roomnum 维（智能体可观测的传感器读数），
            # 不从 BEAR 内部字段读取，避免上帝视角。
            zone_temp = np.asarray(next_obs, dtype=np.float32).reshape(-1)[:roomnum]

            states.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            next_temps.append(zone_temp)
            collected += 1
            obs = next_obs
        ep += 1
        if ep % 20 == 0:
            print(f"  episodes={ep}, transitions={collected}/{args.num_transitions}")

    states_arr = np.stack(states, axis=0)
    actions_arr = np.stack(actions, axis=0)
    next_temp_arr = np.stack(next_temps, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states_arr,
        actions=actions_arr,
        next_temp=next_temp_arr,
        roomnum=np.int64(roomnum),
        state_dim=np.int64(state_dim),
        action_dim=np.int64(action_dim),
        target_temp=np.float32(args.target_temp),
        tolerance=np.float32(args.tolerance),
    )
    print(f"saved {states_arr.shape[0]} transitions -> {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect observed transitions for the comfort surrogate.")
    p.add_argument("--num-transitions", type=int, default=40000)
    p.add_argument("--out", type=str, default=os.path.join(ROOT_DIR, "data", "surrogate_transitions.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-modes", type=str, default="uniform,extreme",
                   help="Comma-separated action sampling modes: uniform, extreme.")
    # 环境
    p.add_argument("--building-type", type=str, default="OfficeSmall")
    p.add_argument("--weather-type", type=str, default="Hot_Dry")
    p.add_argument("--location", type=str, default="Tucson")
    p.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TEMP_TOLERANCE)
    p.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER)
    p.add_argument("--time-resolution", type=int, default=DEFAULT_TIME_RESOLUTION)
    p.add_argument("--episode-length", type=int, default=168)
    # 可选 on-policy 采样
    p.add_argument("--policy-checkpoint", type=str, default=None)
    p.add_argument("--policy-fraction", type=float, default=0.3,
                   help="Probability of using on-policy action when a checkpoint is given.")
    p.add_argument("--policy-noise", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--backbone-variant", type=str, default="residual", choices=["residual", "nores"])
    p.add_argument("--fno-width", type=int, default=48)
    p.add_argument("--fno-modes", type=int, default=4)
    p.add_argument("--fno-layers", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())

