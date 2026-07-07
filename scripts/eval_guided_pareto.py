#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
部署期引导评估：加载已训练策略 + 舒适温度代理模型，扫描引导强度，
测出每个工作点的（能耗, 舒适违规），画出"训一次、部署期可调"的能耗-舒适前沿。

核心叙事
--------
策略只训一次（能耗-舒适权重固定）。部署时用舒适引导（可选叠加能耗引导）在采样阶段
把动作朝"更少违规 / 更省能"推。扫描引导强度 → 同一个模型滑出一整条前沿。这是训练
目标做不到的：训练时权重写死，引导让权衡在部署期在线可调。

合法性
------
- 舒适引导用的代理模型只从观测转移学习（系统辨识），不碰 BEAR 内部动力学。
- 能耗引导只用动作与设备额定功率（铭牌参数）。
- 评估用环境公开接口 reset/step 交互，指标（能耗/违规）来自 wrapper 的 info。

不影响现有代码：本脚本只读取 checkpoint、只新增，不修改任何训练脚本或核心模块。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

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
from deploy_guidance.policy_io import PolicySpec, load_policy  # noqa: E402
from deploy_guidance.surrogate import (  # noqa: E402
    TempSurrogate,
    build_comfort_guidance,
    build_energy_guidance,
    combine_guidance,
)


def _load_surrogate(path: str, device: torch.device) -> TempSurrogate:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = TempSurrogate(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        roomnum=cfg["roomnum"],
        hidden_dim=cfg["hidden_dim"],
        activation=cfg["activation"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[surrogate] loaded (val_RMSE={ckpt.get('val_rmse', float('nan')):.4f} °C) from {path}")
    return model


def _run_episode(env, policy, device: torch.device) -> Dict[str, float]:
    """跑一个 episode，用当前已配置的引导采样动作，返回该 episode 的能耗与违规。"""
    obs, _ = env.reset()
    done = False
    energy = 0.0
    violations = 0.0
    comfort_dev = 0.0
    steps = 0
    while not done:
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy._predict_action(obs_t, use_target=False)
        action = action.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        energy += float(info.get("hvac_energy_kwh", 0.0))
        violations += float(info.get("comfort_violations", 0.0))
        comfort_dev += float(info.get("comfort_mean_abs_dev", 0.0))
        steps += 1
    return {
        "energy": energy,
        "violations": violations / max(1, steps),
        "comfort_dev": comfort_dev / max(1, steps),
        "steps": float(steps),
    }


def _eval_operating_point(env, policy, actor, device, episodes: int) -> Dict[str, float]:
    """在当前引导配置下评估若干 episode，返回均值/标准差。"""
    energies: List[float] = []
    viols: List[float] = []
    devs: List[float] = []
    for _ in range(episodes):
        m = _run_episode(env, policy, device)
        energies.append(m["energy"])
        viols.append(m["violations"])
        devs.append(m["comfort_dev"])
    return {
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "violations_mean": float(np.mean(viols)),
        "violations_std": float(np.std(viols)),
        "comfort_dev_mean": float(np.mean(devs)),
    }


def evaluate(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env, _, _ = make_building_env(
        building_type=args.building_type,
        weather_type=args.weather_type,
        location=args.location,
        target_temp=args.target_temp,
        temp_tolerance=args.tolerance,
        max_power=args.max_power,
        time_resolution=args.time_resolution,
        episode_length=args.episode_length,
        expert_type=None,
        training_num=1,
        test_num=1,
    )
    state_dim = env.state_dim
    action_dim = env.action_dim
    roomnum = env.roomnum
    print(f"env: state_dim={state_dim}, action_dim={action_dim}, rooms={roomnum}")

    spec = PolicySpec(
        state_dim=state_dim,
        action_dim=action_dim,
        backbone_variant=args.backbone_variant,
        fno_width=args.fno_width,
        fno_modes=args.fno_modes,
        fno_layers=args.fno_layers,
        diffusion_steps=args.diffusion_steps,
    )
    policy = load_policy(args.policy_checkpoint, spec, device)
    actor = policy._actor

    surrogate = _load_surrogate(args.surrogate_checkpoint, device)

    # 构造引导项（舒适为主，能耗可选）
    comfort_fn = build_comfort_guidance(
        surrogate,
        target_temp=args.target_temp,
        tolerance=args.tolerance,
        softness=args.comfort_softness,
        penalty=args.comfort_penalty,
    )
    energy_fn = None
    if args.energy_weight > 0:
        energy_fn = build_energy_guidance(
            ac_map=env.ac_map,
            max_power=float(args.max_power),
            device=device,
        )

    scales = [float(s) for s in args.scales.split(",") if s.strip() != ""]
    print(f"sweeping guidance scales: {scales}")

    rows: List[Dict[str, float]] = []
    for scale in scales:
        if scale <= 0:
            actor.set_guidance(None, 0.0)   # 关闭引导 = 基策略基线
        else:
            guidance_fn = combine_guidance(
                [(1.0, comfort_fn), (args.energy_weight, energy_fn)]
            )
            actor.set_guidance(guidance_fn, scale)
        res = _eval_operating_point(env, policy, actor, device, args.episodes)
        res["scale"] = scale
        rows.append(res)
        print(
            f"  scale={scale:6.3f} | energy={res['energy_mean']:8.2f}±{res['energy_std']:.2f} kWh "
            f"| violations={res['violations_mean']:.4f}±{res['violations_std']:.4f} "
            f"| comfort_dev={res['comfort_dev_mean']:.4f} °C"
        )

    actor.set_guidance(None, 0.0)  # 复位，避免影响后续复用

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["guidance_scale", "energy_mean_kwh", "energy_std",
             "violations_mean", "violations_std", "comfort_dev_mean_c"]
        )
        for r in rows:
            writer.writerow([
                f"{r['scale']:.4f}", f"{r['energy_mean']:.4f}", f"{r['energy_std']:.4f}",
                f"{r['violations_mean']:.6f}", f"{r['violations_std']:.6f}",
                f"{r['comfort_dev_mean']:.6f}",
            ])
    print(f"saved Pareto sweep -> {args.out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deployment-time guidance Pareto sweep.")
    p.add_argument("--policy-checkpoint", type=str, required=True)
    p.add_argument("--surrogate-checkpoint", type=str,
                   default=os.path.join(ROOT_DIR, "data", "comfort_surrogate.pth"))
    p.add_argument("--out-csv", type=str,
                   default=os.path.join(ROOT_DIR, "data", "guided_pareto_sweep.csv"))
    p.add_argument("--scales", type=str, default="0,0.02,0.05,0.1,0.2,0.4,0.8",
                   help="Comma-separated guidance scales; 0 = no-guidance baseline.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    # 引导目标
    p.add_argument("--comfort-softness", type=float, default=0.25)
    p.add_argument("--comfort-penalty", type=str, default="softband", choices=["softband", "quadratic"])
    p.add_argument("--energy-weight", type=float, default=0.0,
                   help=">0 to add closed-form energy guidance on top of comfort.")
    # 策略结构（须与训练一致）
    p.add_argument("--backbone-variant", type=str, default="residual", choices=["residual", "nores"])
    p.add_argument("--fno-width", type=int, default=48)
    p.add_argument("--fno-modes", type=int, default=4)
    p.add_argument("--fno-layers", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=6)
    # 环境
    p.add_argument("--building-type", type=str, default="OfficeSmall")
    p.add_argument("--weather-type", type=str, default="Hot_Dry")
    p.add_argument("--location", type=str, default="Tucson")
    p.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TEMP_TOLERANCE)
    p.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER)
    p.add_argument("--time-resolution", type=int, default=DEFAULT_TIME_RESOLUTION)
    p.add_argument("--episode-length", type=int, default=168)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
