#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练舒适度温度代理模型（TempSurrogate）。

输入：collect_transitions.py 产出的 .npz（states, actions, next_temp）。
目标：学习 (state_t, action_t) -> T_{t+1} 的可微映射，供部署期舒适引导使用。

合法性：只用观测转移做监督回归（系统辨识），不碰仿真器内部。

输出：一个 .pth，含 model state_dict、归一化统计量与结构超参，供 eval 脚本加载。
不覆盖任何现有训练产物（默认写到 data/）。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from deploy_guidance.surrogate import TempSurrogate  # noqa: E402


def _compute_stats(states, actions, deltas, eps=1e-6):
    return {
        "state_mean": states.mean(axis=0),
        "state_std": states.std(axis=0) + eps,
        "action_mean": actions.mean(axis=0),
        "action_std": actions.std(axis=0) + eps,
        "delta_mean": deltas.mean(axis=0),
        "delta_std": deltas.std(axis=0) + eps,
    }


def train(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = np.load(args.data)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    next_temp = data["next_temp"].astype(np.float32)
    roomnum = int(data["roomnum"])
    state_dim = int(data["state_dim"])
    action_dim = int(data["action_dim"])

    # 监督目标：温差 delta = T_{t+1} - T_t（T_t 为状态前 roomnum 维）
    cur_temp = states[:, :roomnum]
    deltas = next_temp - cur_temp
    print(f"data: N={states.shape[0]}, state_dim={state_dim}, action_dim={action_dim}, rooms={roomnum}")
    print(f"delta stats: mean={deltas.mean():.4f}, std={deltas.std():.4f}, "
          f"|delta|max={np.abs(deltas).max():.4f}")

    # 训练/验证划分
    n = states.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    stats = _compute_stats(states[train_idx], actions[train_idx], deltas[train_idx])

    model = TempSurrogate(
        state_dim=state_dim,
        action_dim=action_dim,
        roomnum=roomnum,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
    ).to(device)
    model.set_normalization(**stats)

    # 归一化后的监督标签（与 predict_delta_norm 对齐）
    delta_mean_t = torch.as_tensor(stats["delta_mean"], device=device)
    delta_std_t = torch.as_tensor(stats["delta_std"], device=device)

    states_t = torch.as_tensor(states, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    deltas_t = torch.as_tensor(deltas, device=device)
    target_norm = (deltas_t - delta_mean_t) / delta_std_t

    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bs = args.batch_size
    for epoch in range(args.epochs):
        model.train()
        epoch_perm = train_idx_t[torch.randperm(train_idx_t.numel(), device=device)]
        total = 0.0
        nb = 0
        for i in range(0, epoch_perm.numel(), bs):
            idx = epoch_perm[i:i + bs]
            pred = model.predict_delta_norm(states_t[idx], actions_t[idx])
            loss = loss_fn(pred, target_norm[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            nb += 1
        train_loss = total / max(1, nb)

        # 验证：报告物理量纲（°C）的 RMSE，直观可判读
        model.eval()
        with torch.no_grad():
            pred_temp = model(states_t[val_idx_t], actions_t[val_idx_t])
            true_temp = (states_t[val_idx_t][:, :roomnum] + deltas_t[val_idx_t])
            rmse = torch.sqrt(((pred_temp - true_temp) ** 2).mean()).item()
        if rmse < best_val:
            best_val = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} | train_loss(norm)={train_loss:.5f} | val_RMSE={rmse:.4f} °C "
                  f"| best={best_val:.4f} °C")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "roomnum": roomnum,
                "hidden_dim": args.hidden_dim,
                "activation": args.activation,
            },
            "val_rmse": best_val,
            "target_temp": float(data["target_temp"]),
            "tolerance": float(data["tolerance"]),
        },
        args.out,
    )
    print(f"saved surrogate (val_RMSE={best_val:.4f} °C) -> {args.out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the comfort temperature surrogate.")
    p.add_argument("--data", type=str, default=os.path.join(ROOT_DIR, "data", "surrogate_transitions.npz"))
    p.add_argument("--out", type=str, default=os.path.join(ROOT_DIR, "data", "comfort_surrogate.pth"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--activation", type=str, default="mish", choices=["mish", "relu"])
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
