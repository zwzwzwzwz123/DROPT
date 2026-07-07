"""
复现并加载已训练的 DiffFNO 扩散策略（用于部署期引导评估）。

设计原则
--------
- 只读取 checkpoint，不改动任何训练脚本或核心模块。
- 网络结构与 main_building_fno_guided_bcfix_clean_ablation.py 中的构建保持一致：
  Diffusion(actor=DiffFNO / DiffFNO_NoResidual) + DoubleCritic。
- 兼容两种 checkpoint 格式：
    1) 裸 policy.state_dict()            （policy_best/final_fno_guided.pth）
    2) {"model": policy.state_dict(),..} （checkpoint_N.pth）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# 直接从子模块导入，绕开 diffusion/__init__.py（其在 331B 分支引用了已删除的
# rectified_flow，会导致 `from diffusion import Diffusion` 失败）。这样本模块不依赖
# 该 __init__ 是否修复，也完全不改动现有文件。
from diffusion.diffusion import Diffusion
from diffusion.model import DoubleCritic
from diffusion.model_fno import DiffFNO
from policy.diffusion_opt import DiffusionOPT


@dataclass
class PolicySpec:
    """复现策略网络所需的结构超参（须与训练时一致）。"""

    state_dim: int
    action_dim: int
    backbone_variant: str = "residual"   # residual | nores
    fno_width: int = 48
    fno_modes: int = 4
    fno_layers: int = 1
    fno_activation: str = "mish"
    t_dim: int = 16
    hidden_dim: int = 256
    diffusion_steps: int = 6
    beta_schedule: str = "vp"
    max_action: float = 1.0
    bc_coef: bool = True


def _build_backbone(spec: PolicySpec, device: torch.device):
    if spec.backbone_variant == "nores":
        # 延迟导入，避免对训练脚本产生模块级依赖
        from main_building_fno_guided_bcfix_clean_ablation import DiffFNO_NoResidual
        cls = DiffFNO_NoResidual
    else:
        cls = DiffFNO
    return cls(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        width=spec.fno_width,
        modes=spec.fno_modes,
        n_layers=spec.fno_layers,
        t_dim=spec.t_dim,
        activation=spec.fno_activation,
    ).to(device)


def build_policy(spec: PolicySpec, device: torch.device) -> DiffusionOPT:
    """按 spec 复现一个未加载权重的 DiffusionOPT（结构与训练时一致）。"""
    backbone = _build_backbone(spec, device)
    actor_optim = torch.optim.Adam(backbone.parameters(), lr=1e-4)

    critic = DoubleCritic(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        hidden_dim=spec.hidden_dim,
    ).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=2e-5)

    actor = Diffusion(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        model=backbone,
        max_action=spec.max_action,
        beta_schedule=spec.beta_schedule,
        n_timesteps=spec.diffusion_steps,
        bc_coef=spec.bc_coef,
        guidance_scale=0.0,
        guidance_fn=None,
    ).to(device)

    policy = DiffusionOPT(
        state_dim=spec.state_dim,
        actor=actor,
        actor_optim=actor_optim,
        action_dim=spec.action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=device,
        bc_coef=spec.bc_coef,
    )
    return policy


def _extract_state_dict(ckpt) -> dict:
    """从两种 checkpoint 格式中取出 policy.state_dict()。"""
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def load_policy(
    checkpoint_path: str,
    spec: PolicySpec,
    device: torch.device,
    strict: bool = False,
) -> DiffusionOPT:
    """
    复现结构并加载 checkpoint 权重，返回置于 eval 模式的策略。

    strict=False 容忍 target 网络等非必要键的缺失/多余，只要 actor/critic 主体
    能对上即可（加载后会打印 missing/unexpected 供核对）。
    """
    policy = build_policy(spec, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    result = policy.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        print(f"[load_policy] missing keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[load_policy] unexpected keys ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    policy.eval()
    if hasattr(policy, "_actor"):
        policy._actor.eval()
    if hasattr(policy, "_critic"):
        policy._critic.eval()
    return policy
