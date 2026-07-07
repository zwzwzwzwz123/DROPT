"""
舒适度温度预测代理模型（surrogate）+ 部署期引导函数。

合法性说明
----------
本模型仅从"观测到的转移"中学习：输入 (state_t, action_t)，预测下一步各区温度
T_{t+1}。这里的下一步温度取自 t+1 时刻的观测状态前 roomnum 维（next_obs[:roomnum]），
即真实楼宇中传感器在下一时刻测到的区温读数——是标准系统辨识，不触碰 BEAR 的
A_d/B_d 或任何仿真器内部状态。

用途
----
1) TempSurrogate: 可微的 (state, action) -> 下一步区温 预测网络。
2) build_comfort_guidance: 用代理模型构造舒适引导，压低"预测越界"，直接对症
   会议版违规率偏高的问题。
3) build_energy_guidance: 闭式能耗引导（纯动作 + 设备额定功率），单向省能旋钮。
4) combine_guidance: 把多个引导项按权重合成一个 guidance_fn，注入扩散采样。

引导注入约定（与 diffusion.Diffusion.p_mean_variance 一致）：
    x_recon <- x_recon - guidance_scale * guidance_fn(x_recon, state, t)
因此 guidance_fn 需返回"代价对动作的梯度" ∂cost/∂a，采样即沿代价下降方向移动。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


class TempSurrogate(nn.Module):
    """
    区温预测小模型：预测 (T_{t+1} - T_t) 的归一化残差，再反归一化叠加到当前区温。

    输入：
    - state: [B, state_dim] 完整状态（区温在前 roomnum 维）
    - action: [B, action_dim] HVAC 动作，∈[-1, 1]

    输出：
    - pred_next_temp: [B, roomnum] 预测的下一步各区温度（°C）

    采用"预测温差"而非绝对温度，数值更稳，且天然利用了 T_{t+1}≈T_t 的先验。
    归一化统计量随模型一起保存/加载。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        roomnum: int,
        hidden_dim: int = 256,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.roomnum = int(roomnum)
        _act = nn.Mish if activation == "mish" else nn.ReLU

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, roomnum),
        )

        # 归一化 buffer（不参与梯度）。默认单位变换，训练脚本会用数据统计覆盖。
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("delta_mean", torch.zeros(roomnum))
        self.register_buffer("delta_std", torch.ones(roomnum))

    def set_normalization(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        delta_mean: np.ndarray,
        delta_std: np.ndarray,
        eps: float = 1e-6,
    ) -> None:
        """用数据集统计量设置归一化 buffer（std 加 eps 防除零）。"""
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(state_std, dtype=torch.float32).clamp_min(eps))
        self.action_mean.copy_(torch.as_tensor(action_mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(action_std, dtype=torch.float32).clamp_min(eps))
        self.delta_mean.copy_(torch.as_tensor(delta_mean, dtype=torch.float32))
        self.delta_std.copy_(torch.as_tensor(delta_std, dtype=torch.float32).clamp_min(eps))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一步区温（°C）。对 action 可微。"""
        s_norm = (state - self.state_mean) / self.state_std
        a_norm = (action - self.action_mean) / self.action_std
        delta_norm = self.net(torch.cat([s_norm, a_norm], dim=-1))
        delta = delta_norm * self.delta_std + self.delta_mean
        cur_temp = state[..., : self.roomnum]
        return cur_temp + delta

    def predict_delta_norm(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回归一化温差预测，供训练时直接与归一化标签算 MSE。"""
        s_norm = (state - self.state_mean) / self.state_std
        a_norm = (action - self.action_mean) / self.action_std
        return self.net(torch.cat([s_norm, a_norm], dim=-1))


# ============================================================
# 引导函数构造
# ============================================================
# 约定：guidance_fn(x_recon, state, t) -> grad，其中 grad = ∂cost/∂a。
# diffusion 内部执行 x_recon <- x_recon - scale * grad，即沿 cost 下降方向移动。


def build_comfort_guidance(
    surrogate: TempSurrogate,
    target_temp: float,
    tolerance: float,
    softness: float = 0.25,
    penalty: str = "softband",
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
    """
    舒适引导：用代理模型预测下一步区温，惩罚偏离舒适带，返回 ∂C/∂a。

    参数
    ----
    surrogate : 已训练的 TempSurrogate（推理时 eval，参数冻结）
    target_temp : 目标温度 T_ref（°C）
    tolerance : 舒适带半宽 δ（°C），|T - T_ref| <= δ 视为不违规
    softness : 软化尺度（°C），控制带边界附近的平滑过渡
    penalty : 'softband' 只惩罚越界部分（推荐）；'quadratic' 惩罚全部偏差

    代价定义
    --------
    softband: 每区 C_i = softplus((|T_i - T_ref| - δ) / softness) * softness，
              带内近似为 0、带外近似线性，梯度有界、边界可导。
    quadratic: C_i = ((T_i - T_ref) / softness)^2，处处把温度往目标拉。
    """
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    tgt = float(target_temp)
    delta = float(tolerance)
    soft = max(1e-3, float(softness))

    def guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        if state.shape[0] != x_recon.shape[0]:
            return None
        a = x_recon.detach().requires_grad_(True)
        with torch.enable_grad():
            # 采样中间步的 x_recon 可能落在 [-1,1] 之外，而 surrogate 只在合法动作域
            # 上训练过。先 clamp 再喂入，避免域外外推：域内 clamp 为恒等(梯度=1)，
            # 域外梯度=0(不产生虚假引导)，最终动作也会被扩散采样 clamp 回合法域。
            a_in = a.clamp(-1.0, 1.0)
            pred_temp = surrogate(state, a_in)  # [B, roomnum]
            dev = pred_temp - tgt
            if penalty == "quadratic":
                cost = ((dev / soft) ** 2).sum(dim=-1)
            else:  # softband
                excess = (dev.abs() - delta) / soft
                cost = torch.nn.functional.softplus(excess).sum(dim=-1) * soft
            grad = torch.autograd.grad(cost.sum(), a, retain_graph=False, create_graph=False)[0]
        return grad.detach()

    return guidance


def build_energy_guidance(
    ac_map: np.ndarray,
    max_power: float,
    device: torch.device,
    softness: float = 0.05,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
    """
    能耗引导：闭式能耗代价的动作梯度，单向把动作往省能（趋零）方向推。

    合法性：能耗 = Σ|a_i| * ac_map_i * max_power，只用动作与设备额定功率（铭牌参数），
    不碰仿真器内部。

    为避免 |a| 在 0 处不可导，用 sqrt(a^2 + softness^2) 平滑绝对值，其梯度为
    a / sqrt(a^2 + softness^2)，处处有界可导。功率常数按 ac_map 归一，使不同建筑
    的引导强度量纲一致。
    """
    ac = torch.as_tensor(np.asarray(ac_map, dtype=np.float32), device=device)
    denom = float(ac.sum().item()) * float(max_power)
    denom = denom if denom > 0 else 1.0
    soft = max(1e-4, float(softness))

    def guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        a = x_recon.detach()
        smooth_abs_grad = a / torch.sqrt(a * a + soft * soft)  # d|a|/da 的平滑版
        grad = smooth_abs_grad * ac.view(1, -1) * float(max_power) / denom
        return grad.detach()

    return guidance


def combine_guidance(
    terms,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]]]:
    """
    把多个 (weight, guidance_fn) 合成一个 guidance_fn，返回加权梯度和。

    terms: 可迭代的 (weight: float, fn: callable) 列表。weight<=0 或 fn=None 跳过。
    若无有效项返回 None（等价关闭引导）。
    """
    active = [(float(w), fn) for (w, fn) in terms if fn is not None and float(w) != 0.0]
    if not active:
        return None

    def guidance(x_recon: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        total = None
        for w, fn in active:
            g = fn(x_recon, state, t)
            if g is None:
                continue
            contrib = w * g
            total = contrib if total is None else total + contrib
        return total

    return guidance
