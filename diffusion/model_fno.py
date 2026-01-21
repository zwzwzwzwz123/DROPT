"""
FNO 版本的扩散去噪网络（低频保留 + 残差旁路）。

设计原则：
- 在频域只保留前 `modes` 个低频，内置低通先验，契合热惯性/周期主导场景。
- 保留一条线性残差旁路，捕获突发/高频变化，避免响应迟滞。
- 接口与 diffusion.model.MLP 一致：forward(x, time, state) -> 去噪动作。
"""

import torch
import torch.nn as nn

from .helpers import SinusoidalPosEmb


class SpectralConv1d(nn.Module):
    """
    1D 频域卷积：FFT -> 低频乘权 -> IFFT。
    输入 [B, C_in, L]，输出 [B, C_out, L]。
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # 复权重拆成实部/虚部，便于优化器处理
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))

    def compl_mul1d(self, a, b):
        # a: [B, C_in, modes], b: [C_in, C_out, modes]
        return torch.einsum("bix, iox -> box", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, length = x.shape
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(
            b,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        weight = torch.view_as_complex(self.weight[:, :, :modes])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], weight)
        return torch.fft.irfft(out_ft, n=length)


class DiffFNO(nn.Module):
    """
    频域 FNO 去噪骨干，作为 diffusion.Diffusion 的 model 替换件。

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        width: 频域/通道宽度
        modes: 保留的低频数量（越大越保留高频）
        n_layers: 频域层数
        t_dim: 时间嵌入维度
        activation: 'mish' 或 'relu'
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

        # 频域块
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

        # 残差直通
        self.residual = nn.Linear(action_dim, action_dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

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

        res = self.residual(x)
        gate = torch.sigmoid(self.residual_gate)
        return out + gate * res
