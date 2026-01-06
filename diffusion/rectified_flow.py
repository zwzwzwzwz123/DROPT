"""
Rectified Flow policy for连续控制，兼容 DiffusionOPT 接口。

特性对齐论文/官方代码：
- 速度场训练：预测恒定速度 (x0 - z0)，支持 reflow 蒸馏调度与 LPIPS 可选。
- 采样：Euler（默认）或 RK45 ODE（如安装 scipy），支持 sigma_var 噪声、可选 deterministic。
- 时间缩放：t*999 以匹配论文实现。
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .helpers import Losses


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: nn.Module,
        max_action: float,
        n_timesteps: int = 10,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        time_scale: float = 999.0,
        init_type: str = "gaussian",
        noise_scale: float = 1.0,
        use_ode_sampler: str = "euler",
        sigma_var: float = 0.0,
        ode_tol: float = 1e-5,
        sample_N: Optional[int] = None,
        reflow_flag: bool = False,
        reflow_t_schedule: str = "uniform",
        reflow_loss: str = "l2",
    ) -> None:
        """
        Args:
            state_dim: 状态维度。
            action_dim: 动作维度。
            model: 速度网络 f(x_t, t, s) -> v_t。
            max_action: 动作范围 [-max_action, max_action]。
            n_timesteps: Euler 步数（采样）；若 sample_N 提供则覆盖。
            loss_type: 训练损失 l1/l2。
            clip_denoised: 是否裁剪输出。
            time_scale: t 缩放（论文使用 999）。
            init_type/noise_scale: 初始噪声分布配置（当前支持 gaussian）。
            use_ode_sampler: "euler"（默认）或 "rk45"（需 scipy，fallback Euler）。
            sigma_var: 噪声方差调度系数 sigma_t=(1-t)*sigma_var。
            ode_tol: RK45 容差。
            sample_N: 采样步数覆盖（主要给 ODE Euler）。
            reflow_flag/reflow_t_schedule/reflow_loss: 蒸馏/再流训练配置。
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.max_action = float(max_action)
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.time_scale = float(time_scale)
        self.init_type = init_type
        self.noise_scale = float(noise_scale)
        self.use_ode_sampler = use_ode_sampler
        self.sigma_var = float(sigma_var)
        self.ode_tol = float(ode_tol)
        self.sample_N = sample_N
        self.reflow_flag = reflow_flag
        self.reflow_t_schedule = reflow_t_schedule
        self.reflow_loss = reflow_loss
        self.loss_fn = Losses[loss_type]()
        self._lpips_net = None  # lazy init

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prepare_weights(
        self, weights: Union[float, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if weights is None:
            weights = 1.0
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)
        if weights.dim() == 0:
            weights = weights.view(1, 1).expand(batch_size, 1)
        elif weights.dim() == 1:
            weights = weights.view(batch_size, 1)
        return weights

    def _get_z0(self, ref: torch.Tensor) -> torch.Tensor:
        if self.init_type == "gaussian":
            return torch.randn_like(ref) * self.noise_scale
        raise NotImplementedError(f"init_type {self.init_type} 未实现")

    def _time_sample(self, batch_size: int, device: torch.device, eps: float = 1e-3) -> torch.Tensor:
        if not self.reflow_flag:
            return torch.rand(batch_size, device=device) * (1.0 - eps) + eps

        if self.reflow_t_schedule == "t0":
            return torch.zeros(batch_size, device=device) * (1.0 - eps) + eps
        if self.reflow_t_schedule == "t1":
            return torch.ones(batch_size, device=device) * (1.0 - eps) + eps
        if self.reflow_t_schedule == "uniform":
            return torch.rand(batch_size, device=device) * (1.0 - eps) + eps
        if isinstance(self.reflow_t_schedule, int):
            k = max(1, self.reflow_t_schedule)
            return torch.randint(0, k, (batch_size,), device=device) * (1.0 - eps) / k + eps
        raise NotImplementedError(f"reflow_t_schedule {self.reflow_t_schedule} 未实现")

    def _maybe_init_lpips(self, device: torch.device):
        if self._lpips_net is not None:
            return self._lpips_net
        try:
            import lpips
        except ImportError:
            return None
        net = lpips.LPIPS(net="vgg").to(device)
        for p in net.parameters():
            p.requires_grad = False
        self._lpips_net = net
        return net

    # ------------------------------------------------------------------ #
    # Training: velocity matching / reflow distillation
    # ------------------------------------------------------------------ #
    def loss(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        state: torch.Tensor,
        weights: Optional[torch.Tensor] = 1.0,
    ):
        """
        Args:
            x: 目标动作 x0，或 (z0, x0) 二元组（reflow/蒸馏）。
            state: 条件状态。
            weights: PER 权重或标量。
        """
        if isinstance(x, (tuple, list)) and len(x) == 2:
            z0, target_x0 = x
        else:
            target_x0 = x
            z0 = None

        batch_size = target_x0.shape[0]
        device = target_x0.device

        t = self._time_sample(batch_size, device)
        if z0 is None:
            z0 = self._get_z0(target_x0)

        mix = t.view(batch_size, *([1] * (target_x0.dim() - 1)))
        x_t = mix * target_x0 + (1.0 - mix) * z0
        target_velocity = target_x0 - z0

        t_scaled = t * self.time_scale
        pred_velocity = self.model(x_t, t_scaled, state)

        weight_tensor = self._prepare_weights(weights, batch_size, device)

        if self.reflow_flag and "lpips" in self.reflow_loss:
            lpips_net = self._maybe_init_lpips(device)
            if lpips_net is None:
                # 回退 L2，避免缺失依赖导致崩溃
                lpips_loss = 0.0
            else:
                lpips_loss = lpips_net(z0 + pred_velocity, target_x0)
                lpips_loss = lpips_loss.view(batch_size, -1)
            l2_loss = self.loss_fn(pred_velocity, target_velocity, weight_tensor)
            if self.reflow_loss == "lpips":
                return lpips_loss.mean()
            if self.reflow_loss == "lpips+l2":
                lpips_mean = lpips_loss.mean() if torch.is_tensor(lpips_loss) else 0.0
                return lpips_mean + l2_loss
            raise NotImplementedError(f"reflow_loss {self.reflow_loss} 未实现")

        return self.loss_fn(pred_velocity, target_velocity, weight_tensor)

    # ------------------------------------------------------------------ #
    # Sampling: flow integration (Euler / RK45 fallback to Euler)
    # ------------------------------------------------------------------ #
    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        # sigma_t(t) = (1 - t) * sigma_var
        if self.sigma_var <= 0:
            return torch.zeros_like(t)
        return (1.0 - t) * self.sigma_var

    def _integrate_euler(
        self,
        state: torch.Tensor,
        steps: int,
        noise: Optional[torch.Tensor],
        deterministic: bool,
        return_trajectory: bool,
    ):
        device = state.device
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if noise is None:
            noise = torch.zeros(shape, device=device) if deterministic else torch.randn(shape, device=device)
        x = noise

        dt = 1.0 / float(steps)
        traj = [x] if return_trajectory else None

        for step in range(steps):
            t = torch.full((batch_size,), (step + 0.5) * dt, device=device)
            t_scaled = t * self.time_scale
            velocity = self.model(x, t_scaled, state)
            sigma = self._sigma_t(t).view(batch_size, *([1] * (x.dim() - 1)))
            noise_term = 0.0 if deterministic or self.sigma_var <= 0 else sigma * (dt ** 0.5) * torch.randn_like(x)
            x = x + dt * velocity + noise_term
            if return_trajectory:
                traj.append(x)

        if self.clip_denoised:
            x = torch.clamp(x, -self.max_action, self.max_action)
        if return_trajectory:
            return x, torch.stack(traj, dim=1)
        return x

    def _integrate_rk45(
        self,
        state: torch.Tensor,
        noise: Optional[torch.Tensor],
        deterministic: bool,
        return_trajectory: bool,
    ):
        """
        使用 scipy.integrate.solve_ivp 的概率流 ODE；若缺少 scipy 则回退 Euler。
        """
        try:
            from scipy import integrate
        except ImportError:
            return self._integrate_euler(
                state=state,
                steps=self.sample_N or self.n_timesteps,
                noise=noise,
                deterministic=deterministic,
                return_trajectory=return_trajectory,
            )

        device = state.device
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if noise is None:
            noise = torch.zeros(shape, device=device) if deterministic else torch.randn(shape, device=device)
        x0 = noise

        def ode_func(t_float: float, flat_x):
            x_tensor = torch.tensor(flat_x, device=device, dtype=torch.float32).reshape(shape)
            t_tensor = torch.full((batch_size,), t_float, device=device, dtype=torch.float32)
            t_scaled = t_tensor * self.time_scale
            with torch.no_grad():
                v = self.model(x_tensor, t_scaled, state).cpu().numpy().reshape(-1)
            return v

        sol = integrate.solve_ivp(
            ode_func,
            (0.0, 1.0),
            x0.cpu().numpy().reshape(-1),
            rtol=self.ode_tol,
            atol=self.ode_tol,
            method="RK45",
        )
        x = torch.tensor(sol.y[:, -1], device=device, dtype=torch.float32).reshape(shape)
        if self.clip_denoised:
            x = torch.clamp(x, -self.max_action, self.max_action)
        if return_trajectory:
            traj = torch.tensor(sol.y.T, device=device, dtype=torch.float32).reshape(-1, *shape)
            return x, traj
        return x

    def p_sample_loop(
        self,
        state: torch.Tensor,
        shape: torch.Size,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_trajectory: bool = False,
        sampler: Optional[str] = None,
    ):
        sampler = sampler or self.use_ode_sampler
        steps = self.sample_N or self.n_timesteps
        if sampler == "rk45":
            return self._integrate_rk45(state, noise, deterministic, return_trajectory)
        # 默认 Euler
        return self._integrate_euler(
            state=state,
            steps=steps,
            noise=noise,
            deterministic=deterministic,
            return_trajectory=return_trajectory,
        )

    def sample(
        self,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_trajectory: bool = False,
        sampler: Optional[str] = None,
    ):
        return self.p_sample_loop(
            state=state,
            shape=torch.Size((state.shape[0], self.action_dim)),
            noise=noise,
            deterministic=deterministic,
            return_trajectory=return_trajectory,
            sampler=sampler,
        )

    def forward(self, state: torch.Tensor, *args, **kwargs):
        # 兼容 DiffusionOPT：forward 即 sample
        return self.sample(state, *args, **kwargs)
