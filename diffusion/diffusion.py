# ========================================
# 扩散模型核心实现
# ========================================
# 实现了DDPM（Denoising Diffusion Probabilistic Model）算法
# 用于将随机噪声逐步去噪，生成最优动作

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入辅助函数和工具
from .helpers import (
    cosine_beta_schedule,  # 余弦噪声调度
    linear_beta_schedule,  # 线性噪声调度
    vp_beta_schedule,      # 方差保持噪声调度
    extract,               # 按时间步提取参数
    Losses                 # 损失函数字典
)
from .utils import Progress, Silent  # 进度显示工具


class Diffusion(nn.Module):
    """
    扩散模型主类
    
    核心思想：
    1. 前向过程（训练）：最优动作 → 逐步加噪 → 纯高斯噪声
    2. 反向过程（推理）：纯高斯噪声 → 逐步去噪 → 最优动作
    
    参数说明：
    - state_dim: 状态维度
    - action_dim: 动作维度
    - model: 去噪网络（MLP）
    - max_action: 动作最大值
    - beta_schedule: 噪声调度策略（'vp'/'linear'/'cosine'）
    - n_timesteps: 扩散步数（越大越精确但越慢）
    - loss_type: 损失类型（'l2'/'l1'）
    - clip_denoised: 是否裁剪去噪结果
    - bc_coef: 是否使用行为克隆（True=监督学习，False=强化学习）
    """
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='vp', n_timesteps=5,
                 loss_type='l2', clip_denoised=True, bc_coef=False):
        super(Diffusion, self).__init__()

        # ========== 基础属性 ==========
        self.state_dim = state_dim      # 状态维度
        self.action_dim = action_dim    # 动作维度
        self.max_action = max_action    # 动作最大值
        self.model = model              # 去噪网络

        # ========== 噪声调度策略 ==========
        # 选择不同的噪声调度方式，控制每一步加噪/去噪的幅度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)  # 线性增加
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)  # 余弦曲线
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)      # 方差保持（推荐）

        # ========== 计算扩散相关参数 ==========
        # alpha_t = 1 - beta_t（保留的原始信号比例）
        alphas = 1. - betas
        # alpha_bar_t = ∏(alpha_i) 从i=1到t（累积保留比例）
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # alpha_bar_{t-1}（前一步的累积保留比例）
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)    # 总步数
        self.clip_denoised = clip_denoised     # 是否裁剪
        self.bc_coef = bc_coef                 # 训练模式

        # ========== 注册为缓冲区（不参与梯度更新） ==========
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # ========== 预计算前向扩散过程的参数 ==========
        # 用于计算 q(x_t | x_0)：从原始动作直接到t步的噪声动作
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        
        # 用于从噪声预测原始动作
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # ========== 预计算后验分布的参数 ==========
        # 用于计算 q(x_{t-1} | x_t, x_0)：反向去噪的后验分布
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # 对数方差（裁剪避免log(0)）
        # 注意：在扩散链开始时后验方差为0
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # 后验均值的两个系数
        # μ_θ(x_t, t) = coef1 * x_0 + coef2 * x_t
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # ========== 选择损失函数 ==========
        self.loss_fn = Losses[loss_type]()

    # ==================== 采样（推理）相关方法 ====================
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        从噪声预测原始动作
        
        参数：
        - x_t: t时刻的噪声动作
        - t: 当前时间步
        - noise: 模型预测的噪声或动作
        
        返回：
        - x_0: 预测的原始动作（去噪结果）
        
        两种模式：
        1. bc_coef=True（行为克隆）：模型直接预测x_0
        2. bc_coef=False（策略梯度）：模型预测噪声ε，然后反推x_0
           公式：x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        """
        if self.bc_coef:
            # 监督学习模式：直接返回模型输出作为原始动作
            return noise
        else:
            # 强化学习模式：从噪声预测反推原始动作
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )


    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        
        这是真实的反向去噪分布（已知x_0的情况下）
        训练时用于计算目标分布
        
        公式：μ_posterior = coef1 * x_0 + coef2 * x_t
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        """
        计算模型预测的反向分布 p_θ(x_{t-1} | x_t) 的均值和方差
        
        步骤：
        1. 使用模型预测噪声（或直接预测x_0）
        2. 从噪声反推原始动作x_0
        3. 裁剪x_0到合理范围
        4. 计算后验分布参数
        
        参数：
        - x: 当前时刻的噪声动作
        - t: 当前时间步
        - s: 环境状态（条件信息）
        """
        # 通过模型预测并重建原始动作
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        # 裁剪到合理范围
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        # 计算后验分布参数
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        """
        执行一步反向采样：x_t → x_{t-1}
        
        从分布 p_θ(x_{t-1} | x_t) 中采样
        
        公式：x_{t-1} = μ_θ(x_t, t) + σ_t * z
        其中 z ~ N(0, I)
        
        参数：
        - x: 当前时刻的噪声动作 x_t
        - t: 当前时间步
        - s: 环境状态
        
        返回：
        - x_{t-1}: 去噪一步后的动作
        """
        b, *_, device = *x.shape, x.device
        # 获取预测的均值和方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        # 生成随机噪声（用于采样的随机性）
        noise = torch.randn_like(x)
        
        # 当t=0时不添加噪声（最后一步）
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        # x_{t-1} = μ + σ * noise
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        """
        完整的反向采样循环：纯噪声 → 最优动作
        
        从t=T到t=0逐步去噪
        
        过程：
        x_T ~ N(0, I)  →  x_{T-1}  →  ...  →  x_1  →  x_0（最优动作）
        
        参数：
        - state: 环境状态（条件信息）
        - shape: 动作形状 (batch_size, action_dim)
        - verbose: 是否显示进度
        - return_diffusion: 是否返回所有中间步骤
        
        返回：
        - x_0: 最终生成的动作
        """
        device = self.betas.device
        batch_size = shape[0]
        
        # 从标准正态分布采样初始噪声 x_T
        x = torch.randn(shape, device=device)

        # 可选：记录所有中间步骤（用于可视化）
        if return_diffusion: 
            diffusion = [x]

        # 进度显示器
        progress = Progress(self.n_timesteps) if verbose else Silent()
        
        # 反向去噪循环：从T到0
        for i in reversed(range(0, self.n_timesteps)):
            # 创建时间步张量（同一批次所有样本使用相同时间步）
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 执行一步去噪：x_t → x_{t-1}
            x = self.p_sample(x, timesteps, state)
            
            progress.update({'t': i})
            
            if return_diffusion: 
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def sample(self, state, *args, **kwargs):
        """
        生成动作的高层接口
        
        参数：
        - state: 环境状态
        
        返回：
        - action: 生成的动作（已裁剪到合理范围）
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        # 裁剪动作到[-max_action, max_action]
        return action.clamp_(-self.max_action, self.max_action)

    # ==================== 训练相关方法 ====================
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：x_0 → x_t（加噪）
        
        一步到位地从原始动作x_0生成t步后的噪声动作x_t
        不需要逐步迭代
        
        公式：x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        其中 ε ~ N(0, I)
        
        参数：
        - x_start: 原始动作 x_0
        - t: 目标时间步
        - noise: 可选的噪声（用于可复现）
        
        返回：
        - x_t: t时刻的噪声动作
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 应用前向扩散公式
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        """
        计算扩散模型的损失
        
        两种训练模式：
        1. bc_coef=True（行为克隆）：
           - 模型直接预测x_0
           - 损失：||model(x_t, t) - x_0||²
           
        2. bc_coef=False（策略梯度）：
           - 模型预测噪声ε
           - 损失：||model(x_t, t) - ε||²
        
        参数：
        - x_start: 原始动作（最优解或探索动作）
        - state: 环境状态
        - t: 随机采样的时间步
        - weights: 样本权重（用于优先经验回放）
        
        返回：
        - loss: 损失值
        """
        # 生成随机噪声
        noise = torch.randn_like(x_start)

        # 前向扩散：x_0 → x_t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 模型预测（噪声或原始动作）
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        # 根据训练模式选择损失目标
        if self.bc_coef:
            # 监督学习：预测原始动作
            loss = self.loss_fn(x_recon, x_start, weights)
        else:
            # 强化学习：预测噪声
            loss = self.loss_fn(x_recon, noise, weights)
        
        return loss

    def loss(self, x, state, weights=1.0):
        """
        计算批次损失（外部调用接口）
        
        为批次中每个样本随机采样不同的时间步
        这样可以同时学习所有时间步的去噪能力
        
        参数：
        - x: 原始动作批次
        - state: 状态批次
        - weights: 样本权重批次
        
        返回：
        - loss: 平均损失
        """
        batch_size = len(x)
        # 为每个样本随机采样时间步 t ~ Uniform(0, T-1)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        """
        前向传播：生成动作
        
        推理时调用，返回最终生成的动作
        """
        return self.sample(state, *args, **kwargs)