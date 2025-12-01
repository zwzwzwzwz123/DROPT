# ========================================
# 扩散优化策略（DiffusionOPT）
# ========================================
# 结合扩散模型和Actor-Critic框架
# 实现基于扩散模型的强化学习策略

import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from .helpers import (
    Losses
)


def check_numerical_stability(tensor: torch.Tensor, name: str, raise_error: bool = False) -> bool:
    """
    检查张量的数值稳定性

    参数:
    - tensor: 要检查的张量
    - name: 张量名称(用于错误信息)
    - raise_error: 是否抛出异常(False时仅警告)

    返回:
    - is_valid: 是否数值稳定
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        # 计算有效值的统计信息
        valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)
        if valid_mask.any():
            valid_tensor = tensor[valid_mask]
            error_msg = f"⚠️ [数值不稳定] {name}: "
            if has_nan:
                error_msg += f"包含NaN (数量: {torch.isnan(tensor).sum().item()}) "
            if has_inf:
                error_msg += f"包含Inf (数量: {torch.isinf(tensor).sum().item()}) "
            error_msg += f"| 有效值统计: min={valid_tensor.min().item():.4f}, "
            error_msg += f"max={valid_tensor.max().item():.4f}, "
            error_msg += f"mean={valid_tensor.mean().item():.4f}"
        else:
            error_msg = f"⚠️ [数值不稳定] {name}: 全部为NaN或Inf!"

        if raise_error:
            raise ValueError(error_msg)
        else:
            print(error_msg)
            return False

    return True


class DiffusionOPT(BasePolicy):
    """
    扩散优化策略类
    
    架构：
    - Actor: 扩散模型（生成动作）
    - Critic: 双Q网络（评估动作价值）
    
    训练模式：
    1. bc_coef=True（有专家数据）：
       - 使用行为克隆损失
       - 直接模仿专家动作（水注入算法）
       
    2. bc_coef=False（无专家数据）：
       - 使用策略梯度损失
       - 通过与环境交互学习
    
    核心组件：
    - Target Networks: 稳定训练的目标网络
    - Soft Update: 目标网络的指数移动平均更新
    - Exploration: 高斯噪声探索
    """

    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 1,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            bc_coef: bool = False,
            exploration_noise: float = 0.1,
            exploration_decay: bool = True,
            exploration_decay_steps: int = 100000,
            exploration_initial_rate: float = 0.3,
            exploration_final_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        """
        初始化扩散优化策略

        参数说明：
        - state_dim: 状态维度
        - actor: Actor网络（扩散模型）
        - actor_optim: Actor优化器
        - action_dim: 动作维度
        - critic: Critic网络（双Q网络）
        - critic_optim: Critic优化器
        - device: 计算设备
        - tau: 软更新系数（0.005表示目标网络每步更新0.5%）
        - gamma: 折扣因子（1表示不折扣）
        - reward_normalization: 是否标准化奖励
        - estimation_step: N步TD估计
        - lr_decay: 是否使用学习率衰减
        - lr_maxt: 学习率衰减的最大步数
        - bc_coef: 训练模式标志
        - exploration_noise: 探索噪声标准差
        - exploration_decay: 是否启用探索率衰减
        - exploration_decay_steps: 探索率衰减步数
        - exploration_initial_rate: 初始探索率
        - exploration_final_rate: 最终探索率
        """
        super().__init__(**kwargs)
        # 参数检查
        assert 0.0 <= tau <= 1.0, "tau应在[0, 1]范围内"
        assert 0.0 <= gamma <= 1.0, "gamma应在[0, 1]范围内"

        # ========== 初始化Actor网络 ==========
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor网络（扩散模型）
            self._target_actor = deepcopy(actor)  # 目标Actor网络（稳定训练）
            self._target_actor.eval()  # 目标网络设为评估模式
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Actor优化器
            self._action_dim = action_dim  # 动作空间维度

        # ========== 初始化Critic网络 ==========
        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic  # Critic网络（双Q网络）
            self._target_critic = deepcopy(critic)  # 目标Critic网络
            self._critic_optim: torch.optim.Optimizer = critic_optim  # Critic优化器
            self._target_critic.eval()  # 目标网络设为评估模式

        # ========== 学习率调度器 ==========
        if lr_decay:
            # 使用余弦退火调度器逐渐降低学习率
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        # ========== 其他参数 ==========
        self._tau = tau  # 目标网络软更新系数
        self._gamma = gamma  # 折扣因子
        self._rew_norm = reward_normalization  # 是否标准化奖励
        self._n_step = estimation_step  # N步TD估计
        self._lr_decay = lr_decay  # 是否使用学习率衰减
        self._bc_coef = bc_coef  # 训练模式标志
        self._device = device  # 计算设备
        self.noise_generator = GaussianNoise(sigma=exploration_noise)  # 噪声生成器

        # ========== 探索策略 ==========
        if exploration_decay and not bc_coef:
            self.exploration_scheduler = ExplorationScheduler(
                initial_rate=exploration_initial_rate,
                final_rate=exploration_final_rate,
                decay_steps=exploration_decay_steps,
                decay_type='exponential'
            )
            print(f"✓ 启用探索率衰减: {exploration_initial_rate} → {exploration_final_rate} (over {exploration_decay_steps} steps)")
        else:
            self.exploration_scheduler = None

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
        计算目标Q值
        
        用于TD学习的目标值计算：
        target_q = r + γ * Q_target(s', a')
        
        参数：
        - buffer: 经验回放缓冲区
        - indices: 采样的索引
        
        返回：
        - target_q: 目标Q值（使用双Q网络的最小值）
        """
        batch = buffer[indices]  # 获取batch（s_{t+n}）
        
        # 使用目标Actor生成下一状态的动作
        ttt = self(batch, model='_target_actor', input='obs_next').act
        
        # 使用目标Critic评估动作价值
        batch.obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        target_q = self._target_critic.q_min(batch.obs_next, ttt)
        
        # 返回双Q网络的最小值（减少过估计）
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """
        处理采样的batch数据
        
        计算N步TD目标：
        R_t = r_t + γr_{t+1} + ... + γ^n Q(s_{t+n}, a_{t+n})
        
        参数：
        - batch: 采样的数据批次
        - buffer: 经验回放缓冲区
        - indices: 采样索引
        
        返回：
        - batch: 处理后的batch（包含returns字段）
        """
        return self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm
        )

    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        更新策略网络

        标准的off-policy更新流程：
        1. 从经验回放缓冲区采样
        2. 计算N步回报
        3. 更新Critic和Actor
        4. 更新目标网络
        5. （可选）调整学习率和探索率

        参数：
        - sample_size: 批次大小
        - buffer: 经验回放缓冲区

        返回：
        - result: 损失等训练信息
        """
        # 检查缓冲区
        if buffer is None:
            return {}

        self.updating = True  # 标记正在更新

        # ========== 采样和处理数据 ==========
        batch, indices = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indices)

        # ========== 更新网络 ==========
        result = self.learn(batch, **kwargs)

        # ========== 学习率衰减 ==========
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()

        # ========== 更新探索率 ==========
        if self.exploration_scheduler is not None:
            self.exploration_scheduler.step()
            result['exploration_rate'] = self.exploration_scheduler.get_rate()

        self.updating = False  # 更新完成
        return result

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        """
        前向传播：生成动作
        
        流程：
        1. 将观察转为张量
        2. 通过Actor（扩散模型）生成动作
        3. （可选）添加探索噪声
        
        参数：
        - batch: 输入批次
        - state: 状态（用于RNN等，这里未使用）
        - input: 输入字段名（'obs'或'obs_next'）
        - model: 使用的模型（'actor'或'_target_actor'）
        
        返回：
        - Batch: 包含logits, act, state, dist的批次
        """
        # 转换为张量
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        
        # 选择模型（普通或目标）
        model_ = self._actor if model == "actor" else self._target_actor
        
        # 通过扩散模型生成动作
        logits, hidden = model_(obs_), None

        # ========== 探索策略 ==========
        if self._bc_coef:
            # 行为克隆模式：直接使用输出
            acts = logits
        else:
            # 策略梯度模式：根据探索率添加噪声
            if self.exploration_scheduler is not None:
                exploration_rate = self.exploration_scheduler.get_rate()
            else:
                exploration_rate = 0.1  # 固定探索率

            if np.random.rand() < exploration_rate:
                noise = to_torch(self.noise_generator.generate(logits.shape),
                                 dtype=torch.float32, device=self._device)
                acts = logits + noise
                acts = torch.clamp(acts, -1, 1)  # 裁剪到合理范围
            else:
                acts = logits

        # 不使用概率分布（确定性策略）
        dist = None

        return Batch(logits=logits, act=acts, state=obs_, dist=dist)

    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        # Convert the provided data to one-hot representation
        batch_size = data.shape[0]
        one_hot_codes = np.eye(one_hot_dim)
        # print(data[1])
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim))
                       for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    def _update_critic(self, batch: Batch) -> tuple:
        """
        更新Critic网络

        目标：最小化TD误差
        Loss = ||Q(s,a) - target_q||²

        双Q网络同时更新，取最小值用于Actor更新

        参数：
        - batch: 训练批次

        返回：
        - critic_loss: Critic损失
        - critic_grad_norm: Critic梯度范数
        - q_stats: Q值统计信息
        """
        # 转换为张量
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)

        # 目标Q值（N步回报）
        target_q = batch.returns

        # ========== 数值稳定性检测 ==========
        check_numerical_stability(obs_, "Critic输入状态", raise_error=False)
        check_numerical_stability(acts_, "Critic输入动作", raise_error=False)
        check_numerical_stability(target_q, "目标Q值", raise_error=False)

        # 当前Q值（双Q网络）
        current_q1, current_q2 = self._critic(obs_, acts_)

        # ========== 检测Q值 ==========
        check_numerical_stability(current_q1, "当前Q1值", raise_error=False)
        check_numerical_stability(current_q2, "当前Q2值", raise_error=False)

        # 计算MSE损失（两个Q网络的损失之和）
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # ========== 检测损失 ==========
        check_numerical_stability(critic_loss, "Critic损失", raise_error=True)

        # 优化Critic
        self._critic_optim.zero_grad()
        critic_loss.backward()

        # ========== 梯度裁剪 ==========
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic.parameters(),
            max_norm=10.0
        )

        # 检测梯度爆炸
        if critic_grad_norm > 10.0:
            print(f"⚠️ Critic梯度过大: {critic_grad_norm:.2f}, 已裁剪到10.0")

        self._critic_optim.step()

        # ========== 收集Q值统计信息 ==========
        with torch.no_grad():
            q_mean = (current_q1.mean() + current_q2.mean()) / 2
            q_std = (current_q1.std() + current_q2.std()) / 2
            q_min = torch.min(current_q1.min(), current_q2.min())
            q_max = torch.max(current_q1.max(), current_q2.max())
            td_error = torch.abs(current_q1 - target_q).mean()

            q_stats = {
                'q_mean': q_mean.item(),
                'q_std': q_std.item(),
                'q_min': q_min.item(),
                'q_max': q_max.item(),
                'td_error': td_error.item(),
                'target_q_mean': target_q.mean().item(),
                'target_q_std': target_q.std().item(),
            }

        return critic_loss, critic_grad_norm, q_stats


    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        """
        计算行为克隆损失（有专家数据模式）
        
        目标：让扩散模型学习生成专家动作
        Loss = ||π_θ(s) - a_expert||²
        
        参数：
        - batch: 训练批次
        - update: 是否立即更新（False时仅计算损失）
        
        返回：
        - bc_loss: 行为克隆损失
        """
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        
        # 提取专家动作（水注入算法的最优解）
        expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)

        # 计算扩散模型的损失
        bc_loss = self._actor.loss(expert_actions, obs_).mean()

        # 可选：立即更新
        if update:
            self._actor_optim.zero_grad()
            bc_loss.backward()
            self._actor_optim.step()
        
        return bc_loss

    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        """
        计算策略梯度损失（无专家数据模式）
        
        目标：最大化Q值
        Loss = -E[Q(s, π_θ(s))]
        
        参数：
        - batch: 训练批次
        - update: 是否立即更新
        
        返回：
        - pg_loss: 策略梯度损失
        """
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        
        # 生成动作
        acts_ = to_torch(self(batch).act, device=self._device, dtype=torch.float32)
        
        # 计算策略梯度损失（负Q值）
        pg_loss = - self._critic.q_min(obs_, acts_).mean()
        
        # 可选：立即更新
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        
        return pg_loss

    def _update_targets(self):
        """
        软更新目标网络
        
        公式：θ_target ← τ * θ + (1-τ) * θ_target
        
        作用：
        - 稳定训练（避免目标值剧烈变化）
        - 通常τ=0.005，即每步更新0.5%
        """
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        策略学习的核心函数

        完整的更新流程：
        1. 更新Critic（TD学习）
        2. 更新Actor（行为克隆或策略梯度）
        3. 软更新目标网络

        参数：
        - batch: 训练批次

        返回：
        - dict: 包含详细训练信息的字典
        """
        # ========== 步骤1：更新Critic ==========
        # 最小化TD误差
        critic_loss, critic_grad_norm, q_stats = self._update_critic(batch)

        # ========== 步骤2：更新Actor ==========
        if self._bc_coef:
            # 有专家数据：使用行为克隆
            bc_loss = self._update_bc(batch, update=False)
            overall_loss = bc_loss
        else:
            # 无专家数据：使用策略梯度
            pg_loss = self._update_policy(batch, update=False)
            overall_loss = pg_loss

        # ========== 检测Actor损失 ==========
        check_numerical_stability(overall_loss, "Actor损失", raise_error=True)

        # 执行Actor更新
        self._actor_optim.zero_grad()
        overall_loss.backward()

        # ========== Actor梯度裁剪 ==========
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(),
            max_norm=10.0
        )

        # 检测梯度爆炸
        if actor_grad_norm > 10.0:
            print(f"⚠️ Actor梯度过大: {actor_grad_norm:.2f}, 已裁剪到10.0")

        self._actor_optim.step()

        # ========== 步骤3：软更新目标网络 ==========
        self._update_targets()

        # ========== 收集动作统计信息 ==========
        with torch.no_grad():
            obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
            acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)

            action_stats = {
                'action_mean': acts_.mean().item(),
                'action_std': acts_.std().item(),
                'action_min': acts_.min().item(),
                'action_max': acts_.max().item(),
            }

        # ========== 返回详细训练信息 ==========
        result = {
            # 损失
            'loss/critic': critic_loss.item(),
            'loss/actor': overall_loss.item(),

            # 梯度范数
            'grad_norm/critic': critic_grad_norm.item(),
            'grad_norm/actor': actor_grad_norm.item(),
        }

        # 合并Q值统计
        result.update({f'q_value/{k}': v for k, v in q_stats.items()})

        # 合并动作统计
        result.update({f'action/{k}': v for k, v in action_stats.items()})

        # 警告：检测训练异常
        if critic_loss.item() > 1000:
            print(f"⚠️ 警告: Critic损失过高 ({critic_loss.item():.2f}), 训练可能不稳定!")

        if abs(q_stats['q_mean']) > 1000:
            print(f"⚠️ 警告: Q值过大 ({q_stats['q_mean']:.2f}), 可能存在数值问题!")

        return result


class ExplorationScheduler:
    """
    探索率调度器

    支持多种衰减策略:
    - linear: 线性衰减
    - exponential: 指数衰减
    - cosine: 余弦衰减
    """

    def __init__(
        self,
        initial_rate: float = 0.3,
        final_rate: float = 0.01,
        decay_steps: int = 100000,
        decay_type: str = 'exponential'
    ):
        """
        初始化探索调度器

        参数:
        - initial_rate: 初始探索率
        - final_rate: 最终探索率
        - decay_steps: 衰减步数
        - decay_type: 衰减类型 ('linear', 'exponential', 'cosine')
        """
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0

    def get_rate(self) -> float:
        """获取当前探索率"""
        if self.current_step >= self.decay_steps:
            return self.final_rate

        progress = self.current_step / self.decay_steps

        if self.decay_type == 'linear':
            rate = self.initial_rate - (self.initial_rate - self.final_rate) * progress
        elif self.decay_type == 'exponential':
            rate = self.final_rate + (self.initial_rate - self.final_rate) * (0.995 ** self.current_step)
        elif self.decay_type == 'cosine':
            rate = self.final_rate + 0.5 * (self.initial_rate - self.final_rate) * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"未知的衰减类型: {self.decay_type}")

        return max(rate, self.final_rate)

    def step(self):
        """更新步数"""
        self.current_step += 1

    def reset(self):
        """重置"""
        self.current_step = 0


class GaussianNoise:
    """
    高斯噪声生成器

    用于探索：在动作上添加随机噪声
    """

    def __init__(self, mu=0.0, sigma=0.1):
        """
        初始化噪声生成器

        参数：
        - mu: 均值（通常为0）
        - sigma: 标准差（控制探索程度）
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        生成高斯噪声

        参数：
        - shape: 噪声形状（通常与动作形状相同）

        返回：
        - noise: 高斯噪声数组
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise
