# ========================================
# 神经网络模型定义
# ========================================
# 包含：
# 1. MLP：扩散模型的去噪网络
# 2. DoubleCritic：双Q网络

import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch


class MLP(nn.Module):
    """
    多层感知机（扩散模型的去噪网络）
    
    网络结构：
    - 状态编码器：处理环境状态
    - 时间编码器：处理扩散时间步（使用正弦位置编码）
    - 中间层：融合 [动作, 时间, 状态] 并预测去噪结果
    
    输入：
    - x: 带噪声的动作
    - time: 扩散时间步 t
    - state: 环境状态（条件信息）
    
    输出：
    - 去噪后的动作（或噪声预测，取决于训练模式）
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        """
        初始化MLP
        
        参数：
        - state_dim: 状态维度
        - action_dim: 动作维度
        - hidden_dim: 隐藏层维度（默认256）
        - t_dim: 时间编码维度（默认16）
        - activation: 激活函数（'mish'或'relu'）
        """
        super(MLP, self).__init__()
        # 选择激活函数（Mish通常在扩散模型中表现更好）
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        
        # ========== 状态编码器 ==========
        # 将环境状态编码为隐藏表示
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ========== 时间编码器 ==========
        # 使用正弦位置编码处理时间步信息
        # 正弦编码能更好地表达时间的连续性
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),  # 正弦位置编码
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
        # ========== 中间融合层 ==========
        # 输入：[噪声动作, 时间编码, 状态编码]
        # 输出：去噪后的动作
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, time, state):
        """
        前向传播

        参数：
        - x: 带噪声的动作
        - time: 时间步
        - state: 环境状态

        返回：
        - 去噪预测
        """
        # 确保所有输入在同一设备上
        device = x.device
        if time.device != device:
            time = time.to(device)
        if state.device != device:
            state = state.to(device)

        # 编码状态
        processed_state = self.state_mlp(state)
        # 编码时间
        t = self.time_mlp(time)
        # 拼接所有信息
        x = torch.cat([x, t, processed_state], dim=1)
        # 预测去噪结果
        x = self.mid_layer(x)
        return x




class DoubleCritic(nn.Module):
    """
    双Q网络（Critic）
    
    作用：
    - 评估(状态, 动作)对的价值
    - 使用两个独立的Q网络减少价值过估计
    - 在策略更新时取较小的Q值
    
    架构：
    - 状态编码器：共享的状态特征提取
    - Q1网络：第一个价值网络
    - Q2网络：第二个价值网络
    
    优势：
    - 减少DRL中的Q值过估计问题
    - 提高训练稳定性
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        """
        初始化双Q网络
        
        参数：
        - state_dim: 状态维度
        - action_dim: 动作维度
        - hidden_dim: 隐藏层维度
        - activation: 激活函数
        """
        super(DoubleCritic, self).__init__()
        # 使用ReLU激活函数（Critic网络通常用ReLU）
        _act = nn.ReLU
        
        # ========== 共享的状态编码器 ==========
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ========== Q1网络 ==========
        # 输入：[状态特征, 动作]
        # 输出：Q值（标量）
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Q2网络 ==========
        # 与Q1结构相同但参数独立
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        前向传播：计算两个Q值
        
        参数：
        - state: 环境状态
        - action: 动作
        
        返回：
        - (q1, q2): 两个Q网络的输出
        """
        # 编码状态
        processed_state = self.state_mlp(state)
        # 拼接状态和动作
        x = torch.cat([processed_state, action], dim=-1)
        # 分别通过两个Q网络
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, action):
        """
        返回较小的Q值
        
        用于Actor更新和目标Q计算
        取最小值可以减少过估计问题
        
        参数：
        - obs: 观察/状态
        - action: 动作
        
        返回：
        - min_q: 两个Q值中的较小者
        """
        return torch.min(*self.forward(obs, action))
