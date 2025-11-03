# ========================================
# 无线网络功率分配环境
# ========================================
# 实现了一个简化的功率分配问题
# 目标：在多个信道上分配功率以最大化总速率

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility  # 导入效用函数（奖励计算）
import numpy as np


class AIGCEnv(gym.Env):
    """
    AIGC网络优化环境
    
    问题描述：
    - 有多个正交信道（本例中为10个：5个好信道 + 5个差信道）
    - 需要在这些信道上分配功率
    - 目标：最大化总数据速率
    - 约束：总功率预算限制
    
    状态空间：信道增益 + 奖励（11维）
    动作空间：功率分配方案（10维连续动作）
    奖励：实际速率 - 最优速率（水注入算法）
    """

    def __init__(self):
        self._flag = 0
        
        # ========== 定义观察空间 ==========
        # 状态包括：信道增益 + 上一步奖励
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        
        # ========== 定义动作空间 ==========
        # 10个信道的功率分配（离散化为10个维度）
        self._action_space = Discrete(2*5)  # 10维动作空间
        
        # ========== 环境状态变量 ==========
        self._num_steps = 0              # 当前步数
        self._terminated = False         # 是否结束
        self._laststate = None           # 上一个状态
        self.last_expert_action = None  # 专家动作（用于行为克隆）
        
        # ========== 回合设置 ==========
        self._steps_per_episode = 1  # 每个回合1步（单步决策问题）

    @property
    def observation_space(self):
        """返回观察空间定义"""
        return self._observation_space

    @property
    def action_space(self):
        """返回动作空间定义"""
        return self._action_space

    @property
    def state(self):
        """
        生成新的状态（信道增益）
        
        信道模型：
        - 前5个信道：好信道，增益在[1, 2]之间（高质量）
        - 后5个信道：差信道，增益在[0, 1]之间（深衰落）
        
        返回：
        - states: 11维向量 [好信道增益×5, 差信道增益×5, 初始奖励]
        """
        # 生成好信道的增益（高质量）
        states1 = np.random.uniform(1, 2, 5)
        # 生成差信道的增益（深衰落）
        states2 = np.random.uniform(0, 1, 5)

        # 初始化奖励为0
        reward_in = []
        reward_in.append(0)
        
        # 合并为完整状态：[信道增益, 奖励]
        states = np.concatenate([states1, states2, reward_in])

        # 保存纯信道增益（用于计算奖励）
        self.channel_gains = np.concatenate([states1, states2])
        self._laststate = states
        return states


    def step(self, action):
        """
        执行一步环境交互
        
        参数：
        - action: 策略输出的动作（功率分配方案）
        
        返回：
        - next_state: 下一个状态
        - reward: 奖励（实际速率 - 最优速率）
        - terminated: 是否结束
        - info: 额外信息（包含专家动作）
        
        奖励计算：
        reward = 实际速率 - 水注入算法的最优速率
        目标是让reward接近0（即接近最优解）
        """
        # 检查回合是否已结束
        assert not self._terminated, "回合已结束，请调用reset()"
        
        # ========== 计算奖励和专家动作 ==========
        # CompUtility返回：奖励、专家动作、次优动作、实际动作
        reward, expert_action, sub_expert_action, real_action = CompUtility(self.channel_gains, action)

        # ========== 更新状态 ==========
        # 最后一维是奖励
        self._laststate[-1] = reward
        # 前10维是信道增益乘以实际功率分配
        self._laststate[0:-1] = self.channel_gains * real_action
        
        # ========== 更新步数 ==========
        self._num_steps += 1
        
        # ========== 检查是否结束 ==========
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        
        # ========== 返回信息 ==========
        # info包含专家动作，用于行为克隆训练
        info = {
            'num_steps': self._num_steps, 
            'expert_action': expert_action,           # 水注入算法的最优解
            'sub_expert_action': sub_expert_action    # 带噪声的次优解
        }
        return self._laststate, reward, self._terminated, info

    def reset(self):
        """
        重置环境到初始状态
        
        返回：
        - state: 新的初始状态
        - info: 重置信息
        """
        self._num_steps = 0        # 重置步数
        self._terminated = False   # 重置结束标志
        state = self.state         # 生成新的随机状态
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    """
    创建AIGC环境的工厂函数
    
    参数：
    - training_num: 并行训练环境数量
    - test_num: 并行测试环境数量
    
    返回：
    - env: 单个环境实例（用于单步测试）
    - train_envs: 训练用的向量化环境
    - test_envs: 测试用的向量化环境
    
    注意：
    向量化环境可以并行执行多个环境实例，提高数据收集效率
    """
    # 创建单个环境实例
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    
    # ========== 创建训练环境 ==========
    if training_num:
        # 创建多个环境副本用于并行训练
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    # ========== 创建测试环境 ==========
    if test_num:
        # 创建多个环境副本用于并行测试
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
        
    return env, train_envs, test_envs
