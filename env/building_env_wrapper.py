# ========================================
# BEAR 建筑环境适配器
# ========================================
# 将 BEAR 的 BuildingEnvReal 包装成符合 DROPT 接口的环境
# 不修改 BEAR 原始代码，通过适配器层实现集成

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union

# 添加 BEAR 到 Python 路径
bear_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bear')
if bear_path not in sys.path:
    sys.path.insert(0, bear_path)

# 导入 BEAR 模块
from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator

# 导入配置常量（修复：统一管理配置参数）
from env.building_config import (
    DEFAULT_REWARD_SCALE,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_TEMP_WEIGHT,
    DEFAULT_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE,
    DEFAULT_MAX_POWER,
    DEFAULT_TIME_RESOLUTION,
    DEFAULT_VIOLATION_PENALTY,
    calculate_state_dim,
)


class BearEnvWrapper(gym.Env):
    """
    BEAR 环境适配器，使其兼容 DROPT 接口
    
    该类包装了 BEAR 的 BuildingEnvReal 环境，提供以下功能：
    1. 状态空间和动作空间的适配
    2. 奖励函数的适配
    3. 与 DROPT 训练流程的兼容
    4. 专家控制器接口（在第二阶段实现）
    
    状态空间（维度：3*roomnum + 3）：
    - 各房间温度 (roomnum)
    - 室外温度 (1)
    - 全局水平辐照度 GHI (roomnum)
    - 地面温度 (1)
    - 人员热负荷 (roomnum)
    
    动作空间（维度：roomnum）：
    - 每个房间的 HVAC 功率：[-1, 1]
    - 负值表示制冷，正值表示制热
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        building_type: str = 'OfficeSmall',      # 建筑类型
        weather_type: str = 'Hot_Dry',           # 气候类型
        location: str = 'Tucson',                # 地理位置
        target_temp: float = DEFAULT_TARGET_TEMP,               # 目标温度 (°C)
        temp_tolerance: float = DEFAULT_TEMP_TOLERANCE,             # 温度容差 (°C)
        max_power: int = DEFAULT_MAX_POWER,                   # HVAC 最大功率 (W)
        time_resolution: int = DEFAULT_TIME_RESOLUTION,             # 时间分辨率 (秒)
        energy_weight: float = DEFAULT_ENERGY_WEIGHT,            # 能耗权重
        temp_weight: float = DEFAULT_TEMP_WEIGHT,              # 温度偏差权重
        episode_length: Optional[int] = None,    # 回合长度（None表示使用完整年度数据）
        add_violation_penalty: bool = False,     # 是否添加温度越界惩罚
        violation_penalty: float = DEFAULT_VIOLATION_PENALTY,        # 越界惩罚系数
        reward_scale: float = DEFAULT_REWARD_SCALE,               # 奖励缩放系数（降低奖励尺度）
        expert_type: Optional[str] = None,       # 专家控制器类型（第二阶段实现）
        **kwargs                                 # 其他传递给 ParameterGenerator 的参数
    ):
        """
        初始化 BEAR 环境适配器

        参数：
        - building_type: 建筑类型，可选值见 BEAR 文档
        - weather_type: 气候类型，可选值见 BEAR 文档
        - location: 地理位置，用于地面温度数据
        - target_temp: 目标温度 (°C)
        - temp_tolerance: 温度容差 (°C)
        - max_power: HVAC 最大功率 (W)
        - time_resolution: 时间分辨率 (秒)，默认 3600 (1小时)
        - energy_weight: 能耗权重
        - temp_weight: 温度偏差权重
        - episode_length: 回合长度，None 表示使用完整数据
        - add_violation_penalty: 是否添加温度越界惩罚
        - violation_penalty: 越界惩罚系数
        - reward_scale: 奖励缩放系数，默认0.1（将奖励缩小10倍，稳定训练）
        - expert_type: 专家控制器类型（'mpc', 'pid', 'rule_based'）
        """
        super(BearEnvWrapper, self).__init__()
        
        # ========== 保存环境参数 ==========
        self.building_type = building_type
        self.weather_type = weather_type
        self.location = location
        self.target_temp = target_temp
        self.temp_tolerance = temp_tolerance
        self.max_power = max_power
        self.time_resolution = time_resolution
        self.energy_weight = energy_weight
        self.temp_weight = temp_weight
        self.episode_length = episode_length
        self.add_violation_penalty = add_violation_penalty
        self.violation_penalty = violation_penalty
        self.reward_scale = reward_scale  # 奖励缩放系数
        self.expert_type = expert_type
        
        # ========== 生成 BEAR 环境参数 ==========
        try:
            self.bear_params = ParameterGenerator(
                Building=building_type,
                Weather=weather_type,
                Location=location,
                target=target_temp,
                reward_gamma=(energy_weight, temp_weight),
                max_power=max_power,
                time_reso=time_resolution,
                temp_range=(-40, 40),
                spacetype='continuous',
                root='bear/BEAR/Data/',
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"生成 BEAR 参数失败: {e}")
        
        # ========== 创建 BEAR 环境 ==========
        try:
            self.bear_env = BuildingEnvReal(self.bear_params)
        except Exception as e:
            raise RuntimeError(f"创建 BEAR 环境失败: {e}")
        
        # ========== 获取环境信息 ==========
        self.roomnum = self.bear_params['roomnum']
        # BEAR 状态维度计算：
        # - 房间温度 + 室外温度: roomnum + 1
        # - GHI: roomnum
        # - 地面温度: 1
        # - 人员热负荷: roomnum
        # 总维度 = (roomnum+1) + roomnum + 1 + roomnum = 3*roomnum + 2
        self.state_dim = 3 * self.roomnum + 2
        self.action_dim = self.roomnum

        # ========== 适配状态空间 ==========
        self.observation_space = self._adapt_observation_space()

        # ========== 验证状态维度（修复：使用显式异常而非断言） ==========
        # 检查实际的observation_space维度是否与计算的state_dim一致
        actual_obs_dim = self.observation_space.shape[0]

        # 修复：使用显式异常确保维度一致（断言在优化模式下会被禁用）
        if actual_obs_dim != self.state_dim:
            raise ValueError(
                f"状态维度不一致! "
                f"计算的state_dim: {self.state_dim} (3*{self.roomnum}+2), "
                f"实际obs_space维度: {actual_obs_dim}. "
                f"请检查 BEAR 环境配置或状态维度计算公式。"
            )

        print(f"✓ 状态维度验证通过: {self.state_dim} (3*{self.roomnum}+2)")
        
        # ========== 适配动作空间 ==========
        self.action_space = self._adapt_action_space()
        
        # ========== 初始化计数器 ==========
        self.current_step = 0
        self.total_reward = 0.0
        
        # ========== 专家控制器 ==========
        self.expert_controller = None
        if expert_type is not None:
            self._init_expert_controller(expert_type)
    
    def _init_expert_controller(self, expert_type: str):
        """
        初始化专家控制器

        Args:
            expert_type: 控制器类型 ('mpc', 'pid', 'rule', 'bangbang')
        """
        try:
            from env.building_expert_controller import create_expert_controller
            self.expert_controller = create_expert_controller(expert_type, self)
            print(f"✓ 专家控制器 '{expert_type}' 初始化成功")
        except Exception as e:
            print(f"警告：专家控制器 '{expert_type}' 初始化失败: {e}")
            self.expert_controller = None

    def _adapt_observation_space(self) -> spaces.Box:
        """
        适配状态空间
        
        BEAR 状态空间：
        - 房间温度 (roomnum)
        - 室外温度 (1)
        - GHI (roomnum)
        - 地面温度 (1)
        - 人员热负荷 (roomnum)
        
        返回：
        - gym.spaces.Box: 适配后的状态空间
        """
        # 直接使用 BEAR 的状态空间定义
        return self.bear_env.observation_space
    
    def _adapt_action_space(self) -> spaces.Box:
        """
        适配动作空间
        
        BEAR 动作空间：
        - 每个房间的 HVAC 功率：[-1, 1]
        - 负值表示制冷，正值表示制热
        
        返回：
        - gym.spaces.Box: 适配后的动作空间
        """
        # 直接使用 BEAR 的动作空间定义
        return self.bear_env.action_space
    
    def _adapt_state(self, bear_state: np.ndarray) -> np.ndarray:
        """
        适配状态向量

        参数：
        - bear_state: BEAR 环境返回的状态

        返回：
        - np.ndarray: 适配后的状态（保持原格式）
        """
        # BEAR 的状态格式已经符合要求，直接返回
        state = bear_state.astype(np.float32)

        # ========== 维度检查和修正 ==========
        # 确保状态维度与预期一致
        expected_dim = self.state_dim
        actual_dim = len(state)

        if actual_dim != expected_dim:
            # 如果维度不匹配，进行填充或截断
            if actual_dim < expected_dim:
                # 填充零值
                padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                state = np.concatenate([state, padding])
                print(f"警告: 状态维度不匹配! 期望 {expected_dim}, 实际 {actual_dim}, 已填充零值")
            else:
                # 截断多余维度
                state = state[:expected_dim]
                print(f"警告: 状态维度不匹配! 期望 {expected_dim}, 实际 {actual_dim}, 已截断")

        return state
    
    def _adapt_action(self, dropt_action: np.ndarray) -> np.ndarray:
        """
        适配动作向量
        
        参数：
        - dropt_action: DROPT 策略输出的动作
        
        返回：
        - np.ndarray: 适配后的动作（保持原格式）
        """
        # DROPT 的动作格式已经是 [-1, 1]，直接传递给 BEAR
        return dropt_action.astype(np.float32)
    
    def _adapt_reward(
        self,
        bear_reward: float,
        state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        适配奖励函数

        参数：
        - bear_reward: BEAR 环境返回的奖励
        - state: 当前状态
        - info: 环境信息字典

        返回：
        - float: 适配后的奖励
        """
        reward = bear_reward

        # 可选：添加温度越界惩罚
        if self.add_violation_penalty:
            zone_temps = info.get('zone_temperature', state[:self.roomnum])
            target = self.target_temp
            tolerance = self.temp_tolerance

            # 计算越界惩罚
            violation_count = 0
            for temp in zone_temps:
                if temp < target - tolerance or temp > target + tolerance:
                    violation_count += 1

            if violation_count > 0:
                reward -= self.violation_penalty * violation_count

        # ========== 奖励缩放 ==========
        # 将奖励缩放到更小的范围，降低Q值和损失的尺度
        # 例如: reward_scale=0.1 将奖励缩小10倍
        reward = reward * self.reward_scale

        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境

        参数：
        - seed: 随机种子
        - options: 重置选项

        返回：
        - state: 初始状态
        - info: 环境信息
        """
        # 重置 BEAR 环境
        bear_state, bear_info = self.bear_env.reset(seed=seed, options=options)

        # 适配状态
        state = self._adapt_state(bear_state)

        # 重置计数器
        self.current_step = 0
        self.total_reward = 0.0

        # 修复：重置专家控制器状态（避免上一个episode的状态影响下一个episode）
        if self.expert_controller is not None:
            self.expert_controller.reset()

        # 返回空的info字典以兼容Tianshou
        info = {}

        return state, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步
        
        参数：
        - action: 动作向量
        
        返回：
        - next_state: 下一个状态
        - reward: 奖励
        - done: 是否结束（终止）
        - truncated: 是否截断（超时）
        - info: 环境信息
        """
        # 适配动作
        bear_action = self._adapt_action(action)
        
        # 执行 BEAR 环境的 step
        bear_state, bear_reward, bear_done, bear_truncated, bear_info = self.bear_env.step(bear_action)
        
        # 适配状态
        state = self._adapt_state(bear_state)
        
        # 适配奖励
        reward = self._adapt_reward(bear_reward, state, bear_info)
        
        # 更新计数器
        self.current_step += 1
        self.total_reward += reward
        
        # 检查是否达到回合长度限制
        done = bear_done
        truncated = bear_truncated
        if self.episode_length is not None and self.current_step >= self.episode_length:
            truncated = True
            done = True
        
        # 构造信息字典
        zone_temps = np.array(
            bear_info.get('zone_temperature', state[:self.roomnum]),
            dtype=np.float32
        )
        temp_delta = zone_temps - self.target_temp
        abs_delta = np.abs(temp_delta)
        comfort_mean = float(abs_delta.mean()) if abs_delta.size else 0.0
        comfort_max = float(abs_delta.max()) if abs_delta.size else 0.0
        comfort_violation = int((abs_delta > self.temp_tolerance).sum()) if abs_delta.size else 0

        avg_action_usage = float(np.mean(np.abs(bear_action)))
        energy_kwh = avg_action_usage * self.max_power * (self.time_resolution / 3600.0) / 1000.0

        info = {
            'bear_info': bear_info,
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'hvac_power_ratio': avg_action_usage,
            'hvac_energy_kwh': energy_kwh,
            'comfort_mean_abs_dev': comfort_mean,
            'comfort_max_dev': comfort_max,
            'comfort_violations': comfort_violation,
        }
        
        # 添加专家动作
        if self.expert_controller is not None:
            try:
                expert_action = self.expert_controller.get_action(state)
                info['expert_action'] = expert_action
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # 修复：只捕获预期的异常类型，避免隐藏严重错误
                print(f"警告：获取专家动作失败: {e}")
                # 使用零动作作为备用（保守策略）
                info['expert_action'] = np.zeros(self.action_dim, dtype=np.float32)
        
        return state, reward, done, truncated, info
    
    def render(self, mode: str = 'human'):
        """
        渲染环境（可选实现）
        
        参数：
        - mode: 渲染模式
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Total Reward: {self.total_reward:.2f}")
        else:
            raise NotImplementedError(f"渲染模式 '{mode}' 未实现")
    
    def close(self):
        """关闭环境"""
        pass


def make_building_env(
    building_type: str = 'OfficeSmall',
    weather_type: str = 'Hot_Dry',
    location: str = 'Tucson',
    training_num: int = 1,
    test_num: int = 1,
    vector_env_type: str = 'dummy',
    **kwargs
) -> Tuple[BearEnvWrapper, Any, Any]:
    """
    创建建筑环境（兼容 DROPT 接口）

    参数：
    - building_type: 建筑类型
    - weather_type: 气候类型
    - location: 地理位置
    - training_num: 训练环境数量
    - test_num: 测试环境数量
    - **kwargs: 传递给 BearEnvWrapper 的其他参数

    返回：
    - env: 单个环境实例
    - train_envs: 训练环境向量
    - test_envs: 测试环境向量
    """
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    # 修复：定义环境工厂函数，避免 lambda 闭包共享可变对象的潜在风险
    def env_factory():
        return BearEnvWrapper(
            building_type=building_type,
            weather_type=weather_type,
            location=location,
            **kwargs
        )

    # 创建单个环境实例
    env = env_factory()

    vector_cls = DummyVectorEnv if vector_env_type != 'subproc' else SubprocVectorEnv

    # 创建训练环境向量
    train_envs = vector_cls([env_factory for _ in range(training_num)])

    # 创建测试环境向量
    test_envs = vector_cls([env_factory for _ in range(test_num)])

    return env, train_envs, test_envs

