# ========================================
# 鲁棒性增强的数据中心环境
# ========================================
# 添加域随机化、观测噪声、动作延迟等真实因素

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Any
from collections import deque
from .datacenter_env import DataCenterEnv


class RobustDataCenterEnv(DataCenterEnv):
    """
    鲁棒性增强的数据中心环境
    
    新增特性：
    1. 域随机化 (Domain Randomization)
    2. 观测噪声 (Observation Noise)
    3. 动作延迟 (Action Delay)
    4. 动作平滑约束 (Action Smoothing)
    5. 模型不确定性 (Model Uncertainty)
    """
    
    def __init__(
        self,
        # 继承基类参数
        num_crac_units: int = 4,
        target_temp: float = 24.0,
        temp_tolerance: float = 2.0,
        time_step: float = 5.0,
        episode_length: int = 288,
        energy_weight: float = 1.0,
        temp_weight: float = 10.0,
        violation_penalty: float = 100.0,
        use_real_weather: bool = False,
        weather_file: str = None,
        workload_file: str = None,
        
        # 鲁棒性增强参数
        domain_randomization: bool = True,      # 是否启用域随机化
        observation_noise: bool = True,         # 是否添加观测噪声
        action_delay: int = 0,                  # 动作延迟步数（0=无延迟）
        action_smoothing: bool = True,          # 是否启用动作平滑
        action_change_limit: float = 0.2,       # 动作最大变化率
        model_uncertainty: bool = True,         # 是否添加模型不确定性
        
        # 域随机化范围
        thermal_mass_range: Tuple[float, float] = (0.7, 1.3),
        wall_ua_range: Tuple[float, float] = (0.8, 1.2),
        cop_range: Tuple[float, float] = (0.9, 1.1),
        capacity_range: Tuple[float, float] = (0.95, 1.05),
        
        # 观测噪声参数
        temp_noise_std: float = 0.3,            # 温度测量噪声标准差 (°C)
        humidity_noise_std: float = 2.0,        # 湿度测量噪声标准差 (%)
        load_noise_std: float = 5.0,            # 负载测量噪声标准差 (kW)
    ):
        """
        初始化鲁棒性增强环境
        """
        # 保存鲁棒性参数
        self.domain_randomization = domain_randomization
        self.observation_noise = observation_noise
        self.action_delay_steps = action_delay
        self.action_smoothing = action_smoothing
        self.action_change_limit = action_change_limit
        self.model_uncertainty = model_uncertainty
        
        # 域随机化范围
        self.thermal_mass_range = thermal_mass_range
        self.wall_ua_range = wall_ua_range
        self.cop_range = cop_range
        self.capacity_range = capacity_range
        
        # 观测噪声参数
        self.temp_noise_std = temp_noise_std
        self.humidity_noise_std = humidity_noise_std
        self.load_noise_std = load_noise_std
        
        # 初始化基类
        super().__init__(
            num_crac_units=num_crac_units,
            target_temp=target_temp,
            temp_tolerance=temp_tolerance,
            time_step=time_step,
            episode_length=episode_length,
            energy_weight=energy_weight,
            temp_weight=temp_weight,
            violation_penalty=violation_penalty,
            use_real_weather=use_real_weather,
            weather_file=weather_file,
            workload_file=workload_file,
        )
        
        # 动作延迟缓冲区
        if self.action_delay_steps > 0:
            self.action_buffer = deque(maxlen=self.action_delay_steps + 1)
        else:
            self.action_buffer = None
        
        # 上一步动作（用于平滑约束）
        self.last_action = None
        
        # 域随机化参数（每个episode随机化）
        self.randomized_params = {}
    
    def reset(self) -> np.ndarray:
        """
        重置环境，应用域随机化
        """
        # 域随机化：每个episode随机化物理参数
        if self.domain_randomization:
            self._randomize_domain()
        
        # 重置基类
        state = super().reset()
        
        # 重置动作缓冲区
        if self.action_buffer is not None:
            self.action_buffer.clear()
            # 填充初始动作（零动作）
            zero_action = np.zeros(self.action_dim)
            for _ in range(self.action_delay_steps + 1):
                self.action_buffer.append(zero_action)
        
        # 重置上一步动作
        self.last_action = np.zeros(self.action_dim)
        
        # 添加观测噪声
        if self.observation_noise:
            state = self._add_observation_noise(state)
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步，应用动作延迟和平滑约束
        """
        # 1. 动作平滑约束
        if self.action_smoothing and self.last_action is not None:
            action = self._smooth_action(action, self.last_action)
        
        # 2. 动作延迟
        if self.action_buffer is not None:
            self.action_buffer.append(action.copy())
            actual_action = self.action_buffer[0]  # 使用延迟后的动作
        else:
            actual_action = action
        
        # 3. 模型不确定性（随机扰动）
        if self.model_uncertainty:
            actual_action = self._add_action_uncertainty(actual_action)
        
        # 4. 执行动作（调用基类）
        next_state, reward, done, info = super().step(actual_action)
        
        # 5. 添加观测噪声
        if self.observation_noise:
            next_state = self._add_observation_noise(next_state)
        
        # 6. 记录信息
        info['actual_action'] = actual_action
        info['requested_action'] = action
        info['action_delay'] = self.action_delay_steps
        info['randomized_params'] = self.randomized_params
        
        # 7. 更新上一步动作
        self.last_action = action.copy()
        
        return next_state, reward, done, info
    
    def _randomize_domain(self):
        """
        域随机化：随机化物理参数
        """
        # 随机化热容
        thermal_mass_scale = np.random.uniform(*self.thermal_mass_range)
        self.thermal_model.thermal_mass *= thermal_mass_scale
        
        # 随机化墙体热传导
        wall_ua_scale = np.random.uniform(*self.wall_ua_range)
        self.thermal_model.UA_wall *= wall_ua_scale
        
        # 随机化COP
        cop_scale = np.random.uniform(*self.cop_range)
        self.thermal_model.COP_nominal *= cop_scale
        
        # 随机化CRAC容量
        capacity_scale = np.random.uniform(*self.capacity_range)
        self.thermal_model.Q_crac_max *= capacity_scale
        
        # 记录随机化参数
        self.randomized_params = {
            'thermal_mass_scale': thermal_mass_scale,
            'wall_ua_scale': wall_ua_scale,
            'cop_scale': cop_scale,
            'capacity_scale': capacity_scale,
        }
    
    def _add_observation_noise(self, state: np.ndarray) -> np.ndarray:
        """
        添加观测噪声
        
        参数：
        - state: 原始状态
        
        返回：
        - noisy_state: 添加噪声后的状态
        """
        noisy_state = state.copy()
        
        # 温度噪声 (索引0: T_in)
        noisy_state[0] += np.random.normal(0, self.temp_noise_std)
        
        # 室外温度噪声 (索引1: T_out)
        noisy_state[1] += np.random.normal(0, self.temp_noise_std)
        
        # 湿度噪声 (索引2: H_in)
        noisy_state[2] += np.random.normal(0, self.humidity_noise_std)
        noisy_state[2] = np.clip(noisy_state[2], 20.0, 90.0)
        
        # 负载噪声 (索引3: IT_load)
        noisy_state[3] += np.random.normal(0, self.load_noise_std)
        noisy_state[3] = np.clip(noisy_state[3], 0.0, 1000.0)
        
        # 供风温度噪声 (索引4到4+num_crac-1)
        for i in range(4, 4 + self.num_crac):
            noisy_state[i] += np.random.normal(0, self.temp_noise_std * 0.5)
        
        return noisy_state
    
    def _smooth_action(self, action: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        """
        动作平滑约束
        
        参数：
        - action: 当前动作
        - last_action: 上一步动作
        
        返回：
        - smoothed_action: 平滑后的动作
        """
        # 计算动作变化
        action_change = action - last_action
        
        # 限制变化幅度
        action_change = np.clip(
            action_change,
            -self.action_change_limit,
            self.action_change_limit
        )
        
        # 平滑后的动作
        smoothed_action = last_action + action_change
        
        # 确保在动作空间范围内
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
        
        return smoothed_action
    
    def _add_action_uncertainty(self, action: np.ndarray) -> np.ndarray:
        """
        添加动作执行不确定性
        
        参数：
        - action: 原始动作
        
        返回：
        - uncertain_action: 添加不确定性后的动作
        """
        # 添加小的随机扰动（模拟执行器误差）
        noise = np.random.normal(0, 0.02, size=action.shape)
        uncertain_action = action + noise
        
        # 确保在动作空间范围内
        uncertain_action = np.clip(uncertain_action, -1.0, 1.0)
        
        return uncertain_action


def make_robust_datacenter_env(
    num_crac: int = 4,
    robustness_level: str = 'medium',
    **kwargs
) -> RobustDataCenterEnv:
    """
    创建鲁棒性增强环境的工厂函数
    
    参数：
    - num_crac: CRAC数量
    - robustness_level: 鲁棒性级别 ('low', 'medium', 'high')
    - **kwargs: 其他参数
    
    返回：
    - env: 鲁棒性增强环境实例
    """
    # 根据鲁棒性级别设置参数
    if robustness_level == 'low':
        config = {
            'domain_randomization': True,
            'observation_noise': False,
            'action_delay': 0,
            'action_smoothing': False,
            'model_uncertainty': False,
            'thermal_mass_range': (0.9, 1.1),
            'wall_ua_range': (0.95, 1.05),
            'cop_range': (0.98, 1.02),
            'temp_noise_std': 0.1,
        }
    elif robustness_level == 'medium':
        config = {
            'domain_randomization': True,
            'observation_noise': True,
            'action_delay': 1,
            'action_smoothing': True,
            'model_uncertainty': True,
            'thermal_mass_range': (0.7, 1.3),
            'wall_ua_range': (0.8, 1.2),
            'cop_range': (0.9, 1.1),
            'temp_noise_std': 0.3,
            'action_change_limit': 0.2,
        }
    elif robustness_level == 'high':
        config = {
            'domain_randomization': True,
            'observation_noise': True,
            'action_delay': 2,
            'action_smoothing': True,
            'model_uncertainty': True,
            'thermal_mass_range': (0.5, 1.5),
            'wall_ua_range': (0.6, 1.4),
            'cop_range': (0.8, 1.2),
            'temp_noise_std': 0.5,
            'humidity_noise_std': 3.0,
            'load_noise_std': 10.0,
            'action_change_limit': 0.15,
        }
    else:
        raise ValueError(f"未知的鲁棒性级别: {robustness_level}")
    
    # 合并用户提供的参数
    config.update(kwargs)
    
    # 创建环境
    env = RobustDataCenterEnv(num_crac_units=num_crac, **config)
    
    return env


if __name__ == '__main__':
    """测试鲁棒性增强环境"""
    print("="*60)
    print("测试鲁棒性增强环境")
    print("="*60)
    
    # 测试不同鲁棒性级别
    for level in ['low', 'medium', 'high']:
        print(f"\n测试鲁棒性级别: {level}")
        print("-"*40)
        
        env = make_robust_datacenter_env(num_crac=4, robustness_level=level)
        
        state = env.reset()
        print(f"初始状态: {state[:5]}")  # 只打印前5个元素
        print(f"随机化参数: {env.randomized_params}")
        
        # 运行几步
        for step in range(5):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            print(f"\n步骤 {step+1}:")
            print(f"  请求动作: {action[:2]}")  # 只打印前2个元素
            print(f"  实际动作: {info['actual_action'][:2]}")
            print(f"  奖励: {reward:.2f}")
            print(f"  温度: {next_state[0]:.2f}°C")
            
            if done:
                break
        
        print(f"\n✓ {level}级别测试完成")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)

