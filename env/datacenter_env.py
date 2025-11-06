# ========================================
# 数据中心空调优化环境
# ========================================
# 模拟数据中心的热力学行为和空调控制
# 基于DROPT框架改造

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Any
import pandas as pd
from .thermal_model import ThermalModel
from .expert_controller import ExpertController


class DataCenterEnv(gym.Env):
    """
    数据中心空调优化环境
    
    状态空间：
    - T_in: 机房内部温度 (°C)
    - T_out: 室外温度 (°C)
    - H_in: 机房内部湿度 (%)
    - IT_load: IT设备负载 (kW)
    - T_supply: 当前供风温度 (°C)
    - reward_last: 上一步奖励
    
    动作空间：
    - T_set: 空调设定温度 (°C) [18-28]
    - fan_speed: 风机转速比例 [0.3-1.0]
    
    奖励函数：
    reward = -α*能耗 - β*温度偏差² - γ*越界惩罚
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_crac_units: int = 4,           # CRAC空调数量
        target_temp: float = 24.0,         # 目标温度 (°C)
        temp_tolerance: float = 2.0,       # 温度容差 (°C)
        time_step: float = 5.0,            # 时间步长 (分钟)
        episode_length: int = 288,         # 回合长度 (24小时，每5分钟一步)
        energy_weight: float = 1.0,        # 能耗权重 α
        temp_weight: float = 10.0,         # 温度偏差权重 β
        violation_penalty: float = 100.0,  # 越界惩罚 γ
        use_real_weather: bool = False,    # 是否使用真实气象数据
        weather_file: str = None,          # 气象数据文件路径
        workload_file: str = None,         # 负载数据文件路径
    ):
        super(DataCenterEnv, self).__init__()
        
        # ========== 环境参数 ==========
        self.num_crac = num_crac_units
        self.target_temp = target_temp
        self.temp_tolerance = temp_tolerance
        self.time_step = time_step / 60.0  # 转换为小时
        self.episode_length = episode_length
        
        # ========== 奖励函数权重 ==========
        self.alpha = energy_weight
        self.beta = temp_weight
        self.gamma = violation_penalty
        
        # ========== 物理约束 ==========
        self.T_min = target_temp - temp_tolerance  # 最低温度
        self.T_max = target_temp + temp_tolerance  # 最高温度
        self.H_min = 40.0   # 最低湿度 (%)
        self.H_max = 60.0   # 最高湿度 (%)
        
        # ========== 状态空间定义 ==========
        # [T_in, T_out, H_in, IT_load, T_supply_1, ..., T_supply_n, reward_last]
        self.state_dim = 4 + num_crac_units + 1
        
        # 状态范围：[最小值, 最大值]
        state_low = np.array([
            15.0,   # T_in 最低
            -10.0,  # T_out 最低（冬季）
            20.0,   # H_in 最低
            50.0,   # IT_load 最低 (kW)
        ] + [15.0] * num_crac_units +  # T_supply 最低
            [-1000.0])  # reward_last 最低
        
        state_high = np.array([
            35.0,   # T_in 最高
            45.0,   # T_out 最高（夏季）
            80.0,   # H_in 最高
            500.0,  # IT_load 最高 (kW)
        ] + [30.0] * num_crac_units +  # T_supply 最高
            [0.0])  # reward_last 最高
        
        self.observation_space = spaces.Box(
            low=state_low,
            high=state_high,
            dtype=np.float32
        )
        
        # ========== 动作空间定义 ==========
        # 每个CRAC单元：[T_set, fan_speed]
        self.action_dim = num_crac_units * 2
        
        # 动作范围（归一化到[-1, 1]）
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
        # 动作的物理范围
        self.T_set_min = 18.0   # 最低设定温度 (°C)
        self.T_set_max = 28.0   # 最高设定温度 (°C)
        self.fan_min = 0.3      # 最低风速比例
        self.fan_max = 1.0      # 最高风速比例
        
        # ========== 初始化子模块 ==========
        # 热力学模型（模拟温度动态）
        self.thermal_model = ThermalModel(
            num_crac=num_crac_units,
            time_step=self.time_step
        )
        
        # 专家控制器（提供专家动作）
        self.expert_controller = ExpertController(
            num_crac=num_crac_units,
            target_temp=target_temp,
            controller_type='pid'  # 可选：'pid', 'mpc', 'rule_based'
        )
        
        # ========== 加载外部数据 ==========
        self.use_real_weather = use_real_weather
        if use_real_weather and weather_file:
            self.weather_data = pd.read_csv(weather_file)
        else:
            self.weather_data = None
        
        if workload_file:
            self.workload_data = pd.read_csv(workload_file)
        else:
            self.workload_data = None
        
        # ========== 环境状态变量 ==========
        self._state = None
        self._num_steps = 0
        self._terminated = False
        self._episode_energy = 0.0  # 累积能耗
        self._episode_violations = 0  # 温度越界次数
        
        # 内部物理状态
        self.T_in = target_temp      # 机房温度
        self.T_out = 25.0            # 室外温度
        self.H_in = 50.0             # 机房湿度
        self.IT_load = 200.0         # IT负载
        self.T_supply = np.ones(num_crac_units) * 20.0  # 各CRAC供风温度
        
    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态
        
        返回：
        - state: 初始状态向量
        """
        self._num_steps = 0
        self._terminated = False
        self._episode_energy = 0.0
        self._episode_violations = 0
        
        # ========== 随机初始化物理状态 ==========
        # 机房温度：目标温度附近随机扰动
        self.T_in = self.target_temp + np.random.uniform(-1.0, 1.0)
        
        # 室外温度：根据季节随机
        if self.use_real_weather and self.weather_data is not None:
            # 从真实数据随机选择一天的起点
            start_idx = np.random.randint(0, len(self.weather_data) - self.episode_length)
            self.weather_start_idx = start_idx
            self.T_out = self.weather_data.iloc[start_idx]['temperature']
        else:
            # 模拟数据：15-35°C随机
            self.T_out = np.random.uniform(15.0, 35.0)
        
        # 湿度：正常范围随机
        self.H_in = np.random.uniform(45.0, 55.0)
        
        # IT负载：根据时间模拟（早上低，下午高）
        if self.workload_data is not None:
            start_idx = np.random.randint(0, len(self.workload_data) - self.episode_length)
            self.workload_start_idx = start_idx
            self.IT_load = self.workload_data.iloc[start_idx]['load']
        else:
            # 模拟负载：100-400kW
            self.IT_load = np.random.uniform(100.0, 400.0)
        
        # 供风温度：初始设定为20°C
        self.T_supply = np.ones(self.num_crac) * 20.0
        
        # ========== 构造状态向量 ==========
        self._state = self._get_state()
        
        return self._state
    
    def _get_state(self) -> np.ndarray:
        """
        构造当前状态向量
        
        返回：
        - state: [T_in, T_out, H_in, IT_load, T_supply_1, ..., T_supply_n, reward_last]
        """
        state = np.array([
            self.T_in,
            self.T_out,
            self.H_in,
            self.IT_load,
        ] + list(self.T_supply) + [
            0.0 if self._num_steps == 0 else self._last_reward
        ], dtype=np.float32)
        
        return state
    
    def _denormalize_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将归一化动作[-1, 1]转换为物理动作
        
        参数：
        - action: 归一化动作向量 (2*num_crac,)
        
        返回：
        - T_set: 设定温度数组 (num_crac,)
        - fan_speed: 风速比例数组 (num_crac,)
        """
        # 裁剪到[-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # 分离温度和风速动作
        T_set_norm = action[0::2]  # 偶数索引：温度
        fan_norm = action[1::2]    # 奇数索引：风速
        
        # 反归一化
        T_set = (T_set_norm + 1.0) / 2.0 * (self.T_set_max - self.T_set_min) + self.T_set_min
        fan_speed = (fan_norm + 1.0) / 2.0 * (self.fan_max - self.fan_min) + self.fan_min
        
        return T_set, fan_speed
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步环境交互
        
        参数：
        - action: 策略输出的动作（归一化）
        
        返回：
        - next_state: 下一个状态
        - reward: 奖励值
        - done: 是否结束
        - info: 额外信息（包含专家动作、能耗等）
        """
        assert not self._terminated, "回合已结束，请调用reset()"
        
        # ========== 动作处理 ==========
        T_set, fan_speed = self._denormalize_action(action)
        
        # ========== 获取专家动作（用于行为克隆） ==========
        expert_action = self.expert_controller.get_action(
            T_in=self.T_in,
            T_out=self.T_out,
            H_in=self.H_in,
            IT_load=self.IT_load
        )
        
        # ========== 更新环境动态 ==========
        # 使用热力学模型计算下一时刻的温度、湿度
        next_T_in, next_H_in, next_T_supply, energy_consumed = self.thermal_model.step(
            T_in=self.T_in,
            T_out=self.T_out,
            H_in=self.H_in,
            IT_load=self.IT_load,
            T_set=T_set,
            fan_speed=fan_speed
        )
        
        # ========== 更新外部扰动 ==========
        self._num_steps += 1
        
        # 更新室外温度
        if self.use_real_weather and self.weather_data is not None:
            idx = self.weather_start_idx + self._num_steps
            if idx < len(self.weather_data):
                self.T_out = self.weather_data.iloc[idx]['temperature']
        else:
            # 模拟日变化：正弦波 + 随机噪声
            hour = (self._num_steps * self.time_step) % 24
            self.T_out += 0.5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.2)
            self.T_out = np.clip(self.T_out, 15.0, 40.0)
        
        # 更新IT负载
        if self.workload_data is not None:
            idx = self.workload_start_idx + self._num_steps
            if idx < len(self.workload_data):
                self.IT_load = self.workload_data.iloc[idx]['load']
        else:
            # 模拟负载变化：工作时间高，夜间低
            hour = (self._num_steps * self.time_step) % 24
            if 8 <= hour <= 18:  # 工作时间
                self.IT_load = 300.0 + np.random.uniform(-50, 50)
            else:  # 夜间
                self.IT_load = 150.0 + np.random.uniform(-30, 30)
        
        # ========== 计算奖励 ==========
        reward, reward_info = self._compute_reward(
            T_in=next_T_in,
            energy=energy_consumed
        )
        
        # ========== 更新状态 ==========
        self.T_in = next_T_in
        self.H_in = next_H_in
        self.T_supply = next_T_supply
        self._last_reward = reward
        self._episode_energy += energy_consumed
        
        # 检查温度越界
        if next_T_in < self.T_min or next_T_in > self.T_max:
            self._episode_violations += 1
        
        # ========== 构造下一状态 ==========
        next_state = self._get_state()
        
        # ========== 检查是否结束 ==========
        done = self._num_steps >= self.episode_length
        if done:
            self._terminated = True
        
        # ========== 返回信息 ==========
        info = {
            'expert_action': expert_action,  # 专家动作（用于BC训练）
            'energy': energy_consumed,       # 当前步能耗 (kWh)
            'T_in': next_T_in,               # 机房温度
            'T_out': self.T_out,             # 室外温度
            'IT_load': self.IT_load,         # IT负载
            'temp_violation': next_T_in < self.T_min or next_T_in > self.T_max,
            'episode_energy': self._episode_energy,  # 累积能耗
            'episode_violations': self._episode_violations,  # 累积越界次数
            **reward_info  # 奖励分解信息
        }
        
        return next_state, reward, done, info
    
    def _compute_reward(self, T_in: float, energy: float) -> Tuple[float, Dict]:
        """
        计算奖励函数
        
        奖励设计：
        reward = -α*能耗 - β*温度偏差² - γ*越界惩罚
        
        参数：
        - T_in: 当前机房温度
        - energy: 当前步能耗 (kWh)
        
        返回：
        - reward: 总奖励
        - info: 奖励分解信息
        """
        # 1. 能耗惩罚（归一化到合理范围）
        energy_penalty = self.alpha * energy / 100.0  # 假设单步最大能耗~100kWh
        
        # 2. 温度偏差惩罚
        temp_deviation = T_in - self.target_temp
        temp_penalty = self.beta * (temp_deviation ** 2)
        
        # 3. 温度越界惩罚（硬约束）
        if T_in < self.T_min or T_in > self.T_max:
            violation_penalty = self.gamma
        else:
            violation_penalty = 0.0
        
        # 总奖励（负值，最小化）
        reward = -(energy_penalty + temp_penalty + violation_penalty)
        
        # 奖励分解信息
        info = {
            'reward_energy': -energy_penalty,
            'reward_temp': -temp_penalty,
            'reward_violation': -violation_penalty,
            'temp_deviation': temp_deviation
        }
        
        return reward, info
    
    def render(self, mode='human'):
        """
        可视化当前状态
        """
        if mode == 'human':
            print(f"Step: {self._num_steps}/{self.episode_length}")
            print(f"  T_in: {self.T_in:.2f}°C (目标: {self.target_temp:.2f}°C)")
            print(f"  T_out: {self.T_out:.2f}°C")
            print(f"  H_in: {self.H_in:.1f}%")
            print(f"  IT_load: {self.IT_load:.1f}kW")
            print(f"  累积能耗: {self._episode_energy:.2f}kWh")
            print(f"  温度越界次数: {self._episode_violations}")
    
    def close(self):
        """
        清理资源
        """
        pass


def make_datacenter_env(training_num: int = 1, test_num: int = 1, **kwargs):
    """
    创建数据中心环境（兼容DROPT接口）
    
    参数：
    - training_num: 训练环境数量
    - test_num: 测试环境数量
    - **kwargs: 传递给DataCenterEnv的参数
    
    返回：
    - env: 单个环境实例
    - train_envs: 训练环境向量
    - test_envs: 测试环境向量
    """
    from tianshou.env import DummyVectorEnv
    
    # 创建单个环境实例
    env = DataCenterEnv(**kwargs)
    
    # 创建训练环境向量
    train_envs = DummyVectorEnv([
        lambda: DataCenterEnv(**kwargs) for _ in range(training_num)
    ])
    
    # 创建测试环境向量
    test_envs = DummyVectorEnv([
        lambda: DataCenterEnv(**kwargs) for _ in range(test_num)
    ])
    
    return env, train_envs, test_envs

