# ========================================
# 数据中心空调优化环境
# ========================================
# 模拟数据中心的热力学行为和空调控制
# 基于DROPT框架改造

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List, Optional
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
        expert_type: str = 'pid',          # 专家控制器类型
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
            controller_type=expert_type  # 可选：'pid', 'mpc', 'rule_based'
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
        self._episode_records: List[Dict[str, float]] = []
        self._reset_metrics()
        
        # 内部物理状态
        self.T_in = target_temp      # 机房温度
        self.T_out = 25.0            # 室外温度
        self.H_in = 50.0             # 机房湿度
        self.IT_load = 200.0         # IT负载
        self.T_supply = np.ones(num_crac_units) * 20.0  # 各CRAC供风温度

    def _reset_metrics(self) -> None:
        """重置直接监控的episode级指标累加器。"""
        self._metric_energy_sum = 0.0
        self._metric_pue_sum = 0.0
        self._metric_pue_steps = 0
        self._metric_violation_sum = 0.0
        self._metric_steps = 0

    def _finalize_episode(self) -> None:
        """在episode结束时汇总指标。"""
        if self._metric_steps == 0:
            return
        avg_pue = None
        if self._metric_pue_steps:
            avg_pue = self._metric_pue_sum / self._metric_pue_steps
        avg_violations = self._metric_violation_sum / self._metric_steps if self._metric_steps else None
        record = {
            'avg_energy': float(self._episode_energy),
            'avg_pue': float(avg_pue) if avg_pue is not None else None,
            'avg_violations': float(avg_violations) if avg_violations is not None else None,
        }
        self._episode_records.append(record)
        self._reset_metrics()
        self._episode_energy = 0.0
        self._episode_violations = 0
        
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态

        参数：
        - seed: 随机种子（可选，用于 Gymnasium 兼容性）
        - options: 额外选项（可选，用于 Gymnasium 兼容性）

        返回：
        - state: 初始状态向量
        - info: 环境信息字典（符合 Gymnasium API）

        注意：修复了 API 兼容性问题，现在返回 (state, info) 元组
        """
        # 设置随机种子（如果提供）
        if seed is not None:
            np.random.seed(seed)

        self._num_steps = 0
        self._terminated = False
        self._episode_energy = 0.0
        self._episode_violations = 0
        self._last_reward = 0.0
        self._reset_metrics()

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

        # ========== 构造初始信息字典 ==========
        info = {
            'T_in': self.T_in,
            'T_out': self.T_out,
            'H_in': self.H_in,
            'IT_load': self.IT_load,
        }

        # 返回 (state, info) 元组（符合 Gymnasium API）
        return self._state, info
    
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境交互

        参数：
        - action: 策略输出的动作（归一化）

        返回：
        - next_state: 下一个状态
        - reward: 奖励值
        - terminated: 是否因达到终止条件而结束（如温度严重越界）
        - truncated: 是否因达到最大步数而截断
        - info: 额外信息（包含专家动作、能耗等）

        注意：修复了 API 兼容性问题，现在返回 5 个值（符合 Gymnasium API）
        """
        assert not self._terminated, "回合已结束，请调用reset()"

        # ========== 动作处理 ==========
        T_set, fan_speed = self._denormalize_action(action)
        current_it_load = float(self.IT_load)

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
        timestep_hours = max(self.time_step, 1e-6)
        cooling_power_kw = energy_consumed / timestep_hours
        pue = None
        if current_it_load > 1e-6:
            pue = (cooling_power_kw + current_it_load) / current_it_load

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
        temp_violation = next_T_in < self.T_min or next_T_in > self.T_max
        if temp_violation:
            self._episode_violations += 1
        self._metric_energy_sum += energy_consumed
        if pue is not None:
            self._metric_pue_sum += pue
            self._metric_pue_steps += 1
        self._metric_violation_sum += int(temp_violation)
        self._metric_steps += 1
        episode_pue_mean = self._metric_pue_sum / self._metric_pue_steps if self._metric_pue_steps else None

        # ========== 构造下一状态 ==========
        next_state = self._get_state()

        # ========== 检查终止条件 ==========
        # terminated: 因达到终止条件而结束（如严重越界）
        # truncated: 因达到最大步数而截断
        terminated = False  # 数据中心场景通常不提前终止
        truncated = self._num_steps >= self.episode_length

        if terminated or truncated:
            self._terminated = True
            self._finalize_episode()

        # ========== 返回信息 ==========
        info = {
            'expert_action': expert_action,  # 专家动作（用于BC训练）
            'energy': energy_consumed,       # 当前步能耗 (kWh)
            'T_in': next_T_in,               # 机房温度
            'T_out': self.T_out,             # 室外温度
            'IT_load': self.IT_load,         # IT负载
            'temp_violation': temp_violation,
            'pue': pue,
            'episode_pue_mean': episode_pue_mean,
            'episode_energy': self._episode_energy,  # 累积能耗
            'episode_violations': self._episode_violations,  # 累积越界次数
            **reward_info  # 奖励分解信息
        }

        # 返回 5 个值（符合 Gymnasium API）
        return next_state, reward, terminated, truncated, info
    
    def _compute_reward(self, T_in: float, energy: float) -> Tuple[float, Dict]:
        """
        计算奖励函数 (改进版 v2)

        改进点:
        1. 降低惩罚权重 (10倍) - 避免过大负奖励
        2. 添加正向奖励 (温度舒适度) - 提供正向信号
        3. 归一化能耗惩罚 - 统一奖励尺度
        4. 基础存活奖励 - 鼓励策略保持运行

        预期奖励范围: -20 ~ +15 (单步), 回合累积: +500 ~ +3000

        参数：
        - T_in: 当前机房温度 (°C)
        - energy: 当前步能耗 (kWh)

        返回：
        - reward: 总奖励
        - info: 奖励分解信息
        """
        import numpy as np

        # ========== 1. 温度舒适度奖励 (高斯型) ==========
        # 在目标温度时奖励最大, 偏离时指数衰减
        temp_error = abs(T_in - self.target_temp)
        temp_reward = 10.0 * np.exp(-0.5 * (temp_error ** 2))  # 范围: 0-10

        # ========== 2. 温度惩罚 (降低权重) ==========
        # beta: 10.0 → 1.0 (降低10倍)
        temp_penalty = 1.0 * (temp_error ** 2)

        # ========== 3. 能耗惩罚 (归一化) ==========
        # 归一化到 [0, 1], 假设单步最大能耗 10kWh
        energy_normalized = energy / 10.0
        # alpha: 1.0 → 0.1 (降低10倍)
        energy_penalty = 0.1 * energy_normalized

        # ========== 4. 越界惩罚 (降低权重) ==========
        # gamma: 100.0 → 10.0 (降低10倍)
        if T_in < self.T_min or T_in > self.T_max:
            violation_penalty = 10.0
            violation = 1
        else:
            violation_penalty = 0.0
            violation = 0

        # ========== 5. 基础存活奖励 ==========
        # 每步给予小的正奖励
        base_reward = 1.0

        # ========== 6. 总奖励 (正负平衡) ==========
        # reward = 基础 + 温度奖励 - 温度惩罚 - 能耗惩罚 - 越界惩罚
        reward = base_reward + temp_reward - temp_penalty - energy_penalty - violation_penalty

        # ========== 7. 奖励分解信息 ==========
        info = {
            'reward_base': base_reward,
            'reward_temp': temp_reward,
            'reward_temp_penalty': -temp_penalty,
            'reward_energy': -energy_penalty,
            'reward_violation': -violation_penalty,
            'temp_error': temp_error,
            'temp_deviation': T_in - self.target_temp,
            'violation': violation
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
    
    def consume_metrics(self):
        """返回并清空已完成episode的环境指标。"""
        if self._episode_records:
            records = self._episode_records
            self._episode_records = []
        elif self._metric_steps > 0:
            avg_pue = (self._metric_pue_sum / self._metric_pue_steps) if self._metric_pue_steps else None
            return {
                'avg_energy': float(self._metric_energy_sum),
                'avg_pue': float(avg_pue) if avg_pue is not None else None,
                'avg_violations': float(self._metric_violation_sum / self._metric_steps),
            }
        else:
            return None

        def _avg(key: str):
            values = [rec[key] for rec in records if rec.get(key) is not None]
            if not values:
                return None
            return float(np.mean(values))

        return {
            'avg_energy': _avg('avg_energy'),
            'avg_pue': _avg('avg_pue'),
            'avg_violations': _avg('avg_violations'),
        }

    def close(self):
        """
        清理资源
        """
        pass


def make_datacenter_env(
    training_num: int = 1,
    test_num: int = 1,
    vector_env_type: str = 'dummy',
    **kwargs
):
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
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv
    
    def env_factory():
        return DataCenterEnv(**kwargs)
    
    env = env_factory()
    vector_cls = DummyVectorEnv if vector_env_type != 'subproc' else SubprocVectorEnv

    def _build_vector_env(num: int):
        instances: List[DataCenterEnv] = []

        def factory():
            inst = env_factory()
            instances.append(inst)
            return inst

        vec = vector_cls([factory for _ in range(num)])
        setattr(vec, "_env_list", instances)
        return vec
    
    train_envs = _build_vector_env(training_num)
    test_envs = _build_vector_env(test_num)
    
    return env, train_envs, test_envs
