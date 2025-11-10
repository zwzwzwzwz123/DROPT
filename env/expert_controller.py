# ========================================
# 专家控制器
# ========================================
# 提供专家动作用于行为克隆训练
# 支持PID、MPC、基于规则的控制器

import numpy as np
from typing import Tuple, List
from collections import deque


class ExpertController:
    """
    专家控制器基类
    
    为数据中心空调提供专家动作（用于行为克隆训练）
    """
    
    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        controller_type: str = 'pid'
    ):
        """
        初始化专家控制器
        
        参数：
        - num_crac: CRAC单元数量
        - target_temp: 目标温度 (°C)
        - controller_type: 控制器类型 ('pid', 'mpc', 'rule_based')
        """
        self.num_crac = num_crac
        self.target_temp = target_temp
        self.controller_type = controller_type
        
        # 根据类型初始化具体控制器
        if controller_type == 'pid':
            self.controller = PIDController(num_crac, target_temp)
        elif controller_type == 'mpc':
            self.controller = MPCController(num_crac, target_temp)
        elif controller_type == 'rule_based':
            self.controller = RuleBasedController(num_crac, target_temp)
        else:
            raise ValueError(f"未知的控制器类型: {controller_type}")
    
    def get_action(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float
    ) -> np.ndarray:
        """
        获取专家动作（归一化到[-1, 1]）
        
        参数：
        - T_in: 机房温度 (°C)
        - T_out: 室外温度 (°C)
        - H_in: 机房湿度 (%)
        - IT_load: IT负载 (kW)
        
        返回：
        - action: 归一化动作向量 (2*num_crac,)
        """
        return self.controller.get_action(T_in, T_out, H_in, IT_load)


class PIDController:
    """
    PID控制器
    
    经典的比例-积分-微分控制
    适用于温度控制等单变量系统
    """
    
    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        Kp: float = 2.0,    # 比例增益
        Ki: float = 0.1,    # 积分增益
        Kd: float = 0.5,    # 微分增益
        dt: float = 5.0,    # 时间步长（分钟），修复：添加dt参数
    ):
        self.num_crac = num_crac
        self.target_temp = target_temp

        # PID参数
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt  # 修复：保存时间步长

        # 状态变量
        self.error_integral = 0.0  # 积分项
        self.last_error = 0.0      # 上一次误差（用于微分）
        self.error_history = deque(maxlen=10)  # 误差历史

        # 动作范围
        self.T_set_min = 18.0
        self.T_set_max = 28.0
        self.fan_min = 0.3
        self.fan_max = 1.0
    
    def get_action(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float
    ) -> np.ndarray:
        """
        PID控制逻辑
        
        控制策略：
        1. 计算温度误差 e = T_in - T_target
        2. PID输出：u = Kp*e + Ki*∫e + Kd*de/dt
        3. 根据u调整设定温度和风速
        """
        # ========== 1. 计算误差 ==========
        error = T_in - self.target_temp
        self.error_history.append(error)
        
        # ========== 2. PID计算 ==========
        # 比例项
        P = self.Kp * error

        # 积分项（带抗饱和）
        self.error_integral += error * self.dt  # 修复：积分项乘以dt
        self.error_integral = np.clip(self.error_integral, -10.0, 10.0)
        I = self.Ki * self.error_integral

        # 微分项（修复：除以dt进行归一化）
        if len(self.error_history) >= 2:
            D = self.Kd * (error - self.last_error) / self.dt
        else:
            D = 0.0

        # PID输出
        u = P + I + D
        
        # 更新状态
        self.last_error = error
        
        # ========== 3. 转换为控制动作 ==========
        # 设定温度：误差大时降低设定温度
        T_set_base = self.target_temp - 0.5 * u
        T_set_base = np.clip(T_set_base, self.T_set_min, self.T_set_max)
        
        # 风速：根据误差和负载调整
        fan_base = 0.5 + 0.1 * u + 0.0005 * IT_load
        fan_base = np.clip(fan_base, self.fan_min, self.fan_max)
        
        # ========== 4. 负载均衡策略 ==========
        # 所有CRAC使用相同设定（简化）
        T_set = np.ones(self.num_crac) * T_set_base
        fan_speed = np.ones(self.num_crac) * fan_base
        
        # 根据室外温度微调（利用自然冷却）
        if T_out < self.target_temp - 5:
            # 室外温度低，可以降低风速节能
            fan_speed *= 0.8
        
        # ========== 5. 归一化到[-1, 1] ==========
        action = self._normalize_action(T_set, fan_speed)
        
        return action
    
    def _normalize_action(
        self,
        T_set: np.ndarray,
        fan_speed: np.ndarray
    ) -> np.ndarray:
        """
        将物理动作归一化到[-1, 1]
        """
        # 温度归一化
        T_set_norm = 2.0 * (T_set - self.T_set_min) / (self.T_set_max - self.T_set_min) - 1.0
        
        # 风速归一化
        fan_norm = 2.0 * (fan_speed - self.fan_min) / (self.fan_max - self.fan_min) - 1.0
        
        # 交错排列：[T1, fan1, T2, fan2, ...]
        action = np.empty(2 * self.num_crac)
        action[0::2] = T_set_norm
        action[1::2] = fan_norm
        
        return action
    
    def reset(self):
        """重置控制器状态"""
        self.error_integral = 0.0
        self.last_error = 0.0
        self.error_history.clear()


class MPCController:
    """
    模型预测控制器（简化版）
    
    基于模型预测未来状态，优化控制序列
    这里使用简化的启发式规则模拟MPC行为
    """
    
    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        prediction_horizon: int = 6,  # 预测时域（步）
    ):
        self.num_crac = num_crac
        self.target_temp = target_temp
        self.horizon = prediction_horizon
        
        # 动作范围
        self.T_set_min = 18.0
        self.T_set_max = 28.0
        self.fan_min = 0.3
        self.fan_max = 1.0
        
        # 历史数据（用于模型辨识）
        self.T_history = deque(maxlen=20)
        self.load_history = deque(maxlen=20)
    
    def get_action(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float
    ) -> np.ndarray:
        """
        MPC控制逻辑（简化）
        
        策略：
        1. 预测未来温度趋势
        2. 提前调整控制动作
        3. 考虑能耗优化
        """
        # 更新历史
        self.T_history.append(T_in)
        self.load_history.append(IT_load)
        
        # ========== 1. 预测温度趋势 ==========
        if len(self.T_history) >= 3:
            # 简单线性预测
            dT_dt = (self.T_history[-1] - self.T_history[-3]) / 2.0
            T_predicted = T_in + dT_dt * self.horizon
        else:
            T_predicted = T_in
        
        # ========== 2. 计算控制动作 ==========
        error_current = T_in - self.target_temp
        error_predicted = T_predicted - self.target_temp
        
        # 预测性调整：如果预测温度会超标，提前加大制冷
        if error_predicted > 1.0:
            # 预测会过热，提前降温
            T_set = self.target_temp - 3.0
            fan_speed = 0.9
        elif error_predicted < -1.0:
            # 预测会过冷，减少制冷
            T_set = self.target_temp - 1.0
            fan_speed = 0.4
        else:
            # 正常控制
            T_set = self.target_temp - 2.0 - 0.5 * error_current
            fan_speed = 0.6 + 0.1 * error_current
        
        # ========== 3. 能耗优化 ==========
        # 利用室外温度优化
        if T_out < self.target_temp - 3:
            # 可以利用自然冷却
            fan_speed *= 0.7
            T_set += 1.0
        
        # 负载预测优化
        if len(self.load_history) >= 3:
            load_trend = self.load_history[-1] - self.load_history[-3]
            if load_trend > 50:
                # 负载上升趋势，提前增加制冷
                fan_speed += 0.1
        
        # ========== 4. 构造动作 ==========
        T_set = np.clip(T_set, self.T_set_min, self.T_set_max)
        fan_speed = np.clip(fan_speed, self.fan_min, self.fan_max)
        
        T_set_array = np.ones(self.num_crac) * T_set
        fan_array = np.ones(self.num_crac) * fan_speed
        
        # 归一化
        action = self._normalize_action(T_set_array, fan_array)
        
        return action
    
    def _normalize_action(self, T_set: np.ndarray, fan_speed: np.ndarray) -> np.ndarray:
        """归一化动作"""
        T_set_norm = 2.0 * (T_set - self.T_set_min) / (self.T_set_max - self.T_set_min) - 1.0
        fan_norm = 2.0 * (fan_speed - self.fan_min) / (self.fan_max - self.fan_min) - 1.0
        
        action = np.empty(2 * self.num_crac)
        action[0::2] = T_set_norm
        action[1::2] = fan_norm
        
        return action


class RuleBasedController:
    """
    基于规则的控制器
    
    使用简单的if-else规则
    适合作为baseline
    """
    
    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
    ):
        self.num_crac = num_crac
        self.target_temp = target_temp
        
        self.T_set_min = 18.0
        self.T_set_max = 28.0
        self.fan_min = 0.3
        self.fan_max = 1.0
    
    def get_action(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float
    ) -> np.ndarray:
        """
        基于规则的控制
        
        规则：
        1. 温度过高 → 降低设定温度，提高风速
        2. 温度过低 → 提高设定温度，降低风速
        3. 负载高 → 提高风速
        4. 室外温度低 → 利用自然冷却
        """
        error = T_in - self.target_temp
        
        # ========== 规则1：温度控制 ==========
        if error > 2.0:
            # 严重过热
            T_set = 20.0
            fan_speed = 1.0
        elif error > 1.0:
            # 轻微过热
            T_set = 21.0
            fan_speed = 0.8
        elif error > 0.5:
            # 接近目标
            T_set = 22.0
            fan_speed = 0.6
        elif error > -0.5:
            # 正常范围
            T_set = 23.0
            fan_speed = 0.5
        else:
            # 过冷
            T_set = 24.0
            fan_speed = 0.3
        
        # ========== 规则2：负载调整 ==========
        if IT_load > 300:
            fan_speed = min(fan_speed + 0.2, 1.0)
        elif IT_load < 150:
            fan_speed = max(fan_speed - 0.1, 0.3)
        
        # ========== 规则3：自然冷却 ==========
        if T_out < self.target_temp - 5:
            # 室外很冷，可以节能
            fan_speed *= 0.7
            T_set += 1.0
        
        # ========== 构造动作 ==========
        T_set = np.clip(T_set, self.T_set_min, self.T_set_max)
        fan_speed = np.clip(fan_speed, self.fan_min, self.fan_max)
        
        T_set_array = np.ones(self.num_crac) * T_set
        fan_array = np.ones(self.num_crac) * fan_speed
        
        # 归一化
        action = self._normalize_action(T_set_array, fan_array)
        
        return action
    
    def _normalize_action(self, T_set: np.ndarray, fan_speed: np.ndarray) -> np.ndarray:
        """归一化动作"""
        T_set_norm = 2.0 * (T_set - self.T_set_min) / (self.T_set_max - self.T_set_min) - 1.0
        fan_norm = 2.0 * (fan_speed - self.fan_min) / (self.fan_max - self.fan_min) - 1.0
        
        action = np.empty(2 * self.num_crac)
        action[0::2] = T_set_norm
        action[1::2] = fan_norm
        
        return action


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 测试PID控制器
    print("=" * 50)
    print("测试PID控制器")
    print("=" * 50)
    
    pid = PIDController(num_crac=4, target_temp=24.0)
    
    T_in = 26.0  # 初始温度过高
    for step in range(10):
        action = pid.get_action(T_in, 30.0, 50.0, 200.0)
        print(f"Step {step}: T_in={T_in:.2f}°C, Action={action[:2]}")
        
        # 模拟温度下降
        T_in -= 0.3
    
    print("\n" + "=" * 50)
    print("测试MPC控制器")
    print("=" * 50)
    
    mpc = MPCController(num_crac=4, target_temp=24.0)
    
    T_in = 24.0
    for step in range(10):
        action = mpc.get_action(T_in, 30.0, 50.0, 200.0 + step * 10)
        print(f"Step {step}: T_in={T_in:.2f}°C, Load={200+step*10}kW")
        
        # 模拟温度变化
        T_in += np.random.normal(0, 0.2)

