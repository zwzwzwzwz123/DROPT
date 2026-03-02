# ========================================
# 数据中心专家控制器
# ========================================
# 提供 PID / 简化 MPC / 规则基线，便于和强化学习策略对比

from __future__ import annotations

from collections import deque
from typing import Dict, Type

import numpy as np


class ExpertController:
    """
    专家控制器统一封装

    - 通过 controller_type 在 PID、MPC、规则控制之间切换
    - 输出动作已归一化到 [-1, 1]，可直接送入环境
    """

    _CONTROLLER_MAP: Dict[str, Type["_BaseController"]] = {}

    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        controller_type: str = "pid",
        **controller_kwargs,
    ) -> None:
        controller_type = controller_type.lower()
        if not self._CONTROLLER_MAP:
            # 延迟注册以避免类定义顺序影响
            self._CONTROLLER_MAP = {
                "pid": PIDController,
                "mpc": MPCController,
                "rule_based": RuleBasedController,
                "rule": RuleBasedController,
            }
        controller_cls = self._CONTROLLER_MAP.get(controller_type)
        if controller_cls is None:
            raise ValueError(f"未知的控制器类型: {controller_type}")
        self.controller = controller_cls(
            num_crac=num_crac,
            target_temp=target_temp,
            **controller_kwargs,
        )

    def get_action(self, T_in: float, T_out: float, H_in: float, IT_load: float) -> np.ndarray:
        """统一的专家动作接口"""
        return self.controller.get_action(T_in=T_in, T_out=T_out, H_in=H_in, IT_load=IT_load)

    def reset(self) -> None:
        """重置内部控制器状态"""
        reset_fn = getattr(self.controller, "reset", None)
        if callable(reset_fn):
            reset_fn()


class _BaseController:
    """控制器基类，仅约定接口，便于类型提示"""

    def __init__(self, num_crac: int, target_temp: float) -> None:
        self.num_crac = num_crac
        self.target_temp = target_temp
        self.T_set_min = 18.0
        self.T_set_max = 28.0
        self.fan_min = 0.3
        self.fan_max = 1.0

    def get_action(self, T_in: float, T_out: float, H_in: float, IT_load: float) -> np.ndarray:
        raise NotImplementedError

    def _normalize_action(self, T_set: np.ndarray, fan_speed: np.ndarray) -> np.ndarray:
        """将物理动作映射到 [-1, 1]"""
        T_norm = 2.0 * (T_set - self.T_set_min) / (self.T_set_max - self.T_set_min) - 1.0
        fan_norm = 2.0 * (fan_speed - self.fan_min) / (self.fan_max - self.fan_min) - 1.0
        action = np.empty(2 * self.num_crac, dtype=np.float32)
        action[0::2] = T_norm
        action[1::2] = fan_norm
        return action


class PIDController(_BaseController):
    """
    经典 PID 控制

    - 误差: e = T_in - target_temp
    - 输出: u = Kp * e + Ki * ∫e + Kd * de/dt
    - 控制律: 误差大 -> 降低设定温度 / 提高风速
    """

    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        Kp: float = 2.0,
        Ki: float = 0.05,
        Kd: float = 0.5,
        dt_minutes: float = 5.0,
    ) -> None:
        super().__init__(num_crac=num_crac, target_temp=target_temp)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = max(dt_minutes, 1e-3)
        self.error_integral = 0.0
        self.last_error = 0.0
        self.error_history: deque[float] = deque(maxlen=10)

    def get_action(self, T_in: float, T_out: float, H_in: float, IT_load: float) -> np.ndarray:
        error = T_in - self.target_temp
        self.error_history.append(error)
        self.error_integral += error * self.dt
        self.error_integral = float(np.clip(self.error_integral, -20.0, 20.0))
        derivative = (error - self.last_error) / self.dt if self.error_history else 0.0
        u = self.Kp * error + self.Ki * self.error_integral + self.Kd * derivative
        self.last_error = error

        T_set = np.clip(self.target_temp - 0.4 * u, self.T_set_min, self.T_set_max)
        fan = np.clip(0.55 + 0.12 * u + 0.0004 * IT_load, self.fan_min, self.fan_max)
        if T_out < self.target_temp - 5.0:
            fan *= 0.8  # 室外冷时顺势节能
        T_cmd = np.full(self.num_crac, T_set, dtype=np.float32)
        fan_cmd = np.full(self.num_crac, fan, dtype=np.float32)
        return self._normalize_action(T_cmd, fan_cmd)

    def reset(self) -> None:
        self.error_integral = 0.0
        self.last_error = 0.0
        self.error_history.clear()


class MPCController(_BaseController):
    """
    简化模型预测控制

    - 通过线性热模型预估未来温度
    - 穷举候选动作组合，选择预测代价最小的一组
    """

    def __init__(
        self,
        num_crac: int = 4,
        target_temp: float = 24.0,
        prediction_horizon: int = 6,
    ) -> None:
        super().__init__(num_crac=num_crac, target_temp=target_temp)
        self.horizon = max(prediction_horizon, 1)
        self.last_T_in = target_temp

        # 候选动作集合（温度设定、风速的粗网格）
        self._candidate_T = np.linspace(self.T_set_min, self.T_set_max, num=5)
        self._candidate_fan = np.linspace(self.fan_min, self.fan_max, num=5)

    def get_action(self, T_in: float, T_out: float, H_in: float, IT_load: float) -> np.ndarray:
        self.last_T_in = T_in
        best_cost = float("inf")
        best_pair = (self.target_temp, 0.6)
        for T_set in self._candidate_T:
            for fan in self._candidate_fan:
                cost = self._rollout_cost(
                    T_in=T_in,
                    T_out=T_out,
                    IT_load=IT_load,
                    T_set=T_set,
                    fan_speed=fan,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (T_set, fan)
        T_cmd = np.full(self.num_crac, best_pair[0], dtype=np.float32)
        fan_cmd = np.full(self.num_crac, best_pair[1], dtype=np.float32)
        return self._normalize_action(T_cmd, fan_cmd)

    def _rollout_cost(
        self,
        T_in: float,
        T_out: float,
        IT_load: float,
        T_set: float,
        fan_speed: float,
    ) -> float:
        """
        使用一阶简化热模型预测未来温度

        T_{t+1} = T_t + a*(T_out - T_t) + b*IT_load - c*(fan)*(target - T_set)
        代价 = 温度偏差平方 + 能耗 proxy
        """
        a = 0.08
        b = 0.0015
        c = 0.15
        temp = T_in
        cost = 0.0
        for _ in range(self.horizon):
            delta = a * (T_out - temp) + b * (IT_load - 200.0) - c * fan_speed * (self.target_temp - T_set)
            temp += delta
            temp_error = temp - self.target_temp
            energy_proxy = 0.5 * fan_speed + max(0.0, self.target_temp - T_set) * 0.05
            cost += temp_error ** 2 + energy_proxy
        return cost


class RuleBasedController(_BaseController):
    """
    手工规则控制

    - 按温度偏差切换离散档位
    - 根据负载和室外温度进行简单修正
    """

    def __init__(self, num_crac: int = 4, target_temp: float = 24.0) -> None:
        super().__init__(num_crac=num_crac, target_temp=target_temp)

    def get_action(self, T_in: float, T_out: float, H_in: float, IT_load: float) -> np.ndarray:
        error = T_in - self.target_temp
        if error > 2.0:
            T_set, fan = 20.0, 1.0
        elif error > 1.0:
            T_set, fan = 21.0, 0.85
        elif error > 0.5:
            T_set, fan = 22.0, 0.7
        elif error < -1.0:
            T_set, fan = 25.0, 0.4
        else:
            T_set, fan = 23.5, 0.55

        if IT_load > 320.0:
            fan = min(fan + 0.1, self.fan_max)
        elif IT_load < 150.0:
            fan = max(fan - 0.1, self.fan_min)

        if T_out < self.target_temp - 4.0:
            fan *= 0.75
            T_set += 0.5

        T_cmd = np.full(self.num_crac, np.clip(T_set, self.T_set_min, self.T_set_max), dtype=np.float32)
        fan_cmd = np.full(self.num_crac, np.clip(fan, self.fan_min, self.fan_max), dtype=np.float32)
        return self._normalize_action(T_cmd, fan_cmd)


if __name__ == "__main__":
    ctrl = ExpertController(controller_type="pid")
    action = ctrl.get_action(T_in=26.0, T_out=30.0, H_in=50.0, IT_load=250.0)
    print("PID action:", action[:4])
