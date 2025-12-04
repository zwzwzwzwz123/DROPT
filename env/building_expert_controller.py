#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEAR 建筑环境专家控制器

提供三种专家控制器：
1. BearMPCWrapper: 包装 BEAR 内置的 MPC 控制器
2. BearPIDController: PID 控制器
3. BearRuleBasedController: 基于规则的控制器
"""

import sys
import os
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod

# 添加 BEAR 路径
bear_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bear')
if bear_path not in sys.path:
    sys.path.insert(0, bear_path)

from BEAR.Controller.MPC_Controller import MPCAgent


class BaseBearController(ABC):
    """
    BEAR 控制器基类
    
    所有专家控制器都应继承此类并实现 get_action 方法
    """
    
    def __init__(self, env):
        """
        初始化控制器
        
        Args:
            env: BearEnvWrapper 环境实例
        """
        self.env = env
        self.bear_env = env.bear_env
        self.roomnum = env.roomnum
        self.target_temp = env.target_temp
        self.temp_tolerance = env.temp_tolerance
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        根据当前状态获取控制动作
        
        Args:
            state: 当前状态 (state_dim,)
        
        Returns:
            action: 控制动作 (action_dim,)
        """
        pass
    
    def reset(self):
        """重置控制器（如果需要）"""
        pass


class BearMPCWrapper(BaseBearController):
    """
    BEAR MPC 控制器包装器
    
    包装 BEAR 内置的模型预测控制（MPC）控制器
    使用凸优化求解最优控制问题
    """
    
    def __init__(
        self,
        env,
        gamma: Optional[Tuple[float, float]] = None,
        safety_margin: float = 0.9,
        planning_steps: int = 3
    ):
        """
        初始化 MPC 控制器
        
        Args:
            env: BearEnvWrapper 环境实例
            gamma: 奖励权重 (能耗权重, 温度偏差权重)，默认使用环境的权重
            safety_margin: 安全裕度
            planning_steps: 规划步数（预测时域）
        """
        super().__init__(env)
        
        # 使用环境的权重或自定义权重
        if gamma is None:
            gamma = (env.energy_weight, env.temp_weight)
        
        # 创建 BEAR 的 MPC 控制器
        self.mpc_agent = MPCAgent(
            environment=self.bear_env,
            gamma=gamma,
            safety_margin=safety_margin,
            planning_steps=planning_steps
        )
        
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.planning_steps = planning_steps

        # 修复：保存上一步动作作为备用
        self.last_action = None
        self.failure_count = 0

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        使用 MPC 获取最优控制动作

        Args:
            state: 当前状态

        Returns:
            action: MPC 计算的最优动作

        修复：MPC 求解失败时使用备用策略而非零动作
        """
        try:
            # 调用 BEAR 的 MPC 控制器
            action, predicted_state = self.mpc_agent.predict(self.bear_env)

            # 确保动作在有效范围内
            action = np.clip(action, -1.0, 1.0)

            # 保存成功的动作
            self.last_action = action.copy()
            self.failure_count = 0

            return action.astype(np.float32)

        except Exception as e:
            # 修复：MPC 求解失败时使用备用策略
            self.failure_count += 1
            print(f"警告: MPC 求解失败 ({e})，使用备用策略 (失败次数: {self.failure_count})")

            # 备用策略1：使用上一步动作
            if self.last_action is not None:
                return self.last_action.astype(np.float32)

            # 备用策略2：使用保守的默认动作（轻微加热）
            # 假设负值表示加热，正值表示制冷
            default_action = np.full(self.roomnum, -0.2, dtype=np.float32)
            return default_action
    
    def reset(self):
        """重置 MPC 控制器"""
        # MPC 是无状态的，每次都重新求解，不需要重置
        # 修复：重置备用动作和失败计数
        self.last_action = None
        self.failure_count = 0


class BearPIDController(BaseBearController):
    """
    PID 控制器
    
    为每个房间实现独立的 PID 控制
    控制目标：将房间温度维持在目标温度附近
    """
    
    def __init__(
        self,
        env,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.1,
        integral_limit: float = 10.0,
        deadband: Optional[float] = None
    ):
        """
        初始化 PID 控制器
        
        Args:
            env: BearEnvWrapper 环境实例
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            integral_limit: 积分项限制（防止积分饱和）
        """
        super().__init__(env)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        # 为每个房间维护 PID 状态
        self.integral = np.zeros(self.roomnum)  # 积分项
        self.last_error = np.zeros(self.roomnum)  # 上一次误差
        self.first_step = True  # 是否是第一步
        self.deadband = deadband if deadband is not None else env.temp_tolerance
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        使用 PID 计算控制动作
        
        Args:
            state: 当前状态
        
        Returns:
            action: PID 计算的控制动作
        """
        # 提取房间温度（状态的前 roomnum 个元素）
        zone_temps = state[:self.roomnum]
        
        # 计算温度误差（当前温度 - 目标温度）
        error = zone_temps - self.target_temp
        if self.deadband is not None and self.deadband > 0:
            mask = np.abs(error) <= self.deadband
            if np.any(mask):
                error = error.copy()
                error[mask] = 0.0
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项（累积误差）
        self.integral += error
        # 限制积分项防止饱和
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # 微分项（误差变化率）
        if self.first_step:
            d_term = np.zeros(self.roomnum)
            self.first_step = False
        else:
            d_term = self.kd * (error - self.last_error)
        
        # 更新上一次误差
        self.last_error = error.copy()
        
        # PID 输出：负值表示制冷，正值表示制热
        # 温度高于目标 -> error > 0 -> 需要制冷（负动作）
        action = -(p_term + i_term + d_term)
        
        # 限制动作范围 [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)
    
    def reset(self):
        """重置 PID 控制器状态"""
        self.integral = np.zeros(self.roomnum)
        self.last_error = np.zeros(self.roomnum)
        self.first_step = True


class BearRuleBasedController(BaseBearController):
    """
    基于规则的控制器
    
    使用简单的 if-else 规则进行控制：
    - 温度过高 -> 制冷
    - 温度过低 -> 制热
    - 温度适中 -> 不动作
    """
    
    def __init__(
        self,
        env,
        cooling_power: float = 0.8,
        heating_power: float = 0.8,
        deadband: Optional[float] = None
    ):
        """
        初始化规则控制器
        
        Args:
            env: BearEnvWrapper 环境实例
            cooling_power: 制冷功率（0-1）
            heating_power: 制热功率（0-1）
            deadband: 死区范围，默认使用环境的 temp_tolerance
        """
        super().__init__(env)
        
        self.cooling_power = cooling_power
        self.heating_power = heating_power
        self.deadband = deadband if deadband is not None else env.temp_tolerance
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        使用规则计算控制动作
        
        Args:
            state: 当前状态
        
        Returns:
            action: 规则计算的控制动作
        """
        # 提取房间温度
        zone_temps = state[:self.roomnum]
        
        # 初始化动作
        action = np.zeros(self.roomnum)
        
        # 对每个房间应用规则
        for i in range(self.roomnum):
            temp = zone_temps[i]
            
            if temp > self.target_temp + self.deadband:
                # 温度过高 -> 制冷（负动作）
                action[i] = -self.cooling_power
            
            elif temp < self.target_temp - self.deadband:
                # 温度过低 -> 制热（正动作）
                action[i] = self.heating_power
            
            else:
                # 温度在死区内 -> 不动作
                action[i] = 0.0
        
        return action.astype(np.float32)
    
    def reset(self):
        """重置规则控制器（无状态，不需要重置）"""
        pass


class BearBangBangController(BaseBearController):
    """
    Bang-Bang 控制器（开关控制）
    
    最简单的控制策略：
    - 温度高于目标 -> 全功率制冷
    - 温度低于目标 -> 全功率制热
    """
    
    def __init__(self, env):
        """
        初始化 Bang-Bang 控制器
        
        Args:
            env: BearEnvWrapper 环境实例
        """
        super().__init__(env)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        使用 Bang-Bang 策略计算控制动作
        
        Args:
            state: 当前状态
        
        Returns:
            action: Bang-Bang 控制动作
        """
        # 提取房间温度
        zone_temps = state[:self.roomnum]
        
        # 初始化动作
        action = np.zeros(self.roomnum)
        
        # 对每个房间应用 Bang-Bang 控制
        for i in range(self.roomnum):
            if zone_temps[i] > self.target_temp:
                # 温度高 -> 全功率制冷
                action[i] = -1.0
            else:
                # 温度低 -> 全功率制热
                action[i] = 1.0
        
        return action.astype(np.float32)
    
    def reset(self):
        """重置 Bang-Bang 控制器（无状态，不需要重置）"""
        pass


# 控制器工厂函数
def create_expert_controller(
    controller_type: str,
    env,
    **kwargs
) -> BaseBearController:
    """
    创建专家控制器
    
    Args:
        controller_type: 控制器类型 ('mpc', 'pid', 'rule', 'bangbang')
        env: BearEnvWrapper 环境实例
        **kwargs: 控制器特定参数
    
    Returns:
        controller: 专家控制器实例
    
    Examples:
        >>> # 创建 MPC 控制器
        >>> mpc = create_expert_controller('mpc', env, planning_steps=3)
        >>> 
        >>> # 创建 PID 控制器
        >>> pid = create_expert_controller('pid', env, kp=0.5, ki=0.01, kd=0.1)
        >>> 
        >>> # 创建规则控制器
        >>> rule = create_expert_controller('rule', env, cooling_power=0.8)
    """
    controller_type = controller_type.lower()
    
    if controller_type == 'mpc':
        return BearMPCWrapper(env, **kwargs)
    
    elif controller_type == 'pid':
        return BearPIDController(env, **kwargs)
    
    elif controller_type in ['rule', 'rulebased']:
        return BearRuleBasedController(env, **kwargs)
    
    elif controller_type == 'bangbang':
        return BearBangBangController(env, **kwargs)
    
    else:
        raise ValueError(
            f"未知的控制器类型: {controller_type}. "
            f"支持的类型: 'mpc', 'pid', 'rule', 'bangbang'"
        )

