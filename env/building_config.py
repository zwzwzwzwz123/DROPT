#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEAR 建筑环境配置常量

统一管理 BEAR 集成相关的配置参数，避免硬编码重复
"""

# ========== 奖励函数配置 ==========
# 奖励缩放说明:
# - 0.1: 原始配置,奖励仍较大 (约-6400)
# - 0.01: 推荐配置,奖励适中 (约-640) ⭐
# - 0.001: 激进配置,奖励较小 (约-64)
# 根据当前训练,Q值约-7,但原始奖励-64000,建议增加缩放到0.01
DEFAULT_REWARD_SCALE = 0.01  # 奖励缩放因子（将奖励缩小100倍，稳定训练）
DEFAULT_ENERGY_WEIGHT = 0.001  # 能耗权重 α
DEFAULT_TEMP_WEIGHT = 0.999  # 温度偏差权重 β
DEFAULT_VIOLATION_PENALTY = 100.0  # 温度越界惩罚系数 γ

# ========== 环境配置 ==========
DEFAULT_BUILDING_TYPE = 'OfficeSmall'  # 默认建筑类型
DEFAULT_WEATHER_TYPE = 'Hot_Dry'  # 默认气候类型
DEFAULT_LOCATION = 'Tucson'  # 默认地理位置
DEFAULT_TARGET_TEMP = 22.0  # 默认目标温度 (°C)
DEFAULT_TEMP_TOLERANCE = 2.0  # 默认温度容差 (°C)
DEFAULT_MAX_POWER = 8000  # 默认 HVAC 最大功率 (W)
DEFAULT_TIME_RESOLUTION = 3600  # 默认时间分辨率 (秒，1小时)

# ========== 专家控制器配置 ==========
# PID 控制器默认参数
DEFAULT_PID_KP = 0.5  # 比例系数
DEFAULT_PID_KI = 0.01  # 积分系数
DEFAULT_PID_KD = 0.1  # 微分系数
DEFAULT_PID_INTEGRAL_LIMIT = 10.0  # 积分项限制

# MPC 控制器默认参数
DEFAULT_MPC_SAFETY_MARGIN = 0.9  # 安全裕度
DEFAULT_MPC_PLANNING_STEPS = 1  # 规划步数

# 规则控制器默认参数
DEFAULT_RULE_COOLING_POWER = 0.8  # 制冷功率
DEFAULT_RULE_HEATING_POWER = 0.8  # 制热功率

# ========== 训练配置 ==========
DEFAULT_TRAINING_NUM = 4  # 默认并行训练环境数量
DEFAULT_TEST_NUM = 2  # 默认并行测试环境数量
DEFAULT_BUFFER_SIZE = 1000000  # 默认经验回放缓冲区大小
DEFAULT_BATCH_SIZE = 256  # 默认批次大小
DEFAULT_GAMMA = 0.99  # 默认折扣因子
DEFAULT_N_STEP = 3  # 默认 N 步 TD 学习

# ========== 扩散模型配置 ==========
# 扩散步数说明:
# - 5步: 快速但质量较低 (原始配置)
# - 10步: 平衡质量与速度 (推荐配置) ⭐
# - 15步: 高质量但较慢
# - 20+步: 最佳质量但不适合实时控制
# 根据DDIM研究,10步的生成质量是5步的2-3倍,推理时间仅增加1倍
DEFAULT_DIFFUSION_STEPS = 20  # 默认扩散步数 (从5增加到10以提升生成质量)
DEFAULT_BETA_SCHEDULE = 'vp'  # 默认噪声调度类型
DEFAULT_HIDDEN_DIM = 256  # 默认 MLP 隐藏层维度
DEFAULT_ACTOR_LR = 3e-4  # 默认 Actor 学习率
# Critic学习率说明:
# - 3e-4: 原始配置,梯度较大时可能不稳定
# - 1e-4: 推荐配置,更稳定 ⭐
# - 3e-5: 保守配置,收敛较慢
DEFAULT_CRITIC_LR = 1e-4  # 默认 Critic 学习率 (从3e-4降低以稳定训练)
DEFAULT_EXPLORATION_NOISE = 0.1  # 默认探索噪声标准差

# ========== 日志配置 ==========
DEFAULT_LOG_DIR = 'log_building'  # 默认日志目录
DEFAULT_SAVE_INTERVAL = 1000  # 默认模型保存间隔（轮次）

# ========== 状态维度计算公式 ==========
def calculate_state_dim(roomnum: int) -> int:
    """
    计算 BEAR 环境的状态维度
    
    状态组成：
    - 房间温度: roomnum
    - 室外温度: 1
    - GHI (全局水平辐照度): roomnum
    - 地面温度: 1
    - 人员热负荷: roomnum
    
    总维度 = roomnum + 1 + roomnum + 1 + roomnum = 3*roomnum + 2
    
    参数：
    - roomnum: 房间数量
    
    返回：
    - state_dim: 状态维度
    """
    return 3 * roomnum + 2


# ========== 配置验证函数 ==========
def validate_reward_weights(energy_weight: float, temp_weight: float) -> None:
    """
    验证奖励权重配置的合理性
    
    参数：
    - energy_weight: 能耗权重
    - temp_weight: 温度偏差权重
    
    抛出：
    - ValueError: 如果权重配置不合理
    """
    if energy_weight < 0 or temp_weight < 0:
        raise ValueError(f"奖励权重必须非负: energy_weight={energy_weight}, temp_weight={temp_weight}")
    
    if energy_weight + temp_weight == 0:
        raise ValueError("能耗权重和温度权重不能同时为零")
    
    # 警告：如果权重比例极端
    if energy_weight > 0 and temp_weight > 0:
        ratio = max(energy_weight, temp_weight) / min(energy_weight, temp_weight)
        if ratio > 10000:
            import warnings
            warnings.warn(
                f"奖励权重比例过大 ({ratio:.0f}:1)，可能导致训练不稳定。"
                f"建议调整 energy_weight={energy_weight} 和 temp_weight={temp_weight}",
                UserWarning
            )


def validate_temperature_config(target_temp: float, temp_tolerance: float) -> None:
    """
    验证温度配置的合理性
    
    参数：
    - target_temp: 目标温度 (°C)
    - temp_tolerance: 温度容差 (°C)
    
    抛出：
    - ValueError: 如果温度配置不合理
    """
    if not (-40 <= target_temp <= 40):
        raise ValueError(f"目标温度超出合理范围 [-40, 40]°C: {target_temp}°C")
    
    if temp_tolerance <= 0:
        raise ValueError(f"温度容差必须为正数: {temp_tolerance}°C")
    
    if temp_tolerance > 10:
        import warnings
        warnings.warn(
            f"温度容差过大 ({temp_tolerance}°C)，可能导致舒适度不足",
            UserWarning
        )


# ========== 配置字典（用于快速访问） ==========
DEFAULT_CONFIG = {
    # 奖励函数
    'reward_scale': DEFAULT_REWARD_SCALE,
    'energy_weight': DEFAULT_ENERGY_WEIGHT,
    'temp_weight': DEFAULT_TEMP_WEIGHT,
    'violation_penalty': DEFAULT_VIOLATION_PENALTY,
    
    # 环境
    'building_type': DEFAULT_BUILDING_TYPE,
    'weather_type': DEFAULT_WEATHER_TYPE,
    'location': DEFAULT_LOCATION,
    'target_temp': DEFAULT_TARGET_TEMP,
    'temp_tolerance': DEFAULT_TEMP_TOLERANCE,
    'max_power': DEFAULT_MAX_POWER,
    'time_resolution': DEFAULT_TIME_RESOLUTION,
    
    # 训练
    'training_num': DEFAULT_TRAINING_NUM,
    'test_num': DEFAULT_TEST_NUM,
    'buffer_size': DEFAULT_BUFFER_SIZE,
    'batch_size': DEFAULT_BATCH_SIZE,
    'gamma': DEFAULT_GAMMA,
    'n_step': DEFAULT_N_STEP,
    
    # 扩散模型
    'diffusion_steps': DEFAULT_DIFFUSION_STEPS,
    'beta_schedule': DEFAULT_BETA_SCHEDULE,
    'hidden_dim': DEFAULT_HIDDEN_DIM,
    'actor_lr': DEFAULT_ACTOR_LR,
    'critic_lr': DEFAULT_CRITIC_LR,
    'exploration_noise': DEFAULT_EXPLORATION_NOISE,
    
    # 日志
    'log_dir': DEFAULT_LOG_DIR,
    'save_interval': DEFAULT_SAVE_INTERVAL,
}


if __name__ == '__main__':
    # 测试配置
    print("=" * 60)
    print("BEAR 建筑环境默认配置")
    print("=" * 60)
    
    print("\n奖励函数配置:")
    print(f"  奖励缩放因子: {DEFAULT_REWARD_SCALE}")
    print(f"  能耗权重: {DEFAULT_ENERGY_WEIGHT}")
    print(f"  温度偏差权重: {DEFAULT_TEMP_WEIGHT}")
    print(f"  权重比例: {DEFAULT_TEMP_WEIGHT/DEFAULT_ENERGY_WEIGHT:.0f}:1 (舒适度优先)")
    
    print("\n环境配置:")
    print(f"  建筑类型: {DEFAULT_BUILDING_TYPE}")
    print(f"  气候类型: {DEFAULT_WEATHER_TYPE}")
    print(f"  目标温度: {DEFAULT_TARGET_TEMP}°C ± {DEFAULT_TEMP_TOLERANCE}°C")
    
    print("\n训练配置:")
    print(f"  并行训练环境: {DEFAULT_TRAINING_NUM}")
    print(f"  并行测试环境: {DEFAULT_TEST_NUM}")
    print(f"  批次大小: {DEFAULT_BATCH_SIZE}")
    
    print("\n扩散模型配置:")
    print(f"  扩散步数: {DEFAULT_DIFFUSION_STEPS}")
    print(f"  噪声调度: {DEFAULT_BETA_SCHEDULE}")
    print(f"  隐藏层维度: {DEFAULT_HIDDEN_DIM}")
    
    # 测试验证函数
    print("\n配置验证测试:")
    try:
        validate_reward_weights(DEFAULT_ENERGY_WEIGHT, DEFAULT_TEMP_WEIGHT)
        print("  ✓ 奖励权重验证通过")
    except ValueError as e:
        print(f"  ✗ 奖励权重验证失败: {e}")
    
    try:
        validate_temperature_config(DEFAULT_TARGET_TEMP, DEFAULT_TEMP_TOLERANCE)
        print("  ✓ 温度配置验证通过")
    except ValueError as e:
        print(f"  ✗ 温度配置验证失败: {e}")
    
    print("\n" + "=" * 60)

