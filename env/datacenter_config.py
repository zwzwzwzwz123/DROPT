# ========================================
# 数据中心配置文件
# ========================================
# 定义不同规模数据中心的参数配置

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataCenterConfig:
    """数据中心配置类"""
    
    # 基础参数
    name: str
    num_crac: int
    target_temp: float
    temp_tolerance: float
    
    # 物理参数
    room_volume: float      # 机房体积 (m³)
    air_density: float      # 空气密度 (kg/m³)
    air_cp: float           # 空气比热容 (kJ/(kg·K))
    wall_ua: float          # 墙体热传导系数 (kW/K)
    
    # CRAC参数
    crac_capacity: float    # 单个CRAC制冷容量 (kW)
    cop_nominal: float      # 名义能效比
    
    # IT负载参数
    it_load_min: float      # 最小IT负载 (kW)
    it_load_max: float      # 最大IT负载 (kW)
    
    # 奖励函数权重
    energy_weight: float
    temp_weight: float
    violation_penalty: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'num_crac_units': self.num_crac,
            'target_temp': self.target_temp,
            'temp_tolerance': self.temp_tolerance,
            'energy_weight': self.energy_weight,
            'temp_weight': self.temp_weight,
            'violation_penalty': self.violation_penalty,
        }


# ========== 预定义配置 ==========

# 小型数据中心（100kW级）
SMALL_DATACENTER = DataCenterConfig(
    name="small",
    num_crac=2,
    target_temp=24.0,
    temp_tolerance=2.0,
    room_volume=500.0,
    air_density=1.2,
    air_cp=1.005,
    wall_ua=30.0,
    crac_capacity=60.0,
    cop_nominal=3.0,
    it_load_min=30.0,
    it_load_max=100.0,
    energy_weight=1.0,
    temp_weight=10.0,
    violation_penalty=100.0,
)

# 中型数据中心（500kW级）
MEDIUM_DATACENTER = DataCenterConfig(
    name="medium",
    num_crac=4,
    target_temp=24.0,
    temp_tolerance=2.0,
    room_volume=1000.0,
    air_density=1.2,
    air_cp=1.005,
    wall_ua=50.0,
    crac_capacity=100.0,
    cop_nominal=3.0,
    it_load_min=100.0,
    it_load_max=500.0,
    energy_weight=1.0,
    temp_weight=10.0,
    violation_penalty=100.0,
)

# 大型数据中心（2MW级）
LARGE_DATACENTER = DataCenterConfig(
    name="large",
    num_crac=8,
    target_temp=24.0,
    temp_tolerance=1.5,
    room_volume=3000.0,
    air_density=1.2,
    air_cp=1.005,
    wall_ua=100.0,
    crac_capacity=200.0,
    cop_nominal=3.5,
    it_load_min=500.0,
    it_load_max=2000.0,
    energy_weight=1.0,
    temp_weight=15.0,
    violation_penalty=200.0,
)


def get_config(name: str = "medium") -> DataCenterConfig:
    """
    获取预定义配置
    
    参数：
    - name: 配置名称 ('small', 'medium', 'large')
    
    返回：
    - config: 数据中心配置对象
    """
    configs = {
        'small': SMALL_DATACENTER,
        'medium': MEDIUM_DATACENTER,
        'large': LARGE_DATACENTER,
    }
    
    if name not in configs:
        raise ValueError(f"未知配置: {name}. 可选: {list(configs.keys())}")
    
    return configs[name]


# ========== 训练超参数推荐 ==========

TRAINING_CONFIGS = {
    # 行为克隆模式（快速训练）
    'bc_fast': {
        'bc_coef': True,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 256,
        'n_timesteps': 5,
        'epoch': 50000,
        'expert_type': 'pid',
    },
    
    # 策略梯度模式（高性能）
    'pg_high_performance': {
        'bc_coef': False,
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'batch_size': 512,
        'n_timesteps': 8,
        'epoch': 200000,
        'gamma': 0.99,
        'prioritized_replay': True,
    },
    
    # 混合模式（先BC后PG）
    'hybrid': {
        'bc_coef': True,  # 第一阶段
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'batch_size': 256,
        'n_timesteps': 6,
        'epoch': 30000,
        'expert_type': 'mpc',
        # 第二阶段切换到bc_coef=False
    },
}


def get_training_config(name: str = 'bc_fast') -> Dict[str, Any]:
    """
    获取训练配置
    
    参数：
    - name: 配置名称
    
    返回：
    - config: 训练参数字典
    """
    if name not in TRAINING_CONFIGS:
        raise ValueError(f"未知训练配置: {name}. 可选: {list(TRAINING_CONFIGS.keys())}")
    
    return TRAINING_CONFIGS[name]

