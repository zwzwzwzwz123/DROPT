# ========================================
# 数据中心热力学模型
# ========================================
# 模拟数据中心的温度动态和能耗计算
# 基于简化的热平衡方程

import numpy as np
from typing import Tuple


class ThermalModel:
    """
    数据中心热力学模型
    
    基于能量守恒和热传递原理：
    dT/dt = (Q_IT - Q_cooling - Q_loss) / (m * c_p)
    
    其中：
    - Q_IT: IT设备发热功率
    - Q_cooling: 空调制冷功率
    - Q_loss: 通过墙体等散失的热量
    - m: 空气质量
    - c_p: 空气比热容
    """
    
    def __init__(
        self,
        num_crac: int = 4,              # CRAC单元数量
        time_step: float = 5.0/60.0,    # 时间步长（小时）
        room_volume: float = 1000.0,    # 机房体积 (m³)
        air_density: float = 1.2,       # 空气密度 (kg/m³)
        air_cp: float = 1.005,          # 空气比热容 (kJ/(kg·K))
        wall_ua: float = 50.0,          # 墙体热传导系数 (kW/K)
        crac_capacity: float = 100.0,   # 单个CRAC制冷容量 (kW)
        cop_nominal: float = 3.0,       # 名义能效比 (COP)
    ):
        """
        初始化热力学模型
        
        参数：
        - num_crac: CRAC空调数量
        - time_step: 仿真时间步长（小时）
        - room_volume: 机房体积（立方米）
        - air_density: 空气密度（kg/m³）
        - air_cp: 空气比热容（kJ/(kg·K)）
        - wall_ua: 墙体热传导系数（kW/K）
        - crac_capacity: 单个CRAC制冷容量（kW）
        - cop_nominal: 名义能效比
        """
        self.num_crac = num_crac
        self.dt = time_step
        
        # ========== 热容参数 ==========
        self.m_air = room_volume * air_density  # 空气质量 (kg)
        self.c_p = air_cp                       # 比热容 (kJ/(kg·K))
        self.thermal_mass = self.m_air * self.c_p  # 热容 (kJ/K)
        
        # ========== 热传递参数 ==========
        self.UA_wall = wall_ua  # 墙体热传导 (kW/K)
        
        # ========== CRAC参数 ==========
        self.Q_crac_max = crac_capacity  # 单个CRAC最大制冷量 (kW)
        self.COP_nominal = cop_nominal   # 名义COP
        
        # ========== 效率模型参数 ==========
        # COP随负载率和温差变化：COP = COP_nominal * f(load_ratio, ΔT)
        self.cop_load_coef = [0.2, 0.5, 1.0, 0.9]  # 负载率[0.3, 0.5, 1.0, 1.2]对应的COP系数
        self.cop_temp_coef = 0.02  # 温差每增加1K，COP下降2%
    
    def step(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float,
        T_set: np.ndarray,
        fan_speed: np.ndarray
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        执行一步热力学仿真
        
        参数：
        - T_in: 当前机房温度 (°C)
        - T_out: 室外温度 (°C)
        - H_in: 当前机房湿度 (%)
        - IT_load: IT设备负载 (kW)
        - T_set: 各CRAC设定温度数组 (°C)
        - fan_speed: 各CRAC风速比例数组 [0.3-1.0]
        
        返回：
        - next_T_in: 下一时刻机房温度 (°C)
        - next_H_in: 下一时刻机房湿度 (%)
        - next_T_supply: 各CRAC供风温度数组 (°C)
        - energy_consumed: 本步能耗 (kWh)
        """
        # ========== 1. 计算IT设备发热 ==========
        Q_IT = IT_load  # IT设备发热功率 (kW)
        
        # ========== 2. 计算墙体散热 ==========
        # Q_loss = UA * (T_in - T_out)
        Q_loss = self.UA_wall * (T_in - T_out)  # 正值表示向外散热
        
        # ========== 3. 计算CRAC制冷量 ==========
        Q_cooling_total = 0.0  # 总制冷量 (kW)
        P_crac_total = 0.0     # 总电功率 (kW)
        T_supply_list = []     # 各CRAC供风温度
        
        for i in range(self.num_crac):
            # 计算单个CRAC的制冷量和能耗
            Q_cool, P_elec, T_sup = self._crac_model(
                T_in=T_in,
                T_out=T_out,
                T_set=T_set[i],
                fan_speed=fan_speed[i]
            )
            Q_cooling_total += Q_cool
            P_crac_total += P_elec
            T_supply_list.append(T_sup)
        
        # ========== 4. 热平衡方程 ==========
        # dT/dt = (Q_IT - Q_cooling - Q_loss) / (m * c_p)
        Q_net = Q_IT - Q_cooling_total - Q_loss  # 净热量 (kW)
        dT_dt = Q_net / self.thermal_mass  # 温度变化率 (K/h)
        
        # 欧拉法更新温度
        next_T_in = T_in + dT_dt * self.dt
        
        # 温度物理约束（不会瞬间变化太大）
        next_T_in = np.clip(next_T_in, T_in - 2.0, T_in + 2.0)
        
        # ========== 5. 湿度模型（简化） ==========
        # 假设湿度缓慢变化，受空调除湿影响
        dH_dt = -0.5 * np.mean(fan_speed) + np.random.normal(0, 0.1)
        next_H_in = H_in + dH_dt * self.dt
        next_H_in = np.clip(next_H_in, 30.0, 70.0)
        
        # ========== 6. 计算总能耗 ==========
        # 能耗 = CRAC电功率 * 时间步长
        energy_consumed = P_crac_total * self.dt  # kWh
        
        return next_T_in, next_H_in, np.array(T_supply_list), energy_consumed
    
    def _crac_model(
        self,
        T_in: float,
        T_out: float,
        T_set: float,
        fan_speed: float
    ) -> Tuple[float, float, float]:
        """
        单个CRAC单元模型
        
        参数：
        - T_in: 机房温度 (°C)
        - T_out: 室外温度 (°C)
        - T_set: 设定温度 (°C)
        - fan_speed: 风速比例 [0.3-1.0]
        
        返回：
        - Q_cooling: 制冷量 (kW)
        - P_electric: 电功率 (kW)
        - T_supply: 供风温度 (°C)
        """
        # ========== 1. 计算需求制冷量 ==========
        # 根据温差和风速计算
        delta_T = T_in - T_set  # 温差
        
        if delta_T > 0:
            # 需要制冷
            # 制冷量与温差和风速成正比
            Q_demand = self.Q_crac_max * fan_speed * min(delta_T / 5.0, 1.0)
        else:
            # 不需要制冷（或需要加热，这里简化为0）
            Q_demand = 0.0
        
        # 限制在CRAC容量范围内
        Q_cooling = np.clip(Q_demand, 0.0, self.Q_crac_max * fan_speed)
        
        # ========== 2. 计算COP（能效比） ==========
        # COP受负载率和温差影响
        load_ratio = Q_cooling / (self.Q_crac_max + 1e-6)
        
        # 负载率对COP的影响（部分负载效率下降）
        if load_ratio < 0.3:
            cop_load_factor = 0.6
        elif load_ratio < 0.5:
            cop_load_factor = 0.8
        elif load_ratio < 1.0:
            cop_load_factor = 1.0
        else:
            cop_load_factor = 0.95
        
        # 温差对COP的影响（温差越大，效率越低）
        delta_T_cond = T_out - T_in  # 冷凝器温差
        cop_temp_factor = 1.0 - self.cop_temp_coef * max(delta_T_cond, 0)
        cop_temp_factor = np.clip(cop_temp_factor, 0.5, 1.2)
        
        # 实际COP
        COP_actual = self.COP_nominal * cop_load_factor * cop_temp_factor
        COP_actual = np.clip(COP_actual, 1.5, 5.0)  # 物理约束
        
        # ========== 3. 计算电功率 ==========
        # P = Q / COP
        if Q_cooling > 0:
            P_electric = Q_cooling / COP_actual
            # 加上风机功率（简化为制冷功率的10%）
            P_fan = 0.1 * self.Q_crac_max * fan_speed
            P_electric += P_fan
        else:
            # 待机功率
            P_electric = 0.05 * self.Q_crac_max
        
        # ========== 4. 计算供风温度 ==========
        # T_supply = T_in - ΔT_supply
        # ΔT_supply取决于制冷量和风量
        if Q_cooling > 0:
            # 假设风量与风速成正比
            air_flow = fan_speed * 10.0  # m³/s（简化）
            # ΔT = Q / (ρ * V * c_p)
            delta_T_supply = Q_cooling / (1.2 * air_flow * 1.005 + 1e-6)
            T_supply = T_in - delta_T_supply
        else:
            T_supply = T_in
        
        # 供风温度物理约束
        T_supply = np.clip(T_supply, 15.0, 30.0)
        
        return Q_cooling, P_electric, T_supply
    
    def get_steady_state(
        self,
        IT_load: float,
        T_out: float,
        T_set: float
    ) -> Tuple[float, float]:
        """
        计算稳态温度和能耗
        
        用于验证模型或初始化
        
        参数：
        - IT_load: IT负载 (kW)
        - T_out: 室外温度 (°C)
        - T_set: 设定温度 (°C)
        
        返回：
        - T_steady: 稳态机房温度 (°C)
        - P_steady: 稳态功率 (kW)
        """
        # 稳态时：Q_IT = Q_cooling + Q_loss
        # 假设所有CRAC以相同设定运行
        
        # 迭代求解稳态温度
        T_in = T_set
        for _ in range(10):
            Q_loss = self.UA_wall * (T_in - T_out)
            Q_cooling_needed = IT_load - Q_loss
            
            # 假设COP=3.0，风速=0.7
            P_total = Q_cooling_needed / 3.0
            
            # 更新T_in估计
            T_in = T_set + Q_cooling_needed / (self.num_crac * self.Q_crac_max) * 2.0
        
        return T_in, P_total


class SimplifiedThermalModel:
    """
    简化热力学模型（用于快速原型）
    
    使用一阶惯性环节近似：
    T(k+1) = a * T(k) + b * T_set + c * Q_IT + d * T_out
    """
    
    def __init__(
        self,
        num_crac: int = 4,
        time_step: float = 5.0/60.0,
        time_constant: float = 0.5,  # 时间常数（小时）
    ):
        self.num_crac = num_crac
        self.dt = time_step
        self.tau = time_constant
        
        # 一阶惯性系数
        self.a = np.exp(-self.dt / self.tau)  # 温度衰减系数
        self.b = 0.3 * (1 - self.a)           # 设定温度影响
        self.c = 0.01 * (1 - self.a)          # IT负载影响
        self.d = 0.1 * (1 - self.a)           # 室外温度影响
    
    def step(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float,
        T_set: np.ndarray,
        fan_speed: np.ndarray
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        简化的一步仿真
        """
        # 平均设定温度
        T_set_avg = np.mean(T_set)
        fan_avg = np.mean(fan_speed)
        
        # 一阶惯性更新
        next_T_in = (
            self.a * T_in +
            self.b * T_set_avg +
            self.c * IT_load +
            self.d * T_out +
            np.random.normal(0, 0.1)  # 噪声
        )
        
        # 湿度简化模型
        next_H_in = H_in + np.random.normal(0, 0.5)
        next_H_in = np.clip(next_H_in, 30.0, 70.0)
        
        # 供风温度
        next_T_supply = T_set - 2.0 * fan_speed
        
        # 能耗简化模型
        Q_cooling = max(0, IT_load - 50.0 * (T_in - T_out))
        COP = 3.0
        energy_consumed = (Q_cooling / COP) * self.dt * self.num_crac * fan_avg
        
        return next_T_in, next_H_in, next_T_supply, energy_consumed


# ========== 测试代码 ==========
if __name__ == '__main__':
    # 测试热力学模型
    model = ThermalModel(num_crac=4)
    
    T_in = 24.0
    T_out = 30.0
    H_in = 50.0
    IT_load = 200.0
    T_set = np.array([22.0, 22.0, 22.0, 22.0])
    fan_speed = np.array([0.7, 0.7, 0.7, 0.7])
    
    print("初始状态:")
    print(f"  T_in: {T_in}°C")
    print(f"  T_out: {T_out}°C")
    print(f"  IT_load: {IT_load}kW")
    
    print("\n仿真10步:")
    for step in range(10):
        T_in, H_in, T_supply, energy = model.step(
            T_in, T_out, H_in, IT_load, T_set, fan_speed
        )
        print(f"Step {step+1}: T_in={T_in:.2f}°C, Energy={energy:.2f}kWh")
    
    print("\n稳态分析:")
    T_steady, P_steady = model.get_steady_state(IT_load, T_out, 22.0)
    print(f"  稳态温度: {T_steady:.2f}°C")
    print(f"  稳态功率: {P_steady:.2f}kW")

