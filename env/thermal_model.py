# ========================================
# 数据中心热力学模型（优化版本）
# ========================================
# 模拟数据中心的温度动态和能耗计算
# 基于简化的热平衡方程
#
# 优化内容：
# 1. 修复物理模型错误（COP计算、湿度模型）
# 2. 改进数值稳定性（除零保护、边界检查）
# 3. 添加输入验证和异常处理
# 4. 改进代码可读性和可维护性

import numpy as np
from typing import Tuple
import warnings


class ThermalModel:
    """
    数据中心热力学模型

    基于能量守恒和热传递原理：
    dT/dt = (Q_IT - Q_cooling - Q_loss) / (m * c_p)

    其中：
    - Q_IT: IT设备发热功率 (kW)
    - Q_cooling: 空调制冷功率 (kW)
    - Q_loss: 通过墙体等散失的热量 (kW)
    - m: 空气质量 (kg)
    - c_p: 空气比热容 (kJ/(kg·K))

    物理约束：
    - 温度范围: 15-35°C
    - 湿度范围: 30-70%
    - COP范围: 1.5-5.0
    - 风速范围: 0.3-1.0
    """

    # ========== 物理常量 ==========
    MIN_TEMP = 15.0      # 最低温度 (°C)
    MAX_TEMP = 35.0      # 最高温度 (°C)
    MIN_HUMIDITY = 30.0  # 最低湿度 (%)
    MAX_HUMIDITY = 70.0  # 最高湿度 (%)
    MIN_COP = 1.5        # 最低COP
    MAX_COP = 5.0        # 最高COP
    MIN_FAN_SPEED = 0.3  # 最低风速
    MAX_FAN_SPEED = 1.0  # 最高风速

    # ========== 模型参数（可配置的魔法数字） ==========
    TEMP_CHANGE_LIMIT = 3.0      # 单步最大温度变化 (°C)
    HUMIDITY_CHANGE_LIMIT = 5.0  # 单步最大湿度变化 (%)
    COOLING_TEMP_RANGE = 5.0     # 制冷量计算的温差范围 (°C)
    FAN_POWER_RATIO = 0.08       # 风机功率占制冷容量的比例
    STANDBY_POWER_RATIO = 0.03   # 待机功率占额定功率的比例
    AIR_FLOW_BASE = 10.0         # 基准风量 (m³/s)

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
        # 输入验证
        assert num_crac > 0, "num_crac must be positive"
        assert time_step > 0, "time_step must be positive"
        assert room_volume > 0, "room_volume must be positive"
        assert air_density > 0, "air_density must be positive"
        assert air_cp > 0, "air_cp must be positive"
        assert wall_ua >= 0, "wall_ua must be non-negative"
        assert crac_capacity > 0, "crac_capacity must be positive"
        assert self.MIN_COP <= cop_nominal <= self.MAX_COP, f"cop_nominal must be in [{self.MIN_COP}, {self.MAX_COP}]"

        self.num_crac = num_crac
        self.dt = time_step
        self.air_density = air_density  # 保存用于后续计算

        # ========== 热容参数 ==========
        self.m_air = room_volume * air_density  # 空气质量 (kg)
        self.c_p = air_cp                       # 比热容 (kJ/(kg·K))
        self.thermal_mass = self.m_air * self.c_p  # 热容 (kJ/K)

        # ========== 热传递参数 ==========
        self.UA_wall = wall_ua  # 墙体热传导 (kW/K)

        # ========== CRAC参数 ==========
        self.Q_crac_max = crac_capacity  # 单个CRAC最大制冷量 (kW)
        self.COP_nominal = cop_nominal   # 名义COP

        # 计算额定功率（用于待机功率计算）
        self.P_crac_rated = self.Q_crac_max / self.COP_nominal  # 额定电功率 (kW)

        # ========== 效率模型参数 ==========
        # COP随负载率和温差变化：COP = COP_nominal * f(load_ratio, ΔT)
        # 基于实际CRAC性能曲线
        self.cop_load_coef = [0.6, 0.8, 1.0, 0.95]  # 负载率[0.3, 0.5, 1.0, 1.2]对应的COP系数
        self.cop_temp_coef = 0.015  # 温差每增加1K，COP下降1.5%（修正为更合理的值）

        # ========== 湿度模型参数 ==========
        # 改进的湿度模型：考虑空调除湿和IT设备产湿
        self.dehumidify_rate = 0.8   # 空调除湿速率 (%/h per unit fan speed)
        self.humidity_gain_rate = 0.3  # IT设备产湿速率 (%/h per 100kW load)
    
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
        # ========== 输入验证 ==========
        self._validate_inputs(T_in, T_out, H_in, IT_load, T_set, fan_speed)

        # ========== 1. 计算IT设备发热 ==========
        Q_IT = IT_load  # IT设备发热功率 (kW)

        # ========== 2. 计算墙体散热 ==========
        # Q_loss = UA * (T_in - T_out)
        # 正值表示向外散热，负值表示从外界吸热
        Q_loss = self.UA_wall * (T_in - T_out)

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
        next_T_in_raw = T_in + dT_dt * self.dt

        # 温度物理约束：
        # 1. 限制单步变化幅度（避免数值不稳定）
        # 2. 限制在物理合理范围内
        next_T_in = np.clip(
            next_T_in_raw,
            max(self.MIN_TEMP, T_in - self.TEMP_CHANGE_LIMIT),
            min(self.MAX_TEMP, T_in + self.TEMP_CHANGE_LIMIT)
        )

        # 如果温度被裁剪，发出警告（可能表示模型不稳定）
        if abs(next_T_in - next_T_in_raw) > 0.1:
            warnings.warn(
                f"Temperature change clipped: {next_T_in_raw:.2f} -> {next_T_in:.2f}°C. "
                f"Consider reducing time step or checking model parameters.",
                RuntimeWarning
            )

        # ========== 5. 湿度模型（改进版） ==========
        # 考虑空调除湿和IT设备产湿的物理过程
        # dH/dt = -除湿速率 * 风速 + 产湿速率 * 负载 + 噪声
        fan_avg = np.mean(fan_speed)

        # 除湿：空调运行时除湿
        dehumidify = self.dehumidify_rate * fan_avg

        # 产湿：IT设备运行产生湿度（简化模型）
        humidify = self.humidity_gain_rate * (IT_load / 100.0)

        # 湿度变化率 (%/h)
        dH_dt = -dehumidify + humidify

        # 添加小的随机扰动（模拟测量噪声和未建模因素）
        noise_std = 0.05  # 标准差 (%/h)
        dH_dt += np.random.normal(0, noise_std)

        # 更新湿度
        next_H_in_raw = H_in + dH_dt * self.dt

        # 湿度物理约束
        next_H_in = np.clip(
            next_H_in_raw,
            max(self.MIN_HUMIDITY, H_in - self.HUMIDITY_CHANGE_LIMIT),
            min(self.MAX_HUMIDITY, H_in + self.HUMIDITY_CHANGE_LIMIT)
        )

        # ========== 6. 计算总能耗 ==========
        # 能耗 = CRAC电功率 * 时间步长
        energy_consumed = P_crac_total * self.dt  # kWh

        return next_T_in, next_H_in, np.array(T_supply_list), energy_consumed

    def _validate_inputs(
        self,
        T_in: float,
        T_out: float,
        H_in: float,
        IT_load: float,
        T_set: np.ndarray,
        fan_speed: np.ndarray
    ) -> None:
        """
        验证输入参数的合理性

        抛出 ValueError 如果输入不合理
        """
        # 温度范围检查
        if not (self.MIN_TEMP <= T_in <= self.MAX_TEMP):
            raise ValueError(f"T_in={T_in}°C out of range [{self.MIN_TEMP}, {self.MAX_TEMP}]")

        if not (-20 <= T_out <= 50):
            raise ValueError(f"T_out={T_out}°C out of reasonable range [-20, 50]")

        # 湿度范围检查
        if not (0 <= H_in <= 100):
            raise ValueError(f"H_in={H_in}% out of range [0, 100]")

        # 负载检查
        if IT_load < 0:
            raise ValueError(f"IT_load={IT_load}kW must be non-negative")

        # 数组长度检查
        if len(T_set) != self.num_crac:
            raise ValueError(f"T_set length {len(T_set)} != num_crac {self.num_crac}")

        if len(fan_speed) != self.num_crac:
            raise ValueError(f"fan_speed length {len(fan_speed)} != num_crac {self.num_crac}")

        # 设定温度范围检查
        if not np.all((self.MIN_TEMP <= T_set) & (T_set <= self.MAX_TEMP)):
            raise ValueError(f"T_set contains values out of range [{self.MIN_TEMP}, {self.MAX_TEMP}]")

        # 风速范围检查
        if not np.all((self.MIN_FAN_SPEED <= fan_speed) & (fan_speed <= self.MAX_FAN_SPEED)):
            raise ValueError(f"fan_speed contains values out of range [{self.MIN_FAN_SPEED}, {self.MAX_FAN_SPEED}]")
    
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
        delta_T = T_in - T_set  # 温差 (°C)

        if delta_T > 0:
            # 需要制冷
            # 制冷量与温差和风速成正比，使用归一化的温差
            # 当温差达到COOLING_TEMP_RANGE时，制冷量达到最大值
            temp_factor = min(delta_T / self.COOLING_TEMP_RANGE, 1.0)
            Q_demand = self.Q_crac_max * fan_speed * temp_factor
        else:
            # 不需要制冷（或需要加热，这里简化为0）
            Q_demand = 0.0

        # 限制在CRAC容量范围内（考虑风速限制）
        Q_cooling = np.clip(Q_demand, 0.0, self.Q_crac_max * fan_speed)

        # ========== 2. 计算COP（能效比） ==========
        # COP受负载率和温差影响

        # 2.1 负载率对COP的影响（部分负载效率下降）
        if self.Q_crac_max > 0:
            load_ratio = Q_cooling / self.Q_crac_max
        else:
            load_ratio = 0.0

        # 使用分段线性插值计算负载系数
        if load_ratio < 0.3:
            cop_load_factor = 0.6
        elif load_ratio < 0.5:
            cop_load_factor = 0.8
        elif load_ratio < 1.0:
            cop_load_factor = 1.0
        else:
            cop_load_factor = 0.95

        # 2.2 温差对COP的影响（温差越大，效率越低）
        # 使用绝对温差（修复冬季COP计算错误）
        delta_T_cond = abs(T_out - T_in)  # 冷凝器与蒸发器的温差

        # COP随温差增加而下降（基于卡诺循环效率）
        # cop_temp_factor = 1 / (1 + coef * ΔT)
        cop_temp_factor = 1.0 / (1.0 + self.cop_temp_coef * delta_T_cond)
        cop_temp_factor = np.clip(cop_temp_factor, 0.5, 1.0)  # 限制在合理范围

        # 2.3 实际COP
        COP_actual = self.COP_nominal * cop_load_factor * cop_temp_factor
        COP_actual = np.clip(COP_actual, self.MIN_COP, self.MAX_COP)  # 物理约束

        # ========== 3. 计算电功率 ==========
        if Q_cooling > 0:
            # 压缩机功率：P_comp = Q / COP
            P_compressor = Q_cooling / COP_actual

            # 风机功率：与风速的三次方成正比（流体力学）
            P_fan_rated = self.FAN_POWER_RATIO * self.Q_crac_max
            P_fan = P_fan_rated * (fan_speed ** 3)

            # 总电功率
            P_electric = P_compressor + P_fan
        else:
            # 待机功率（控制系统、传感器等）
            P_electric = self.STANDBY_POWER_RATIO * self.P_crac_rated

        # ========== 4. 计算供风温度 ==========
        # T_supply = T_in - ΔT_supply
        # ΔT_supply取决于制冷量和风量
        if Q_cooling > 0 and fan_speed > 0:
            # 风量与风速成正比（简化假设）
            air_flow = fan_speed * self.AIR_FLOW_BASE  # m³/s

            # 供风温差：ΔT = Q / (ρ * V * c_p)
            # Q: kW = kJ/s
            # ρ: kg/m³
            # V: m³/s
            # c_p: kJ/(kg·K)
            # ΔT: K
            air_mass_flow = self.air_density * air_flow  # kg/s
            heat_capacity_flow = air_mass_flow * self.c_p  # kJ/(s·K) = kW/K

            # 除零保护
            if heat_capacity_flow > 1e-3:
                delta_T_supply = Q_cooling / heat_capacity_flow  # K
            else:
                delta_T_supply = 0.0

            T_supply = T_in - delta_T_supply
        else:
            # 无制冷时，供风温度等于回风温度
            T_supply = T_in

        # 供风温度物理约束
        T_supply = np.clip(T_supply, self.MIN_TEMP, self.MAX_TEMP)

        return Q_cooling, P_electric, T_supply
    
    def get_steady_state(
        self,
        IT_load: float,
        T_out: float,
        T_set: float,
        fan_speed: float = 0.7,
        max_iter: int = 50,
        tol: float = 0.01
    ) -> Tuple[float, float]:
        """
        计算稳态温度和能耗（改进版：使用迭代法求解）

        用于验证模型或初始化

        稳态条件：dT/dt = 0
        即：Q_IT = Q_cooling + Q_loss

        参数：
        - IT_load: IT负载 (kW)
        - T_out: 室外温度 (°C)
        - T_set: 设定温度 (°C)
        - fan_speed: 风速比例（默认0.7）
        - max_iter: 最大迭代次数
        - tol: 收敛容差 (°C)

        返回：
        - T_steady: 稳态机房温度 (°C)
        - P_steady: 稳态功率 (kW)
        """
        # 初始猜测：从设定温度开始
        T_in = T_set

        # 迭代求解稳态温度
        for iteration in range(max_iter):
            T_in_old = T_in

            # 计算墙体散热
            Q_loss = self.UA_wall * (T_in - T_out)

            # 所需制冷量
            Q_cooling_needed = IT_load - Q_loss

            # 如果需要的制冷量为负（冬季），设定温度可能过低
            if Q_cooling_needed < 0:
                # 不需要制冷，温度会低于设定值
                T_in = T_set
                P_total = self.num_crac * self.STANDBY_POWER_RATIO * self.P_crac_rated
                break

            # 计算每个CRAC的制冷量（假设均匀分配）
            Q_per_crac = Q_cooling_needed / self.num_crac

            # 检查是否超过CRAC容量
            if Q_per_crac > self.Q_crac_max * fan_speed:
                # 制冷能力不足，温度会高于设定值
                Q_per_crac = self.Q_crac_max * fan_speed
                Q_cooling_total = Q_per_crac * self.num_crac

                # 重新计算稳态温度
                # Q_IT = Q_cooling + UA * (T_in - T_out)
                # T_in = (Q_IT - Q_cooling) / UA + T_out
                if self.UA_wall > 0:
                    T_in = (IT_load - Q_cooling_total) / self.UA_wall + T_out
                else:
                    T_in = T_set + 5.0  # 如果没有墙体散热，温度会持续上升
            else:
                # 制冷能力足够，温度接近设定值
                # 考虑温差的影响
                delta_T = T_in - T_set
                T_in = T_set + delta_T * 0.5  # 阻尼更新

            # 检查收敛
            if abs(T_in - T_in_old) < tol:
                break
        else:
            # 未收敛，发出警告
            warnings.warn(
                f"Steady state calculation did not converge after {max_iter} iterations. "
                f"Final temperature change: {abs(T_in - T_in_old):.3f}°C",
                RuntimeWarning
            )

        # 计算稳态功率
        # 使用最终的温度计算CRAC功率
        T_set_array = np.full(self.num_crac, T_set)
        fan_speed_array = np.full(self.num_crac, fan_speed)

        P_total = 0.0
        for i in range(self.num_crac):
            Q_cool, P_elec, _ = self._crac_model(T_in, T_out, T_set_array[i], fan_speed_array[i])
            P_total += P_elec

        return T_in, P_total


class SimplifiedThermalModel:
    """
    简化热力学模型（用于快速原型和测试）

    使用一阶惯性环节近似：
    T(k+1) = a * T(k) + b * T_set + c * Q_IT + d * T_out

    注意：此模型仅用于快速测试，不适合精确仿真
    """

    def __init__(
        self,
        num_crac: int = 4,
        time_step: float = 5.0/60.0,
        time_constant: float = 0.5,  # 时间常数（小时）
    ):
        """
        初始化简化模型

        参数：
        - num_crac: CRAC数量
        - time_step: 时间步长（小时）
        - time_constant: 热时间常数（小时）
        """
        self.num_crac = num_crac
        self.dt = time_step
        self.tau = time_constant

        # 一阶惯性系数
        self.a = np.exp(-self.dt / self.tau)  # 温度衰减系数
        self.b = 0.3 * (1 - self.a)           # 设定温度影响
        self.c = 0.01 * (1 - self.a)          # IT负载影响
        self.d = 0.1 * (1 - self.a)           # 室外温度影响

        # 能耗模型参数
        self.wall_ua_approx = 50.0  # 近似墙体热传导系数 (kW/K)
        self.cop_approx = 3.0       # 近似COP

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

        参数和返回值与ThermalModel.step()相同
        """
        # 平均设定温度和风速
        T_set_avg = np.mean(T_set)
        fan_avg = np.mean(fan_speed)

        # 一阶惯性更新温度
        next_T_in = (
            self.a * T_in +
            self.b * T_set_avg +
            self.c * IT_load +
            self.d * T_out +
            np.random.normal(0, 0.1)  # 噪声
        )

        # 温度约束
        next_T_in = np.clip(next_T_in, 15.0, 35.0)

        # 湿度简化模型
        dH = -0.5 * fan_avg + 0.3 * (IT_load / 100.0)
        next_H_in = H_in + dH * self.dt + np.random.normal(0, 0.05)
        next_H_in = np.clip(next_H_in, 30.0, 70.0)

        # 供风温度（简化）
        next_T_supply = T_set - 2.0 * fan_speed
        next_T_supply = np.clip(next_T_supply, 15.0, 30.0)

        # 能耗简化模型（修复：移除重复的num_crac）
        Q_loss = self.wall_ua_approx * (T_in - T_out)
        Q_cooling = max(0, IT_load - Q_loss)

        # 总功率 = 制冷功率 / COP（已经是所有CRAC的总和）
        P_total = Q_cooling / self.cop_approx

        # 加上风机功率
        P_fan_total = 0.08 * 100.0 * self.num_crac * fan_avg  # 假设每个CRAC 100kW
        P_total += P_fan_total

        # 能耗 = 功率 * 时间
        energy_consumed = P_total * self.dt  # kWh

        return next_T_in, next_H_in, next_T_supply, energy_consumed


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("数据中心热力学模型测试（优化版）")
    print("=" * 60)

    # 测试热力学模型
    model = ThermalModel(num_crac=4)

    T_in = 24.0
    T_out = 30.0
    H_in = 50.0
    IT_load = 200.0
    T_set = np.array([22.0, 22.0, 22.0, 22.0])
    fan_speed = np.array([0.7, 0.7, 0.7, 0.7])

    print("\n初始状态:")
    print(f"  机房温度: {T_in}°C")
    print(f"  室外温度: {T_out}°C")
    print(f"  机房湿度: {H_in}%")
    print(f"  IT负载: {IT_load}kW")
    print(f"  设定温度: {T_set[0]}°C")
    print(f"  风速: {fan_speed[0]}")

    print("\n" + "=" * 60)
    print("动态仿真（10步）:")
    print("=" * 60)
    print(f"{'Step':<6} {'T_in(°C)':<10} {'H_in(%)':<10} {'Energy(kWh)':<12} {'T_supply(°C)':<12}")
    print("-" * 60)

    for step in range(10):
        T_in, H_in, T_supply, energy = model.step(
            T_in, T_out, H_in, IT_load, T_set, fan_speed
        )
        print(f"{step+1:<6} {T_in:<10.2f} {H_in:<10.2f} {energy:<12.3f} {T_supply[0]:<12.2f}")

    print("\n" + "=" * 60)
    print("稳态分析:")
    print("=" * 60)
    T_steady, P_steady = model.get_steady_state(IT_load, T_out, 22.0)
    print(f"  稳态温度: {T_steady:.2f}°C")
    print(f"  稳态功率: {P_steady:.2f}kW")
    print(f"  稳态PUE: {(P_steady + IT_load) / IT_load:.2f}")

    print("\n" + "=" * 60)
    print("边界条件测试:")
    print("=" * 60)

    # 测试1: 极端高温
    print("\n测试1: 极端高温（T_out=45°C）")
    T_in_test = 28.0
    T_out_test = 45.0
    T_in_test, H_in_test, T_supply_test, energy_test = model.step(
        T_in_test, T_out_test, 50.0, IT_load, T_set, fan_speed
    )
    print(f"  结果: T_in={T_in_test:.2f}°C, Energy={energy_test:.3f}kWh")

    # 测试2: 低负载
    print("\n测试2: 低负载（IT_load=50kW）")
    T_in_test = 24.0
    T_out_test = 30.0
    IT_load_test = 50.0
    T_in_test, H_in_test, T_supply_test, energy_test = model.step(
        T_in_test, T_out_test, 50.0, IT_load_test, T_set, fan_speed
    )
    print(f"  结果: T_in={T_in_test:.2f}°C, Energy={energy_test:.3f}kWh")

    # 测试3: 冬季工况
    print("\n测试3: 冬季工况（T_out=5°C）")
    T_in_test = 22.0
    T_out_test = 5.0
    T_in_test, H_in_test, T_supply_test, energy_test = model.step(
        T_in_test, T_out_test, 50.0, IT_load, T_set, fan_speed
    )
    print(f"  结果: T_in={T_in_test:.2f}°C, Energy={energy_test:.3f}kWh")

    print("\n" + "=" * 60)
    print("输入验证测试:")
    print("=" * 60)

    # 测试无效输入
    try:
        model.step(100.0, 30.0, 50.0, IT_load, T_set, fan_speed)  # 温度过高
        print("  ❌ 温度验证失败")
    except ValueError as e:
        print(f"  ✅ 温度验证通过: {e}")

    try:
        model.step(24.0, 30.0, 50.0, -10.0, T_set, fan_speed)  # 负载为负
        print("  ❌ 负载验证失败")
    except ValueError as e:
        print(f"  ✅ 负载验证通过: {e}")

    try:
        model.step(24.0, 30.0, 50.0, IT_load, np.array([22.0, 22.0]), fan_speed)  # 数组长度错误
        print("  ❌ 数组长度验证失败")
    except ValueError as e:
        print(f"  ✅ 数组长度验证通过: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

