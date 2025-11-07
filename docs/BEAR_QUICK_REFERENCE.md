# BEAR 快速参考手册

## 🎯 核心概念速查

### 物理模型

| 概念 | 公式 | 说明 |
|------|------|------|
| **热平衡** | `C·dT/dt = Q_in - Q_out` | 房间温度变化 |
| **状态方程** | `dX/dt = A·X + B·U` | 多房间耦合 |
| **离散化** | `X[k+1] = A_d·X[k] + B_d·U[k]` | 仿真更新 |
| **矩阵指数** | `A_d = e^(A·Δt)` | 精确离散化 |
| **输入矩阵** | `B_d = A^(-1)·(A_d - I)·B` | 输入影响 |

---

### 状态空间

```
观测 = [T_rooms, T_outdoor, GHI, T_ground, Occupancy]
维度 = 3n + 2  (n = 房间数)
```

**示例**（6房间）：
```python
state = [
    22.1, 22.3, 22.0, 21.8, 22.2, 22.4,  # 6个房间温度
    35.2,                                 # 室外温度
    800, 800, 800, 800, 800, 800,        # 6个房间的太阳辐射
    28.5,                                 # 地面温度
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12  # 6个房间的人员热负荷
]
# 总维度: 6 + 1 + 6 + 1 + 6 = 20
```

---

### 动作空间

```
动作 = [power_1, power_2, ..., power_n]
范围 = [-1, 1]
```

**物理意义**：
- `-1`: 最大制冷
- `0`: 关闭
- `+1`: 最大制热

**实际功率**：
```python
Q_HVAC_i = action[i] * max_power  # W
```

---

### 奖励函数

```
reward = -α·||action||₂ - β·||error||₂
```

**默认权重**：
- `α = 0.001 × 24 = 0.024`: 能耗权重
- `β = 0.999`: 温度偏差权重

---

## 📊 数据格式

### EPW 文件

```
每小时一行，8760行/年
关键字段：
- temp_air: 室外温度 (°C)
- ghi: 全球水平辐照度 (W/m²)
```

### 建筑几何 (.table.htm)

```html
<tr><td>Zone Name</td><td>CORE_ZN</td></tr>
<tr><td>X Minimum</td><td>3.05</td></tr>
<tr><td>X Maximum</td><td>24.38</td></tr>
<tr><td>Floor Area</td><td>260.13</td></tr>
<tr><td>Exterior Window Area</td><td>0.0</td></tr>
```

### 参数字典

```python
Parameter = {
    'OutTemp': np.array,      # (8760,) 室外温度
    'roomnum': int,           # 房间数量
    'connectmap': np.array,   # (n, n+1) 连接矩阵
    'RCtable': np.array,      # (n, n+1) RC表
    'target': np.array,       # (n,) 目标温度
    'gamma': tuple,           # (2,) 奖励权重
    'ghi': np.array,          # (8760,) 太阳辐射
    'GroundTemp': np.array,   # (8760,) 地面温度
    'Occupancy': np.array,    # (8760,) 人员占用
    'max_power': int,         # 最大功率
    'time_resolution': int    # 时间分辨率（秒）
}
```

---

## 🔧 关键代码片段

### 创建环境

```python
from bear.BEAR.Utils.utils_building import ParameterGenerator
from bear.BEAR.Env.env_building import BuildingEnvReal

# 生成参数
params = ParameterGenerator(
    Building='OfficeSmall',
    Weather='Hot_Dry',
    Location='Tucson',
    target=22.0,
    reward_gamma=(0.001, 0.999),
    max_power=8000,
    time_reso=3600,
    root='bear/BEAR/Data/'
)

# 创建环境
env = BuildingEnvReal(params)
```

---

### 运行仿真

```python
# 重置
state, info = env.reset()

# 单步
action = np.array([0.5, -0.3, 0.0, 0.2, -0.1, 0.4])  # 6个房间
next_state, reward, done, truncated, info = env.step(action)

# 提取信息
room_temps = next_state[:env.roomnum]
outdoor_temp = next_state[env.roomnum]
zone_temps = info['zone_temperature']
```

---

### 状态更新（核心）

```python
# 准备输入
X = state[:roomnum]  # 当前温度
U = [Occupower, T_ground, T_outdoor, *actions, *GHI]

# 状态更新
X_new = A_d @ X + B_d @ U

# 计算奖励
error = X_new - target
reward = -||action||₂ * α - ||error||₂ * β
```

---

### 自定义奖励

```python
def my_reward(env, state, action, error, state_new):
    energy = np.linalg.norm(action, 2) * 0.01
    comfort = np.linalg.norm(error, 2) * 1.0
    return -energy - comfort

env = BuildingEnvReal(params, user_reward_function=my_reward)
```

---

## 📐 数学公式

### RC 模型

```
单房间：
C·dT/dt = (T_out - T)/R + Q_HVAC + Q_solar + Q_occupancy

多房间：
C_i·dT_i/dt = Σ_j (T_j - T_i)/R_ij + Q_HVAC_i + Q_solar_i + Q_occ_i
```

---

### 矩阵形式

```
A 矩阵（n×n）：
A_ii = -Σ_j (1/R_ij·C_i)  # 对角元素
A_ij = 1/(R_ij·C_i)        # 非对角元素

B 矩阵（n×m）：
B = [B_occ, B_ground, B_outdoor, B_HVAC, B_solar] / C
```

---

### 人员热负荷

```
Q_occ = c0 + c1·M + c2·M² - c3·T·M + c4·T·M² 
        - c5·T² + c6·T²·M - c7·T²·M²

其中：
M = 人员数量
T = 平均温度
c0...c7 = 系数（来自 EnergyPlus）
```

---

## 🎛️ 参数调优指南

### 时间分辨率

| 值 (秒) | 数据点/年 | 用途 |
|---------|----------|------|
| 3600 | 8,760 | 快速原型 |
| 1800 | 17,520 | 一般研究 |
| 900 | 35,040 | 高精度 |
| 300 | 105,120 | 实时控制 |

---

### 奖励权重

| 场景 | energy_weight | temp_weight | 说明 |
|------|--------------|-------------|------|
| 节能优先 | 0.01 | 0.99 | 最小化能耗 |
| 舒适优先 | 0.001 | 0.999 | 最小化温度偏差 |
| 平衡 | 0.005 | 0.995 | 能耗和舒适平衡 |

---

### HVAC 功率

| 建筑类型 | 推荐功率 (W) |
|---------|-------------|
| OfficeSmall | 8,000 |
| OfficeMedium | 15,000 |
| OfficeLarge | 30,000 |
| Hospital | 50,000 |
| HotelLarge | 40,000 |

---

## 🐛 常见问题

### Q: 状态维度不匹配

```python
# 错误
state_dim = 20  # 但实际是 21

# 解决
state_dim = env.observation_space.shape[0]
# 或
state_dim = 3 * env.roomnum + 2
```

---

### Q: 动作超出范围

```python
# 错误
action = np.array([1.5, -2.0, ...])  # 超出 [-1, 1]

# 解决
action = np.clip(action, -1.0, 1.0)
```

---

### Q: 奖励值过大/过小

```python
# 调整权重
params['gamma'] = (0.001, 0.999)  # 默认

# 或添加归一化
reward = reward / 100.0
```

---

### Q: 仿真速度慢

```python
# 1. 降低时间分辨率
time_reso = 3600  # 而非 300

# 2. 缩短回合
episode_length = 288  # 24小时而非全年

# 3. 使用并行环境
from tianshou.env import SubprocVectorEnv
envs = SubprocVectorEnv([make_env for _ in range(8)])
```

---

## 📚 支持的配置

### 建筑类型（16种）

```
OfficeSmall, OfficeMedium, OfficeLarge
Hospital
HotelSmall, HotelLarge
SchoolPrimary, SchoolSecondary
ApartmentHighRise, ApartmentMidRise
RestaurantFastFood, RestaurantSitDown
RetailStandalone, RetailStripmall
OutPatientHealthCare
Warehouse
```

---

### 气候类型（16种）

```
Very_Hot_Humid, Hot_Humid, Hot_Dry
Warm_Humid, Warm_Dry, Warm_Marine
Mixed_Humid, Mixed_Dry, Mixed_Marine
Cool_Humid, Cool_Dry, Cool_Marine
Cold_Humid, Cold_Dry
Very_Cold
Subarctic/Arctic
```

---

### 地理位置（19个）

```
Tucson, Tampa, Honolulu, Atlanta
NewYork, Seattle, SanDiego
Albuquerque, Denver, ElPaso
Buffalo, Rochester, GreatFalls
InternationalFalls, Fairbanks
PortAngeles, Dubai, HoChiMinh, NewDelhi
```

---

## 🔗 相关资源

### 文档

- **完整技术解析**: `docs/BEAR_TECHNICAL_DEEP_DIVE.md`
- **集成方案**: `docs/BEAR_INTEGRATION_PLAN.md`
- **快速开始**: `docs/BEAR_QUICKSTART.md`

### 代码

- **环境适配器**: `env/building_env_wrapper.py`
- **专家控制器**: `env/building_expert_controller.py`
- **训练脚本**: `main_building.py`

### 测试

- **基础测试**: `scripts/test_building_env_basic.py`
- **专家测试**: `scripts/test_building_expert.py`
- **演示**: `scripts/demo_building_env.py`

---

## 💡 最佳实践

### 1. 开发流程

```
1. 快速原型（1小时分辨率，24小时回合）
   ↓
2. 算法调优（30分钟分辨率，1周回合）
   ↓
3. 最终验证（15分钟分辨率，全年回合）
```

---

### 2. 超参数搜索

```python
# 网格搜索
for energy_w in [0.001, 0.005, 0.01]:
    for temp_w in [0.999, 0.995, 0.99]:
        env = BearEnvWrapper(
            energy_weight=energy_w,
            temp_weight=temp_w
        )
        # 训练和评估
```

---

### 3. 性能基准

```python
# 与专家对比
from env.building_expert_controller import create_expert_controller

expert = create_expert_controller(env, 'mpc')
expert_reward = evaluate(env, expert)
agent_reward = evaluate(env, agent)

print(f"Expert: {expert_reward:.2f}")
print(f"Agent: {agent_reward:.2f}")
print(f"Improvement: {(agent_reward/expert_reward - 1)*100:.1f}%")
```

---

## 🎯 总结

**BEAR 的核心优势**：
- ✅ 物理真实（RC 模型）
- ✅ 数据真实（EPW 气象）
- ✅ 场景丰富（304 种组合）
- ✅ 数值稳定（矩阵指数）
- ✅ 易于扩展（自定义奖励）

**适用场景**：
- 建筑 HVAC 控制研究
- 强化学习算法验证
- 能源管理策略开发
- 需求响应研究

**下一步**：
1. 查看 `docs/BEAR_TECHNICAL_DEEP_DIVE.md` 了解详细原理
2. 运行 `scripts/demo_building_env.py` 体验环境
3. 使用 `main_building.py` 开始训练
4. 发表你的研究成果！

祝研究顺利！🏢🌡️🚀

