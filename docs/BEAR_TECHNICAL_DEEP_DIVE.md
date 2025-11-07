# BEAR 建筑模拟环境深度技术解析

## 📚 目录

1. [物理模型基础](#1-物理模型基础)
2. [数据输入系统](#2-数据输入系统)
3. [仿真流程详解](#3-仿真流程详解)
4. [控制接口设计](#4-控制接口设计)
5. [关键代码解析](#5-关键代码解析)
6. [数学公式推导](#6-数学公式推导)

---

## 1. 物理模型基础

### 1.1 RC 热力学模型原理

BEAR 使用 **RC (Resistance-Capacitance) 网络模型** 来模拟建筑的热力学行为，这是一种经典的建筑能源建模方法。

#### **基本概念**

RC 模型将建筑热力学系统类比为电路：

| 热力学量 | 电路类比 | 单位 |
|---------|---------|------|
| 温度 (T) | 电压 (V) | °C |
| 热流 (Q) | 电流 (I) | W |
| 热阻 (R) | 电阻 (R) | °C/W |
| 热容 (C) | 电容 (C) | J/°C |

#### **物理方程**

对于单个房间，热平衡方程为：

```
C · dT/dt = Q_in - Q_out
```

其中：
- `C`: 房间热容 (J/°C)
- `T`: 房间温度 (°C)
- `Q_in`: 输入热流（HVAC、太阳辐射、人员等）
- `Q_out`: 输出热流（通过墙体、窗户等散热）

#### **多房间耦合**

对于 n 个房间的建筑，热平衡方程组为：

```
C_i · dT_i/dt = Σ_j (T_j - T_i)/R_ij + Q_HVAC_i + Q_solar_i + Q_occupancy_i
```

其中：
- `i, j`: 房间索引
- `R_ij`: 房间 i 和 j 之间的热阻
- `Q_HVAC_i`: 房间 i 的 HVAC 功率
- `Q_solar_i`: 房间 i 的太阳辐射热增益
- `Q_occupancy_i`: 房间 i 的人员热负荷

---

### 1.2 状态空间方程

#### **连续时间状态空间**

将多房间热平衡方程写成矩阵形式：

```
dX/dt = A·X + B·U
```

其中：
- **X**: 状态向量 (n×1)，表示 n 个房间的温度
  ```
  X = [T_1, T_2, ..., T_n]^T
  ```

- **U**: 输入向量 (m×1)，包含所有外部输入
  ```
  U = [Occupower, T_ground, T_outdoor, Q_HVAC_1, ..., Q_HVAC_n, GHI_1, ..., GHI_n]^T
  ```

- **A**: 系统矩阵 (n×n)，描述房间间热传导
- **B**: 输入矩阵 (n×m)，描述外部输入的影响

#### **A 矩阵的构建**

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
# 定义 A 矩阵（系统矩阵）
Amatrix = self.RCtable[:, :-1]  # RCtable = R/C，热阻除以热容
diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
np.fill_diagonal(Amatrix, np.diag(diagvalue))
Amatrix += self.nonlinear * self.OCCU_COEF_LINEAR / self.roomnum
````
</augment_code_snippet>

**A 矩阵的物理意义**：
- **对角元素** `A_ii`: 房间 i 的总热损失系数（负值）
  ```
  A_ii = -Σ_j (1/R_ij·C_i) - (1/R_i_ground·C_i)
  ```
- **非对角元素** `A_ij`: 房间 i 和 j 之间的热传导系数
  ```
  A_ij = 1/(R_ij·C_i)
  ```

#### **B 矩阵的构建**

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
# 定义 B 矩阵（输入矩阵）
Bmatrix = self.weightCmap.T
Bmatrix[2] = self.connectmap[:, -1] * (self.RCtable[:, -1])
Bmatrix = (Bmatrix.T)
````
</augment_code_snippet>

**B 矩阵的结构**：
```
B = [B_occupancy, B_ground, B_outdoor, B_HVAC, B_solar] / C
```

每一列对应一个输入源的影响系数。

---

### 1.3 离散化处理

由于仿真是离散时间的，需要将连续时间方程离散化。

#### **离散时间状态空间**

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
# 计算离散时间系统矩阵
self.A_d = expm(Amatrix * self.timestep)  # 矩阵指数
self.B_d = LA.inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix
````
</augment_code_snippet>

**数学推导**：

对于连续系统 `dX/dt = A·X + B·U`，离散化后：

```
X[k+1] = A_d·X[k] + B_d·U[k]
```

其中：
- `A_d = e^(A·Δt)`: 使用矩阵指数函数
- `B_d = A^(-1)·(A_d - I)·B`: 精确离散化公式

**为什么使用矩阵指数？**
- 保证数值稳定性
- 精确求解线性微分方程
- 避免欧拉法的累积误差

---

### 1.4 房间间热传导建模

#### **连接矩阵 (connectmap)**

`connectmap` 是一个 (n×(n+1)) 矩阵，表示房间之间的连接关系：

```
connectmap[i][j] = 1  如果房间 i 和 j 相邻
connectmap[i][j] = 0  否则
connectmap[i][n] = 1  如果房间 i 与室外相连
```

#### **热阻表 (Rtable)**

`Rtable` 是一个 (n×(n+1)) 矩阵，存储热阻值：

```
Rtable[i][j] = R_ij  房间 i 和 j 之间的热阻 (°C/W)
Rtable[i][n] = R_i_out  房间 i 与室外的热阻
```

**热阻的计算**：

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
# 墙体热阻
U = height * length * Walltype  # U = 传热系数 × 面积
Rtable[i][j] = U

# 窗户热阻
Rtable[i][-1] = ExteriorArea * OutWall + WindowArea * Window
````
</augment_code_snippet>

其中：
- `Walltype`: 墙体传热系数 (W/(m²·°C))
- `OutWall`: 外墙传热系数
- `Window`: 窗户传热系数

---

### 1.5 太阳辐射建模

#### **太阳热增益系数 (SHGC)**

太阳辐射通过窗户进入房间的热量：

```
Q_solar_i = GHI_i × WindowArea_i × SHGC
```

其中：
- `GHI_i`: 全球水平辐照度 (W/m²)
- `WindowArea_i`: 窗户面积 (m²)
- `SHGC`: 太阳热增益系数 (Solar Heat Gain Coefficient)

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
# 计算 SHGC
SHGC = shgc * shgc_weight * (max(data[0]['ghi']) / (abs(data[1]['TZ']) / 60))
````
</augment_code_snippet>

---

### 1.6 人员热负荷建模

#### **非线性人员热负荷模型**

人员热负荷不是简单的线性关系，而是考虑了温度和人数的非线性交互：

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
def _calc_occupower(self, avg_temp: float, Meta: float) -> float:
    return (
        self.OCCU_COEF[0]
        + self.OCCU_COEF[1] * Meta
        + self.OCCU_COEF[2] * Meta**2
        - self.OCCU_COEF[3] * avg_temp * Meta
        + self.OCCU_COEF[4] * avg_temp * Meta**2
        - self.OCCU_COEF[5] * avg_temp**2
        + self.OCCU_COEF[6] * avg_temp**2 * Meta
        - self.OCCU_COEF[7] * avg_temp**2 * Meta**2
    )
````
</augment_code_snippet>

**公式**：

```
Q_occ = c0 + c1·M + c2·M² - c3·T·M + c4·T·M² - c5·T² + c6·T²·M - c7·T²·M²
```

其中：
- `M`: 人员占用率 (人数)
- `T`: 平均温度 (°C)
- `c0...c7`: 系数（来自 EnergyPlus 工程参考手册）

**系数来源**：
```python
OCCU_COEF = [6.461927, 0.946892, 0.0000255737, 0.0627909, 
             0.0000589172, 0.19855, 0.000940018, 0.00000149532]
```

这些系数来自 [EnergyPlus Engineering Reference, Page 1299](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v23.1.0/EngineeringReference.pdf)

---

## 2. 数据输入系统

### 2.1 EPW 气象文件

#### **EPW 文件格式**

EPW (EnergyPlus Weather) 是建筑能源模拟的标准气象数据格式。

**文件结构**：
```
LOCATION,Tucson,AZ,USA,TMY3,722745,32.12,-110.93,-7.0,779.0
DESIGN CONDITIONS,...
TYPICAL/EXTREME PERIODS,...
GROUND TEMPERATURES,...
HOLIDAYS/DAYLIGHT SAVING,...
COMMENTS 1,...
COMMENTS 2,...
DATA PERIODS,...
2001,1,1,1,0,?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,8.3,2.8,68,...
...
```

**数据字段**（每小时一行，8760 行/年）：
1. Year, Month, Day, Hour, Minute
2. Data Source and Uncertainty Flags
3. **Dry Bulb Temperature** (°C) - 室外温度
4. Dew Point Temperature (°C)
5. Relative Humidity (%)
6. Atmospheric Station Pressure (Pa)
7. **Global Horizontal Radiation** (Wh/m²) - 太阳辐射
8. Direct Normal Radiation (Wh/m²)
9. Diffuse Horizontal Radiation (Wh/m²)
10. ... (共 35 个字段)

#### **EPW 文件读取**

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
# 使用 pvlib 读取 EPW 文件
data = pvlib.iotools.read_epw(weatherfile[0])

# 提取室外温度
oneyear = data[0]['temp_air']  # 8760 个数据点

# 提取全球水平辐照度 (GHI)
oneyearrad = data[0]['ghi']  # 8760 个数据点
````
</augment_code_snippet>

**返回值**：
- `data[0]`: DataFrame，包含所有气象数据
- `data[1]`: 元数据字典（位置、时区等）

---

### 2.2 时间分辨率插值

EPW 文件默认是每小时一个数据点，但仿真可能需要更高的时间分辨率（如 5 分钟）。

#### **线性插值**

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
# 原始数据点
num_datapoint = len(oneyear)  # 8760
x = np.arange(0, num_datapoint)
y = np.array(oneyear)

# 创建插值函数
f = interpolate.interp1d(x, y)

# 生成新的时间点
xnew = np.arange(0, num_datapoint-1, 1/3600*time_reso)
outtempdatanew = f(xnew)
````
</augment_code_snippet>

**示例**：
- 原始：8760 点（每小时）
- `time_reso = 3600` (1小时)：8760 点
- `time_reso = 1800` (30分钟)：17520 点
- `time_reso = 300` (5分钟)：105120 点

---

### 2.3 建筑几何信息

#### **.table.htm 文件**

这是 EnergyPlus 生成的 HTML 表格文件，包含建筑的几何信息。

**文件示例**：
```html
<table>
  <tr><td>Zone Name</td><td>CORE_ZN</td></tr>
  <tr><td>Z Axis</td><td>0.0</td></tr>
  <tr><td>X Minimum</td><td>3.05</td></tr>
  <tr><td>X Maximum</td><td>24.38</td></tr>
  <tr><td>Y Minimum</td><td>3.05</td></tr>
  <tr><td>Y Maximum</td><td>15.24</td></tr>
  <tr><td>Z Minimum</td><td>0.0</td></tr>
  <tr><td>Z Maximum</td><td>2.74</td></tr>
  <tr><td>Floor Area</td><td>260.13</td></tr>
  <tr><td>Exterior Gross Wall Area</td><td>0.0</td></tr>
  <tr><td>Exterior Window Area</td><td>0.0</td></tr>
</table>
```

#### **解析几何信息**

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
def Getroominfor(filename: str):
    """解析 HTML 文件获取房间信息"""
    htmllines = open(filename).readlines()
    
    for line in htmllines:
        if 'Zone Name' in line:
            zone_name = extract_value(line)
        if 'X Minimum' in line:
            x_min = float(extract_value(line))
        # ... 提取其他字段
    
    return Layerall, roomnum, buildall
````
</augment_code_snippet>

**返回值**：
- `Layerall`: 按楼层分组的房间列表
- `roomnum`: 房间总数
- `buildall`: 所有房间的完整信息

---

### 2.4 地面温度数据

地面温度按月份预定义，基于地理位置。

<augment_code_snippet path="bear/BEAR/Utils/utils_building.py" mode="EXCERPT">
````python
GroundTemp_dic = {
    'Tucson': [20.9, 15.4, 11.9, 14.8, 12.7, 15.4, 
               23.3, 26.3, 31.2, 30.4, 29.8, 27.8],  # 12个月
    'Tampa': [24.2, 18.9, 15.7, 13.6, 15.5, 17.1, 
              21.2, 26.9, 27.6, 27.9, 27.4, 26.2],
    # ... 19个位置
}
````
</augment_code_snippet>

**扩展到全年**：

```python
groundtemp = np.concatenate([
    np.ones(31*24*3600//time_reso) * city[0],  # 1月
    np.ones(28*24*3600//time_reso) * city[1],  # 2月
    # ... 12个月
])
```

---

### 2.5 人员占用模式

人员占用模式通过活动时间表定义。

```python
activity_sch = np.array([...])  # 每个时间步的人员数量
```

**典型模式**：
- **办公楼**：工作日 8:00-18:00 高占用，夜间和周末低占用
- **医院**：24小时持续占用
- **学校**：工作日 8:00-15:00 高占用

---

## 3. 仿真流程详解

### 3.1 环境初始化

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
def __init__(self, Parameter: Dict[str, Any]):
    # 1. 加载参数
    self.OutTemp = Parameter['OutTemp']
    self.RCtable = Parameter['RCtable']
    self.roomnum = Parameter['roomnum']
    # ...
    
    # 2. 定义动作空间
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(roomnum,)
    )
    
    # 3. 定义观测空间
    self.observation_space = gym.spaces.Box(
        low=self.low, high=self.high
    )
    
    # 4. 构建 A 和 B 矩阵
    self.A_d = expm(Amatrix * self.timestep)
    self.B_d = LA.inv(Amatrix) @ (self.A_d - I) @ Bmatrix
````
</augment_code_snippet>

---

### 3.2 重置环境

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
def reset(self, seed=None, options=None):
    # 1. 重置时间步
    self.epochs = 0
    
    # 2. 初始化温度
    T_initial = self.target  # 从目标温度开始
    
    # 3. 计算初始人员热负荷
    avg_temp = np.sum(T_initial) / self.roomnum
    Meta = self.Occupancy[self.epochs]
    self.Occupower = self._calc_occupower(avg_temp, Meta)
    
    # 4. 构建初始状态
    self.state = np.concatenate((
        T_initial,                          # 房间温度
        self.OutTemp[self.epochs],          # 室外温度
        np.full(..., self.ghi[self.epochs]),  # 太阳辐射
        self.GroundTemp[self.epochs],       # 地面温度
        np.full(..., self.Occupower/1000)   # 人员热负荷
    ))
    
    return self.state, {}
````
</augment_code_snippet>

---

### 3.3 单步仿真

这是 BEAR 的核心！

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
def step(self, action: np.ndarray):
    # 1. 准备输入向量
    X = self.state[:self.roomnum]  # 当前温度
    Y = np.insert(
        np.append(action, self.ghi[self.epochs]),  # HVAC + 太阳辐射
        0, self.OutTemp[self.epochs]  # 室外温度
    )
    Y = np.insert(Y, 0, self.GroundTemp[self.epochs])  # 地面温度
    
    # 2. 计算人员热负荷
    avg_temp = np.sum(X) / self.roomnum
    Meta = self.Occupancy[self.epochs]
    self.Occupower = self._calc_occupower(avg_temp, Meta)
    Y = np.insert(Y, 0, self.Occupower)
    
    # 3. 状态更新（核心方程）
    X_new = self.A_d @ X + self.B_d @ Y
    
    # 4. 计算奖励
    error = X_new * self.acmap - self.target * self.acmap
    reward = -LA.norm(action, 2) * self.q_rate - LA.norm(error, 2) * self.error_rate
    
    # 5. 更新状态
    self.state = np.concatenate((
        X_new,
        self.OutTemp[self.epochs],
        np.full(..., self.ghi[self.epochs]),
        self.GroundTemp[self.epochs],
        np.full(..., self.Occupower/1000)
    ))
    
    # 6. 更新时间步
    self.epochs += 1
    done = (self.epochs >= self.length_of_weather - 1)
    
    return self.state, reward, done, done, {}
````
</augment_code_snippet>

---

## 4. 控制接口设计

### 4.1 动作空间

**定义**：
```python
action_space = Box(low=-1.0, high=1.0, shape=(n,))
```

**物理意义**：
- `action[i] = -1`: 房间 i 最大制冷功率
- `action[i] = 0`: 房间 i HVAC 关闭
- `action[i] = +1`: 房间 i 最大制热功率

**映射到实际功率**：
```python
Q_HVAC_i = action[i] * max_power  # W
```

---

### 4.2 观测空间

**结构**：
```python
observation = [
    T_1, ..., T_n,          # 房间温度 (n维)
    T_outdoor,              # 室外温度 (1维)
    GHI_1, ..., GHI_n,      # 太阳辐射 (n维)
    T_ground,               # 地面温度 (1维)
    Occ_1, ..., Occ_n       # 人员热负荷 (n维)
]
```

**总维度**：`3n + 2`

---

### 4.3 奖励函数

<augment_code_snippet path="bear/BEAR/Env/env_building.py" mode="EXCERPT">
````python
def default_reward_function(self, state, action, error, state_new):
    reward = -LA.norm(action, 2) * self.q_rate - LA.norm(error, 2) * self.error_rate
    return reward
````
</augment_code_snippet>

**公式**：
```
reward = -α·||action||₂ - β·||error||₂
```

其中：
- `α = gamma[0] × 24`: 能耗权重
- `β = gamma[1]`: 温度偏差权重
- `||action||₂ = √(Σ action_i²)`: 动作的 L2 范数
- `||error||₂ = √(Σ (T_i - T_target_i)²)`: 温度误差的 L2 范数

---

## 5. 关键代码解析

### 5.1 参数字典结构

```python
Parameter = {
    'OutTemp': np.array,        # (8760,) 室外温度
    'connectmap': np.array,     # (n, n+1) 连接矩阵
    'RCtable': np.array,        # (n, n+1) RC表
    'roomnum': int,             # 房间数量
    'weightcmap': np.array,     # (n, 5) 权重矩阵
    'target': np.array,         # (n,) 目标温度
    'gamma': tuple,             # (2,) 奖励权重
    'ghi': np.array,            # (8760,) 太阳辐射
    'GroundTemp': np.array,     # (8760,) 地面温度
    'Occupancy': np.array,      # (8760,) 人员占用
    'ACmap': np.array,          # (n,) AC映射
    'max_power': int,           # 最大功率
    'nonlinear': np.array,      # (n,) 非线性项
    'temp_range': tuple,        # (2,) 温度范围
    'spacetype': str,           # 'continuous' 或 'discrete'
    'time_resolution': int      # 时间分辨率（秒）
}
```

---

### 5.2 RC 网络构建

完整流程见 `utils_building.py` 中的 `Nfind_neighbor` 函数。

**步骤**：
1. 解析建筑几何信息
2. 识别相邻房间
3. 计算墙体面积
4. 计算热阻和热容
5. 构建连接矩阵

---

## 6. 数学公式推导

### 6.1 矩阵指数的计算

```python
A_d = expm(A * Δt)
```

**泰勒展开**：
```
e^(A·Δt) = I + A·Δt + (A·Δt)²/2! + (A·Δt)³/3! + ...
```

**实际计算**：使用 Padé 近似（`scipy.linalg.expm`）

---

### 6.2 B_d 矩阵推导

对于 `dX/dt = A·X + B·U`，精确解为：

```
X(t) = e^(A·t)·X(0) + ∫₀ᵗ e^(A·(t-τ))·B·U(τ) dτ
```

假设 U 在 [0, Δt] 内恒定：

```
X(Δt) = e^(A·Δt)·X(0) + [∫₀^Δt e^(A·τ) dτ]·B·U
```

其中：
```
∫₀^Δt e^(A·τ) dτ = A^(-1)·(e^(A·Δt) - I)
```

因此：
```
B_d = A^(-1)·(A_d - I)·B
```

---

## 📚 参考资料

1. **EnergyPlus Engineering Reference**
   - https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v23.1.0/EngineeringReference.pdf
   - 人员热负荷模型（Page 1299）

2. **BEAR 论文**
   - ACM e-Energy 2023
   - "BEAR: A Unified Framework for Evaluating Building Control Algorithms"

3. **RC 模型理论**
   - ISO 13790: Energy performance of buildings
   - ASHRAE Handbook - Fundamentals

4. **EPW 文件格式**
   - EnergyPlus Auxiliary Programs Documentation
   - https://bigladdersoftware.com/epx/docs/

---

## 🎯 总结

BEAR 的核心优势：

1. **物理真实性**：基于 RC 热力学模型，考虑多种物理因素
2. **数据真实性**：使用真实 EPW 气象数据和建筑几何信息
3. **数值稳定性**：使用矩阵指数精确离散化
4. **灵活性**：支持多种建筑类型和气候条件
5. **可扩展性**：支持自定义奖励函数和数据驱动模型

**关键技术点**：
- RC 网络建模
- 矩阵指数离散化
- 非线性人员热负荷
- EPW 数据处理
- 多房间热耦合

这使得 BEAR 成为建筑 HVAC 控制研究的理想平台！🏢🌡️

---

## 附录 A: 完整代码示例

### A.1 从零开始创建环境

```python
import numpy as np
from bear.BEAR.Utils.utils_building import ParameterGenerator
from bear.BEAR.Env.env_building import BuildingEnvReal

# 1. 生成参数
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

# 2. 创建环境
env = BuildingEnvReal(params)

# 3. 运行仿真
state, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # 随机动作
    next_state, reward, done, truncated, info = env.step(action)

    print(f"Step {step}:")
    print(f"  Room Temps: {next_state[:env.roomnum]}")
    print(f"  Outdoor Temp: {next_state[env.roomnum]}")
    print(f"  Reward: {reward:.2f}")

    if done:
        break
```

---

### A.2 自定义奖励函数

```python
def custom_reward_function(env, state, action, error, state_new):
    """
    自定义奖励函数示例

    考虑：
    1. 能耗成本（分时电价）
    2. 舒适度（温度偏差）
    3. 温度变化率（避免剧烈波动）
    """
    # 1. 能耗成本（假设峰谷电价）
    hour = env.epochs % 24
    if 8 <= hour < 22:  # 峰时
        electricity_price = 1.0
    else:  # 谷时
        electricity_price = 0.5

    energy_cost = np.linalg.norm(action, 2) * electricity_price

    # 2. 舒适度惩罚
    comfort_penalty = np.linalg.norm(error, 2)

    # 3. 温度变化率惩罚
    if len(env.statelist) > 0:
        temp_change = np.linalg.norm(state_new - env.statelist[-1][:env.roomnum])
        change_penalty = 0.1 * temp_change
    else:
        change_penalty = 0

    # 总奖励
    reward = -energy_cost - 10.0 * comfort_penalty - change_penalty

    return reward

# 使用自定义奖励函数
env = BuildingEnvReal(params, user_reward_function=custom_reward_function)
```

---

### A.3 数据驱动模型训练

```python
# 1. 收集数据
states = []
actions = []

state, _ = env.reset()
for step in range(8760):  # 一年
    action = expert_controller.get_action(state)  # 使用专家控制器
    next_state, reward, done, _, _ = env.step(action)

    states.append(state[:env.roomnum])
    actions.append(action)

    state = next_state
    if done:
        break

# 2. 训练数据驱动模型
env.train(np.array(states), np.array(actions))

# 3. 现在 env 使用学习到的 A_d 和 B_d 矩阵
print("Data-driven model trained!")
print(f"A_d shape: {env.A_d.shape}")
print(f"B_d shape: {env.B_d.shape}")
```

---

## 附录 B: 可视化工具

### B.1 温度轨迹可视化

```python
import matplotlib.pyplot as plt

def visualize_temperature_trajectory(env, num_steps=288):
    """可视化24小时温度轨迹"""
    state, _ = env.reset()

    room_temps = []
    outdoor_temps = []
    actions_list = []
    rewards_list = []

    for step in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

        room_temps.append(next_state[:env.roomnum])
        outdoor_temps.append(next_state[env.roomnum])
        actions_list.append(action)
        rewards_list.append(reward)

        state = next_state
        if done:
            break

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. 温度轨迹
    room_temps = np.array(room_temps)
    for i in range(env.roomnum):
        axes[0].plot(room_temps[:, i], label=f'Room {i+1}')
    axes[0].plot(outdoor_temps, 'k--', label='Outdoor', linewidth=2)
    axes[0].axhline(y=env.target[0], color='r', linestyle=':', label='Target')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Room Temperatures')
    axes[0].legend()
    axes[0].grid(True)

    # 2. HVAC 动作
    actions_array = np.array(actions_list)
    for i in range(env.roomnum):
        axes[1].plot(actions_array[:, i], label=f'Room {i+1}')
    axes[1].set_ylabel('HVAC Power (normalized)')
    axes[1].set_title('HVAC Actions')
    axes[1].legend()
    axes[1].grid(True)

    # 3. 奖励
    axes[2].plot(rewards_list)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Reward over Time')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('temperature_trajectory.png', dpi=300)
    plt.show()

# 使用
visualize_temperature_trajectory(env)
```

---

### B.2 RC 网络可视化

```python
import networkx as nx

def visualize_rc_network(env):
    """可视化 RC 网络拓扑"""
    G = nx.Graph()

    # 添加节点
    for i in range(env.roomnum):
        G.add_node(f"Room_{i}", type='room')
    G.add_node("Outdoor", type='outdoor')
    G.add_node("Ground", type='ground')

    # 添加边（基于 connectmap）
    for i in range(env.roomnum):
        for j in range(i+1, env.roomnum):
            if env.connectmap[i][j] == 1:
                R_ij = 1.0 / env.RCtable[i][j] if env.RCtable[i][j] != 0 else np.inf
                G.add_edge(f"Room_{i}", f"Room_{j}",
                          weight=R_ij, label=f"R={R_ij:.2f}")

        # 与室外的连接
        if env.connectmap[i][-1] == 1:
            R_out = 1.0 / env.RCtable[i][-1] if env.RCtable[i][-1] != 0 else np.inf
            G.add_edge(f"Room_{i}", "Outdoor",
                      weight=R_out, label=f"R={R_out:.2f}")

    # 绘图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # 节点颜色
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'room':
            node_colors.append('lightblue')
        elif G.nodes[node]['type'] == 'outdoor':
            node_colors.append('orange')
        else:
            node_colors.append('brown')

    nx.draw(G, pos, node_color=node_colors, node_size=1000,
            with_labels=True, font_size=10, font_weight='bold')

    # 边标签
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title('RC Network Topology')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('rc_network.png', dpi=300)
    plt.show()

# 使用
visualize_rc_network(env)
```

---

### B.3 能耗分析

```python
def analyze_energy_consumption(env, policy, num_days=7):
    """分析一周的能耗"""
    steps_per_day = 24 * 3600 // env.timestep
    total_steps = num_days * steps_per_day

    daily_energy = []
    daily_comfort = []

    state, _ = env.reset()
    day_energy = 0
    day_comfort = 0

    for step in range(total_steps):
        action = policy(state)  # 使用策略
        next_state, reward, done, _, info = env.step(action)

        # 累积能耗
        energy = np.sum(np.abs(action)) * env.maxpower * env.timestep / 3600  # kWh
        day_energy += energy

        # 累积舒适度误差
        temps = next_state[:env.roomnum]
        comfort_error = np.mean(np.abs(temps - env.target))
        day_comfort += comfort_error

        # 每天统计
        if (step + 1) % steps_per_day == 0:
            daily_energy.append(day_energy)
            daily_comfort.append(day_comfort / steps_per_day)
            day_energy = 0
            day_comfort = 0

        state = next_state
        if done:
            break

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    days = np.arange(1, len(daily_energy) + 1)

    axes[0].bar(days, daily_energy, color='steelblue')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Energy (kWh)')
    axes[0].set_title('Daily Energy Consumption')
    axes[0].grid(True, axis='y')

    axes[1].bar(days, daily_comfort, color='coral')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Avg Temperature Error (°C)')
    axes[1].set_title('Daily Comfort Level')
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('energy_analysis.png', dpi=300)
    plt.show()

    # 打印统计
    print(f"Total Energy: {sum(daily_energy):.2f} kWh")
    print(f"Avg Daily Energy: {np.mean(daily_energy):.2f} kWh")
    print(f"Avg Comfort Error: {np.mean(daily_comfort):.2f} °C")

# 使用
analyze_energy_consumption(env, lambda s: env.action_space.sample())
```

---

## 附录 C: 常见问题解答

### Q1: 为什么使用矩阵指数而不是简单的欧拉法？

**A**: 矩阵指数提供精确解，避免数值不稳定：

```python
# 欧拉法（一阶近似）
X_new = X + dt * (A @ X + B @ U)  # 可能不稳定

# 矩阵指数（精确解）
X_new = expm(A * dt) @ X + B_d @ U  # 数值稳定
```

**对比**：
- 欧拉法：误差 O(dt²)，可能发散
- 矩阵指数：精确到机器精度，始终稳定

---

### Q2: 如何选择时间分辨率？

**A**: 权衡计算成本和精度：

| 时间分辨率 | 数据点/年 | 计算时间 | 适用场景 |
|-----------|----------|---------|---------|
| 3600s (1h) | 8,760 | 快 | 快速原型 |
| 1800s (30min) | 17,520 | 中等 | 一般研究 |
| 900s (15min) | 35,040 | 慢 | 高精度研究 |
| 300s (5min) | 105,120 | 很慢 | 实时控制仿真 |

**建议**：
- 算法开发：1小时
- 论文实验：30分钟
- 实际部署验证：5-15分钟

---

### Q3: 如何处理多建筑类型的泛化？

**A**: 使用建筑特征作为额外输入：

```python
# 方法1: 增强状态空间
building_features = [
    env.roomnum,           # 房间数量
    np.sum(env.Windowtable),  # 总窗户面积
    np.mean(env.RCtable),  # 平均热阻
]
augmented_state = np.concatenate([state, building_features])

# 方法2: 条件策略
class BuildingAwarePolicy:
    def __init__(self, building_encoder, policy_network):
        self.building_encoder = building_encoder
        self.policy_network = policy_network

    def forward(self, state, building_params):
        building_embedding = self.building_encoder(building_params)
        combined = torch.cat([state, building_embedding], dim=-1)
        action = self.policy_network(combined)
        return action
```

---

### Q4: 如何验证模型的物理真实性？

**A**: 与 EnergyPlus 对比：

```python
# 1. 在 BEAR 中运行
bear_temps = run_bear_simulation(env, actions)

# 2. 在 EnergyPlus 中运行相同场景
energyplus_temps = run_energyplus_simulation(building, weather, actions)

# 3. 计算误差
rmse = np.sqrt(np.mean((bear_temps - energyplus_temps)**2))
print(f"RMSE: {rmse:.2f} °C")

# 4. 可视化对比
plt.plot(bear_temps, label='BEAR')
plt.plot(energyplus_temps, label='EnergyPlus')
plt.legend()
plt.show()
```

---

### Q5: 如何加速训练？

**A**: 多种策略：

```python
# 1. 并行环境
from tianshou.env import SubprocVectorEnv

envs = SubprocVectorEnv([
    lambda: BearEnvWrapper(building_type='OfficeSmall')
    for _ in range(8)
])

# 2. 缩短回合长度
env = BearEnvWrapper(
    building_type='OfficeSmall',
    episode_length=288  # 24小时而非全年
)

# 3. 降低时间分辨率
env = BearEnvWrapper(
    building_type='OfficeSmall',
    time_resolution=3600  # 1小时而非5分钟
)

# 4. 使用 GPU
policy = DiffusionOPT(..., device='cuda:0')
```

---

## 附录 D: 扩展阅读

### D.1 相关论文

1. **BEAR 原始论文**
   ```
   @inproceedings{bear2023,
     title={BEAR: A Unified Framework for Evaluating Building Control Algorithms},
     author={...},
     booktitle={ACM e-Energy},
     year={2023}
   }
   ```

2. **RC 模型理论**
   - Ramallo-González, A. P., et al. "Lumped parameter models for building thermal modelling." Energy and Buildings (2013).

3. **建筑 MPC 控制**
   - Oldewurtel, F., et al. "Use of model predictive control and weather forecasts for energy efficient building climate control." Energy and Buildings (2012).

---

### D.2 相关工具

1. **EnergyPlus**: 详细建筑能源仿真
   - https://energyplus.net/

2. **OpenStudio**: EnergyPlus 的图形界面
   - https://openstudio.net/

3. **pvlib**: 太阳能数据处理
   - https://pvlib-python.readthedocs.io/

4. **Sinergym**: 另一个建筑 RL 环境
   - https://github.com/ugr-sail/sinergym

---

### D.3 数据资源

1. **EPW 气象数据**
   - https://energyplus.net/weather
   - 全球 2100+ 个位置

2. **建筑原型**
   - DOE Commercial Reference Buildings
   - ASHRAE 90.1 Prototype Buildings

3. **真实建筑数据**
   - Building Data Genome Project
   - https://github.com/buds-lab/building-data-genome-project-2

---

## 🎓 结语

BEAR 是一个强大而灵活的建筑 HVAC 控制仿真平台。通过本文档，你应该能够：

✅ 理解 RC 热力学模型的物理原理
✅ 掌握 BEAR 的数据输入和处理流程
✅ 理解状态空间方程的构建和离散化
✅ 使用 BEAR 进行强化学习研究
✅ 自定义奖励函数和扩展功能
✅ 可视化和分析仿真结果

**下一步建议**：
1. 运行附录 A 中的代码示例
2. 尝试不同的建筑类型和气候条件
3. 实现自己的控制算法
4. 与 EnergyPlus 对比验证
5. 发表你的研究成果！

祝研究顺利！🏢🌡️🚀

