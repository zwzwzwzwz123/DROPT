# BEAR 建筑模拟环境集成方案

## 📋 目录
1. [项目概述](#项目概述)
2. [BEAR 代码分析](#bear-代码分析)
3. [DROPT 环境接口分析](#dropt-环境接口分析)
4. [集成架构设计](#集成架构设计)
5. [实现步骤](#实现步骤)
6. [使用示例](#使用示例)

---

## 1. 项目概述

### 1.1 BEAR 项目简介
**BEAR** (Building Environment for Control And Reinforcement Learning) 是一个基于物理原理的建筑环境模拟器，专为控制和强化学习设计。

**核心特性**：
- ✅ **16种建筑类型**：办公楼、医院、酒店、学校、仓库等
- ✅ **19个地理位置**：覆盖全球不同气候区
- ✅ **物理建模**：基于RC热力学模型（电阻-电容网络）
- ✅ **真实天气数据**：EPW格式气象文件（8760小时/年）
- ✅ **OpenAI Gym接口**：标准RL环境接口
- ✅ **可定制奖励函数**：支持用户自定义奖励
- ✅ **MPC控制器**：内置模型预测控制基线

**GitHub**: https://github.com/chz056/BEAR

### 1.2 DROPT 项目简介
**DROPT** 是一个基于扩散模型的强化学习框架，当前应用于数据中心空调优化。

**核心特性**：
- ✅ **扩散模型Actor**：使用DDPM生成动作
- ✅ **双Q网络Critic**：减少价值过估计
- ✅ **行为克隆支持**：可利用专家数据加速训练
- ✅ **Tianshou框架**：高效的RL训练流程
- ✅ **模块化设计**：易于扩展到新环境

---

## 2. BEAR 代码分析

### 2.1 核心文件结构
```
bear/BEAR/
├── Env/
│   └── env_building.py          # 建筑环境主类 (433行)
├── Controller/
│   └── MPC_Controller.py        # MPC控制器 (172行)
├── Utils/
│   └── utils_building.py        # 工具函数 (830行)
├── Customize/
│   └── reward_functions.py      # 自定义奖励函数
├── Data/                        # 建筑和天气数据
│   ├── *.table.htm              # 16种建筑的几何数据
│   └── *.epw                    # 19个城市的天气数据
└── examples/
    └── quickstart.py            # 快速开始示例
```

### 2.2 环境接口分析

#### 2.2.1 初始化参数
```python
from BEAR.Utils.utils_building import ParameterGenerator
from BEAR.Env.env_building import BuildingEnvReal

# 生成环境参数
Parameter = ParameterGenerator(
    Building='OfficeSmall',      # 建筑类型
    Weather='Hot_Dry',           # 气候类型
    Location='Tucson',           # 地理位置
    max_power=8000,              # HVAC最大功率 (W)
    time_reso=3600,              # 时间分辨率 (秒)
    reward_gamma=(0.001, 0.999), # [能耗权重, 温度权重]
    target=22,                   # 目标温度 (°C)
    temp_range=(-40, 40),        # 温度范围
    spacetype='continuous'       # 连续动作空间
)

# 创建环境
env = BuildingEnvReal(Parameter)
```

#### 2.2.2 状态空间 (Observation Space)
```python
# 状态维度：3*roomnum + 3
# 组成：
# - 各房间温度 (roomnum)
# - 室外温度 (1)
# - 全局水平辐照度 GHI (roomnum)
# - 地面温度 (1)
# - 人员热负荷 (roomnum)

# 示例：6个房间的办公楼
# 状态维度 = 6 + 1 + 6 + 1 + 6 = 20
```

#### 2.2.3 动作空间 (Action Space)
```python
# 动作维度：roomnum
# 每个房间的HVAC功率：[-1, 1]
# - 负值：制冷 (cooling)
# - 正值：制热 (heating)
# - 归一化到 [-1, 1]，实际功率 = action * max_power

# 示例：6个房间
# 动作维度 = 6
# action = [-0.5, -0.3, 0.0, -0.2, -0.4, -0.1]
# 实际功率 = action * 8000W
```

#### 2.2.4 奖励函数
```python
# 默认奖励函数
def default_reward_function(self, state, action, error, state_new):
    reward = 0
    # 能耗惩罚
    reward -= LA.norm(action, 2) * self.q_rate
    # 温度偏差惩罚
    reward -= LA.norm(error, 2) * self.error_rate
    return reward

# error = (当前温度 - 目标温度) * AC_map
# AC_map: 标记哪些房间有空调
```

#### 2.2.5 环境动态
```python
# 基于RC热力学模型
# 状态更新方程：
# X_{t+1} = A_d @ X_t + B_d @ Y_t

# 其中：
# - A_d: 离散化系统矩阵 (roomnum x roomnum)
# - B_d: 输入矩阵 (roomnum x (4+roomnum+1))
# - Y_t: 输入向量 [人员热负荷, 地面温度, 室外温度, HVAC功率, GHI]
```

### 2.3 关键类和方法

#### BuildingEnvReal 类
```python
class BuildingEnvReal(gym.Env):
    """建筑环境类"""
    
    def __init__(self, Parameter: Dict[str, Any], 
                 user_reward_function=None,
                 reward_breakdown_keys=None):
        """初始化环境"""
        # 解析参数
        self.OutTemp = Parameter['OutTemp']      # 室外温度序列
        self.roomnum = Parameter['roomnum']      # 房间数量
        self.target = Parameter['target']        # 目标温度
        self.gamma = Parameter['gamma']          # 奖励权重
        self.ghi = Parameter['ghi']              # 太阳辐照度
        self.Occupancy = Parameter['Occupancy']  # 人员占用率
        # ... 更多参数
        
        # 定义动作和状态空间
        self.action_space = gym.spaces.Box(...)
        self.observation_space = gym.spaces.Box(...)
        
        # 计算系统矩阵
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = LA.inv(Amatrix) @ (self.A_d - I) @ Bmatrix
    
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        self.epochs = 0
        T_initial = self.target  # 初始温度
        # 构造初始状态
        self.state = np.concatenate([
            T_initial,                    # 房间温度
            self.OutTemp[0],              # 室外温度
            self.ghi[0],                  # GHI
            self.GroundTemp[0],           # 地面温度
            self.Occupower/1000           # 人员热负荷
        ])
        return self.state, {}
    
    def step(self, action: np.ndarray):
        """执行一步"""
        # 状态更新
        X_new = self.A_d @ X + self.B_d @ Y
        
        # 计算奖励
        error = X_new * self.acmap - self.target * self.acmap
        reward = self.reward_function(self.state, action, error, X_new)
        
        # 检查是否结束
        done = (self.epochs >= len(self.OutTemp) - 1)
        
        return self.state, reward, done, done, info
```

#### ParameterGenerator 函数
```python
def ParameterGenerator(
    Building: str,              # 建筑类型或文件路径
    Weather: str,               # 气候类型或EPW文件路径
    Location: str,              # 地理位置
    U_Wall: List[float],        # 墙体热传导系数
    max_power: int = 8000,      # HVAC最大功率
    time_reso: int = 3600,      # 时间分辨率
    reward_gamma: Tuple = (0.001, 0.999),  # 奖励权重
    target: float = 22,         # 目标温度
    temp_range: Tuple = (-40, 40),  # 温度范围
    spacetype: str = 'continuous',  # 动作空间类型
    root: str = 'BEAR/Data/'    # 数据根目录
) -> Dict[str, Any]:
    """生成环境参数字典"""
    
    # 1. 解析建筑类型
    Building_dic = {
        'OfficeSmall': ('ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm', [...]),
        'Hospital': (...),
        # ... 16种建筑
    }
    
    # 2. 解析天气类型
    weather_dic = {
        'Hot_Dry': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',
        'Cold_Humid': 'USA_MN_Rochester.Intl.AP.726440_TMY3.epw',
        # ... 16种气候
    }
    
    # 3. 读取建筑几何信息
    Layerall, roomnum, buildall = Getroominfor(filename)
    
    # 4. 读取天气数据
    data = pvlib.iotools.read_epw(weatherfile)
    outtempdatanew = interpolate_temperature(data, time_reso)
    solardatanew = interpolate_ghi(data, time_reso)
    
    # 5. 计算RC网络参数
    dicRoom, Rtable, Ctable, Windowtable = Nfind_neighbor(...)
    
    # 6. 返回参数字典
    return {
        'OutTemp': outtempdatanew,
        'roomnum': roomnum,
        'connectmap': connectmap,
        'RCtable': RCtable,
        'target': target,
        'gamma': reward_gamma,
        'ghi': solardatanew,
        'GroundTemp': groundtemp,
        'Occupancy': occupancy_schedule,
        'ACmap': AC_map,
        'max_power': max_power,
        'nonlinear': nonlinear_term,
        'temp_range': temp_range,
        'spacetype': spacetype,
        'time_resolution': time_reso
    }
```

---

## 3. DROPT 环境接口分析

### 3.1 当前环境结构 (DataCenterEnv)

```python
class DataCenterEnv(gym.Env):
    """数据中心空调优化环境"""
    
    def __init__(
        self,
        num_crac_units: int = 4,
        target_temp: float = 24.0,
        temp_tolerance: float = 2.0,
        time_step: float = 5.0,
        episode_length: int = 288,
        energy_weight: float = 1.0,
        temp_weight: float = 10.0,
        violation_penalty: float = 100.0,
        use_real_weather: bool = False,
        weather_file: str = None,
        workload_file: str = None,
    ):
        # 状态空间：[T_in, T_out, H_in, IT_load, T_supply_1, ..., T_supply_n, reward_last]
        self.state_dim = 4 + num_crac_units + 1
        
        # 动作空间：[T_set_1, fan_speed_1, ..., T_set_n, fan_speed_n]
        self.action_dim = num_crac_units * 2
        
        # 子模块
        self.thermal_model = ThermalModel(...)
        self.expert_controller = ExpertController(...)
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 初始化物理状态
        self.T_in = self.target_temp + random()
        self.T_out = random_outdoor_temp()
        # ...
        return self._get_state()
    
    def step(self, action: np.ndarray):
        """执行一步"""
        # 动作反归一化
        T_set, fan_speed = self._denormalize_action(action)
        
        # 获取专家动作
        expert_action = self.expert_controller.get_action(...)
        
        # 更新环境动态
        next_T_in, next_H_in, next_T_supply, energy = self.thermal_model.step(...)
        
        # 计算奖励
        reward, info = self._compute_reward(...)
        
        return next_state, reward, done, info
```

### 3.2 环境创建接口

```python
def make_datacenter_env(training_num: int = 1, test_num: int = 1, **kwargs):
    """创建数据中心环境"""
    from tianshou.env import DummyVectorEnv
    
    env = DataCenterEnv(**kwargs)
    
    train_envs = DummyVectorEnv([
        lambda: DataCenterEnv(**kwargs) for _ in range(training_num)
    ])
    
    test_envs = DummyVectorEnv([
        lambda: DataCenterEnv(**kwargs) for _ in range(test_num)
    ])
    
    return env, train_envs, test_envs
```

---

## 4. 集成架构设计

### 4.1 设计目标

1. **最小侵入性**：不修改BEAR原始代码
2. **接口兼容性**：符合DROPT的环境接口规范
3. **功能完整性**：保留BEAR的所有特性
4. **易用性**：简化环境创建流程
5. **可扩展性**：支持自定义配置

### 4.2 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    DROPT Training Pipeline                   │
│                  (main_building.py)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              BearEnvWrapper (适配器层)                       │
│  - 状态空间映射                                              │
│  - 动作空间映射                                              │
│  - 奖励函数适配                                              │
│  - 专家控制器集成                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              BuildingEnvReal (BEAR原始环境)                  │
│  - RC热力学模型                                              │
│  - 真实天气数据                                              │
│  - 建筑几何信息                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 核心组件设计

#### 4.3.1 BearEnvWrapper (适配器类)
```python
class BearEnvWrapper(gym.Env):
    """BEAR环境适配器，使其兼容DROPT接口"""
    
    def __init__(
        self,
        building_type: str = 'OfficeSmall',
        weather_type: str = 'Hot_Dry',
        location: str = 'Tucson',
        target_temp: float = 22.0,
        temp_tolerance: float = 2.0,
        max_power: int = 8000,
        time_resolution: int = 3600,
        energy_weight: float = 0.001,
        temp_weight: float = 0.999,
        episode_length: int = None,
        expert_type: str = 'mpc',
        **kwargs
    ):
        """初始化适配器"""
        # 1. 生成BEAR参数
        self.bear_params = ParameterGenerator(
            Building=building_type,
            Weather=weather_type,
            Location=location,
            target=target_temp,
            reward_gamma=(energy_weight, temp_weight),
            max_power=max_power,
            time_reso=time_resolution,
            **kwargs
        )
        
        # 2. 创建BEAR环境
        self.bear_env = BuildingEnvReal(self.bear_params)
        
        # 3. 适配状态和动作空间
        self.observation_space = self._adapt_observation_space()
        self.action_space = self._adapt_action_space()
        
        # 4. 创建专家控制器
        self.expert_controller = self._create_expert_controller(expert_type)
        
        # 5. 设置回合长度
        self.episode_length = episode_length or len(self.bear_params['OutTemp'])
        self.current_step = 0
    
    def reset(self):
        """重置环境"""
        state, info = self.bear_env.reset()
        self.current_step = 0
        return self._adapt_state(state), info
    
    def step(self, action):
        """执行一步"""
        # 1. 适配动作
        bear_action = self._adapt_action(action)
        
        # 2. 执行BEAR环境
        state, reward, done, truncated, info = self.bear_env.step(bear_action)
        
        # 3. 获取专家动作
        expert_action = self.expert_controller.get_action(state, self.bear_env)
        info['expert_action'] = expert_action
        
        # 4. 适配状态和奖励
        adapted_state = self._adapt_state(state)
        adapted_reward = self._adapt_reward(reward, state, info)
        
        # 5. 检查回合结束
        self.current_step += 1
        if self.episode_length and self.current_step >= self.episode_length:
            done = True
        
        return adapted_state, adapted_reward, done, info
```

#### 4.3.2 状态空间映射
```python
def _adapt_observation_space(self):
    """适配状态空间"""
    # BEAR状态：[房间温度(n), 室外温度(1), GHI(n), 地面温度(1), 人员热负荷(n)]
    # DROPT期望：标准化的Box空间
    
    roomnum = self.bear_params['roomnum']
    state_dim = 3 * roomnum + 3
    
    # 定义状态范围
    temp_min, temp_max = self.bear_params['temp_range']
    
    low = np.array([temp_min] * (roomnum + 1) +  # 温度
                   [0] * roomnum +                 # GHI
                   [temp_min] +                    # 地面温度
                   [0] * roomnum)                  # 人员热负荷
    
    high = np.array([temp_max] * (roomnum + 1) +
                    [1000] * roomnum +
                    [temp_max] +
                    [1000] * roomnum)
    
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)

def _adapt_state(self, bear_state):
    """适配状态向量"""
    # BEAR状态已经是正确格式，可能需要归一化
    return bear_state.astype(np.float32)
```

#### 4.3.3 动作空间映射
```python
def _adapt_action_space(self):
    """适配动作空间"""
    # BEAR动作：每个房间的HVAC功率 [-1, 1]
    # DROPT期望：归一化的Box空间 [-1, 1]

    roomnum = self.bear_params['roomnum']

    return gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(roomnum,),
        dtype=np.float32
    )

def _adapt_action(self, dropt_action):
    """适配动作向量"""
    # DROPT动作已经是[-1, 1]，直接传递给BEAR
    return dropt_action
```

#### 4.3.4 奖励函数适配
```python
def _adapt_reward(self, bear_reward, state, info):
    """适配奖励函数"""
    # BEAR奖励：-能耗惩罚 - 温度偏差惩罚
    # 可以保持不变，或添加额外的惩罚项

    # 选项1：直接使用BEAR奖励
    return bear_reward

    # 选项2：添加温度越界惩罚（类似DataCenterEnv）
    zone_temps = info['zone_temperature']
    target = self.bear_params['target']
    tolerance = self.temp_tolerance

    violation_penalty = 0.0
    for temp in zone_temps:
        if temp < target - tolerance or temp > target + tolerance:
            violation_penalty += 100.0

    return bear_reward - violation_penalty
```

#### 4.3.5 专家控制器集成
```python
def _create_expert_controller(self, expert_type):
    """创建专家控制器"""
    if expert_type == 'mpc':
        # 使用BEAR内置的MPC控制器
        from bear.BEAR.Controller.MPC_Controller import MPCAgent
        return BearMPCWrapper(self.bear_env, self.bear_params)

    elif expert_type == 'rule_based':
        # 创建基于规则的控制器
        return BearRuleBasedController(self.bear_params)

    elif expert_type == 'pid':
        # 创建PID控制器（需要实现）
        return BearPIDController(self.bear_params)

    else:
        raise ValueError(f"Unknown expert type: {expert_type}")

class BearMPCWrapper:
    """BEAR MPC控制器包装器"""

    def __init__(self, bear_env, bear_params):
        from bear.BEAR.Controller.MPC_Controller import MPCAgent
        self.mpc = MPCAgent(
            environment=bear_env,
            gamma=bear_params['gamma'],
            planning_steps=1
        )

    def get_action(self, state, env):
        """获取专家动作"""
        action, _ = self.mpc.predict(env)
        # 归一化到[-1, 1]
        return action
```

### 4.4 环境创建接口

```python
def make_building_env(
    building_type: str = 'OfficeSmall',
    weather_type: str = 'Hot_Dry',
    location: str = 'Tucson',
    training_num: int = 1,
    test_num: int = 1,
    **kwargs
):
    """创建建筑环境（兼容DROPT接口）"""
    from tianshou.env import DummyVectorEnv

    # 创建单个环境实例
    env = BearEnvWrapper(
        building_type=building_type,
        weather_type=weather_type,
        location=location,
        **kwargs
    )

    # 创建训练环境向量
    train_envs = DummyVectorEnv([
        lambda: BearEnvWrapper(
            building_type=building_type,
            weather_type=weather_type,
            location=location,
            **kwargs
        ) for _ in range(training_num)
    ])

    # 创建测试环境向量
    test_envs = DummyVectorEnv([
        lambda: BearEnvWrapper(
            building_type=building_type,
            weather_type=weather_type,
            location=location,
            **kwargs
        ) for _ in range(test_num)
    ])

    return env, train_envs, test_envs
```

---

## 5. 实现步骤

### 5.1 文件创建清单

需要创建以下新文件：

1. **`env/building_env_wrapper.py`** (约400行)
   - `BearEnvWrapper` 类
   - 状态/动作/奖励适配方法
   - 专家控制器包装器

2. **`env/building_expert_controller.py`** (约300行)
   - `BearMPCWrapper` 类
   - `BearPIDController` 类
   - `BearRuleBasedController` 类

3. **`env/building_config.py`** (约200行)
   - 预定义建筑配置
   - 训练超参数推荐

4. **`main_building.py`** (约300行)
   - 建筑环境训练主程序
   - 参数解析
   - 训练流程

5. **`scripts/test_building_env.py`** (约200行)
   - 环境测试脚本
   - 功能验证

6. **`docs/BEAR_INTEGRATION_GUIDE.md`** (约150行)
   - 使用指南
   - 示例代码

### 5.2 修改现有文件

需要修改以下文件：

1. **`env/__init__.py`**
   - 添加 `from .building_env_wrapper import BearEnvWrapper, make_building_env`

2. **`requirements.txt`** 或创建 **`bear_requirements.txt`**
   - 添加BEAR依赖：`pvlib`, `cvxpy`, `scikit-learn`

### 5.3 详细实现步骤

#### 步骤1：安装BEAR依赖
```bash
# 安装BEAR所需的额外依赖
pip install pvlib scikit-learn cvxpy
```

#### 步骤2：创建适配器类
创建 `env/building_env_wrapper.py`，实现：
- `BearEnvWrapper` 主类
- 状态空间适配
- 动作空间适配
- 奖励函数适配
- 专家控制器接口

#### 步骤3：创建专家控制器
创建 `env/building_expert_controller.py`，实现：
- MPC控制器包装器
- PID控制器（参考 `expert_controller.py`）
- 基于规则的控制器

#### 步骤4：创建配置文件
创建 `env/building_config.py`，定义：
- 常用建筑类型配置
- 训练超参数推荐
- 环境参数模板

#### 步骤5：创建训练脚本
创建 `main_building.py`，实现：
- 参数解析（扩展自 `main_datacenter.py`）
- 环境创建
- 网络初始化
- 训练循环

#### 步骤6：测试和验证
创建 `scripts/test_building_env.py`，测试：
- 环境创建
- 状态/动作空间
- 专家控制器
- 训练流程

---

## 6. 使用示例

### 6.1 基本使用

```python
from env.building_env_wrapper import make_building_env

# 创建小型办公楼环境
env, train_envs, test_envs = make_building_env(
    building_type='OfficeSmall',
    weather_type='Hot_Dry',
    location='Tucson',
    target_temp=22.0,
    temp_tolerance=2.0,
    max_power=8000,
    time_resolution=3600,  # 1小时
    energy_weight=0.001,
    temp_weight=0.999,
    training_num=4,
    test_num=2
)

# 测试环境
state, info = env.reset()
print(f"状态维度: {state.shape}")
print(f"动作维度: {env.action_space.shape}")

for step in range(10):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(f"Step {step}: Reward={reward:.2f}, Done={done}")
    if done:
        break
```

### 6.2 训练示例

```bash
# 快速训练（行为克隆模式）
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --location Tucson \
    --bc-coef \
    --expert-type mpc \
    --epoch 50000 \
    --batch-size 256 \
    --n-timesteps 5 \
    --device cuda:0

# 高性能训练（策略梯度模式）
python main_building.py \
    --building-type Hospital \
    --weather-type Cold_Humid \
    --location Rochester \
    --epoch 200000 \
    --batch-size 512 \
    --n-timesteps 8 \
    --gamma 0.99 \
    --device cuda:0
```

### 6.3 多建筑类型对比

```python
# 测试不同建筑类型
building_types = ['OfficeSmall', 'Hospital', 'SchoolPrimary', 'Warehouse']

for building in building_types:
    env, _, _ = make_building_env(
        building_type=building,
        weather_type='Hot_Dry',
        location='Tucson'
    )

    print(f"\n{building}:")
    print(f"  房间数: {env.bear_params['roomnum']}")
    print(f"  状态维度: {env.observation_space.shape}")
    print(f"  动作维度: {env.action_space.shape}")
```

### 6.4 自定义奖励函数

```python
def custom_reward_function(self, state, action, error, state_new):
    """自定义奖励函数"""
    reward = 0

    # 能耗惩罚
    energy_penalty = LA.norm(action, 2) * self.q_rate
    reward -= energy_penalty

    # 温度偏差惩罚
    temp_penalty = LA.norm(error, 2) * self.error_rate
    reward -= temp_penalty

    # 舒适度奖励（温度在目标范围内）
    comfort_bonus = 0
    for temp in state_new:
        if 20 <= temp <= 24:
            comfort_bonus += 1.0
    reward += comfort_bonus

    # 记录奖励分解
    self._reward_breakdown['energy'] = -energy_penalty
    self._reward_breakdown['temperature'] = -temp_penalty
    self._reward_breakdown['comfort'] = comfort_bonus

    return reward

# 使用自定义奖励函数
from bear.BEAR.Env.env_building import BuildingEnvReal
from bear.BEAR.Utils.utils_building import ParameterGenerator

Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson')
env = BuildingEnvReal(
    Parameter,
    user_reward_function=custom_reward_function,
    reward_breakdown_keys=['energy', 'temperature', 'comfort']
)
```

---

## 7. 关键技术细节

### 7.1 状态空间设计

**BEAR原始状态**：
- 房间温度 (n维)
- 室外温度 (1维)
- 全局水平辐照度 GHI (n维)
- 地面温度 (1维)
- 人员热负荷 (n维)

**适配策略**：
- 保持原始状态结构
- 添加归一化（可选）
- 添加历史信息（可选）

### 7.2 动作空间设计

**BEAR原始动作**：
- 每个房间的HVAC功率：[-1, 1]
- 负值=制冷，正值=制热

**适配策略**：
- 直接使用BEAR的动作空间
- 与DROPT的归一化动作空间完全兼容

### 7.3 奖励函数设计

**BEAR默认奖励**：
```
reward = -α * ||action||₂ - β * ||error||₂
```

**可选增强**：
1. 添加温度越界惩罚
2. 添加舒适度奖励
3. 添加能效比奖励
4. 添加峰值功率惩罚

### 7.4 专家控制器设计

**MPC控制器**：
- 使用BEAR内置的 `MPCAgent`
- 基于凸优化求解最优控制序列
- 需要 `cvxpy` 库

**PID控制器**：
- 参考 `env/expert_controller.py` 中的实现
- 为每个房间独立设计PID控制器
- 考虑房间间的耦合

**基于规则的控制器**：
- 简单的if-else规则
- 适合作为baseline

---

## 8. 预期效果

### 8.1 环境特性

| 特性 | BEAR环境 | DataCenter环境 |
|------|----------|----------------|
| 状态维度 | 3n+3 (n=房间数) | 4+m+1 (m=CRAC数) |
| 动作维度 | n | 2m |
| 时间分辨率 | 可配置 (默认1小时) | 5分钟 |
| 回合长度 | 8760步 (1年) | 288步 (24小时) |
| 物理模型 | RC热力学模型 | 简化热力学模型 |
| 真实数据 | EPW天气文件 | 可选 |

### 8.2 训练性能

**预期训练时间**（GPU）：
- 快速演示（BC模式，1000 epochs）：~10分钟
- 标准训练（BC模式，50000 epochs）：~2小时
- 高性能训练（PG模式，200000 epochs）：~8小时

**预期性能提升**：
- 相比随机策略：节能 30-50%
- 相比MPC基线：节能 5-15%
- 温度控制精度：±0.5°C

### 8.3 应用场景

1. **办公楼能源管理**
   - 建筑类型：OfficeSmall/Medium/Large
   - 优化目标：节能 + 舒适度

2. **医院温度控制**
   - 建筑类型：Hospital
   - 优化目标：精确温度控制

3. **学校HVAC调度**
   - 建筑类型：SchoolPrimary/Secondary
   - 优化目标：考虑占用率的动态调度

4. **仓库温度管理**
   - 建筑类型：Warehouse
   - 优化目标：最小化能耗

---

## 9. 后续扩展

### 9.1 短期扩展（1-2周）

1. **多目标优化**
   - 能耗 vs 舒适度的Pareto前沿
   - 可视化权衡曲线

2. **迁移学习**
   - 在一个建筑上训练，迁移到另一个建筑
   - 跨气候区域的迁移

3. **鲁棒性增强**
   - 添加模型不确定性
   - 添加传感器噪声

### 9.2 中期扩展（1-2月）

1. **数据驱动建模**
   - 使用BEAR的 `train()` 方法
   - 从真实数据学习系统矩阵

2. **分布式控制**
   - 多建筑协同优化
   - 区域能源管理

3. **实时控制**
   - 降低时间分辨率（5分钟）
   - 在线学习和适应

### 9.3 长期扩展（3-6月）

1. **真实建筑部署**
   - 与BMS系统集成
   - 实际建筑测试

2. **经济优化**
   - 考虑电价
   - 需求响应

3. **可再生能源集成**
   - 太阳能发电
   - 储能系统

---

## 10. 总结

### 10.1 集成优势

✅ **丰富的建筑类型**：16种建筑 × 19个地理位置 = 304种组合
✅ **真实物理模型**：基于RC网络的热力学模拟
✅ **真实天气数据**：EPW格式，8760小时/年
✅ **成熟的基线**：内置MPC控制器
✅ **最小侵入性**：通过适配器层集成，不修改原始代码
✅ **完全兼容**：符合DROPT的环境接口规范

### 10.2 实施建议

1. **先实现基础功能**：
   - 创建 `BearEnvWrapper` 类
   - 实现状态/动作/奖励适配
   - 测试基本功能

2. **再添加专家控制器**：
   - 包装BEAR的MPC控制器
   - 实现PID控制器
   - 测试行为克隆训练

3. **最后优化和扩展**：
   - 性能调优
   - 添加可视化
   - 编写文档

### 10.3 预期成果

完成集成后，你将拥有：
- ✅ 一个功能完整的建筑环境模拟器
- ✅ 支持16种建筑类型和19个地理位置
- ✅ 与DROPT框架无缝集成
- ✅ 支持行为克隆和策略梯度训练
- ✅ 内置专家控制器（MPC/PID/规则）
- ✅ 完整的测试和文档

---

## 附录

### A. BEAR建筑类型列表

| 建筑类型 | 描述 | 典型房间数 |
|---------|------|-----------|
| ApartmentHighRise | 高层公寓 | 20-50 |
| ApartmentMidRise | 中层公寓 | 10-30 |
| Hospital | 医院 | 30-80 |
| HotelLarge | 大型酒店 | 40-100 |
| HotelSmall | 小型酒店 | 10-30 |
| OfficeLarge | 大型办公楼 | 30-80 |
| OfficeMedium | 中型办公楼 | 10-30 |
| OfficeSmall | 小型办公楼 | 5-15 |
| OutPatientHealthCare | 门诊医疗 | 10-30 |
| RestaurantFastFood | 快餐店 | 3-8 |
| RestaurantSitDown | 正餐餐厅 | 5-15 |
| RetailStandalone | 独立零售店 | 5-15 |
| RetailStripmall | 购物中心 | 10-30 |
| SchoolPrimary | 小学 | 15-40 |
| SchoolSecondary | 中学 | 20-60 |
| Warehouse | 仓库 | 3-10 |

### B. BEAR气候类型列表

| 气候类型 | 描述 | 代表城市 |
|---------|------|---------|
| Very_Hot_Humid | 极热湿润 | Honolulu |
| Hot_Humid | 热湿润 | Tampa |
| Hot_Dry | 热干燥 | Tucson |
| Warm_Humid | 温暖湿润 | Atlanta |
| Warm_Dry | 温暖干燥 | El Paso |
| Warm_Marine | 温暖海洋性 | San Diego |
| Mixed_Humid | 混合湿润 | New York |
| Mixed_Dry | 混合干燥 | Albuquerque |
| Mixed_Marine | 混合海洋性 | Seattle |
| Cool_Humid | 凉爽湿润 | Buffalo |
| Cool_Dry | 凉爽干燥 | Denver |
| Cool_Marine | 凉爽海洋性 | Port Angeles |
| Cold_Humid | 寒冷湿润 | Rochester |
| Cold_Dry | 寒冷干燥 | Great Falls |
| Very_Cold | 极寒 | International Falls |
| Subarctic/Arctic | 亚北极/北极 | Fairbanks |

### C. 参考资料

1. **BEAR论文**：
   - Zhang, C., Shi, Y., & Chen, Y. (2023). BEAR: Physics-Principled Building Environment for Control and Reinforcement Learning. ACM e-Energy 2023.

2. **DROPT相关**：
   - 扩散模型：Ho et al. (2020). Denoising Diffusion Probabilistic Models.
   - Tianshou框架：https://github.com/thu-ml/tianshou

3. **建筑能源管理**：
   - ASHRAE标准：https://www.ashrae.org/
   - EnergyPlus文档：https://energyplus.net/

---

**文档版本**: v1.0
**最后更新**: 2025-11-07
**作者**: DROPT Team


