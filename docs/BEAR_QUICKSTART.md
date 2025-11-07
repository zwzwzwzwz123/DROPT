# BEAR 集成快速开始指南

## 🚀 5 分钟快速开始

### 第 1 步：安装依赖（1 分钟）

```bash
pip install pvlib scikit-learn cvxpy gymnasium
```

或使用安装脚本：

```bash
python scripts/install_bear_deps.py
```

### 第 2 步：运行测试（2 分钟）

```bash
python scripts/test_building_env_basic.py
```

**预期输出**：
```
🎉 所有测试通过！环境基础功能正常。
```

### 第 3 步：运行演示（2 分钟）

```bash
python scripts/demo_building_env.py
```

**预期输出**：
- 演示 1: 基本使用
- 演示 2: 简单温度控制策略
- 演示 3: 不同建筑类型对比

---

## 💡 基本使用

### 创建环境

```python
from env.building_env_wrapper import BearEnvWrapper

# 创建小型办公楼环境
env = BearEnvWrapper(
    building_type='OfficeSmall',
    weather_type='Hot_Dry',
    location='Tucson'
)

print(f"房间数量: {env.roomnum}")
print(f"状态维度: {env.state_dim}")
print(f"动作维度: {env.action_dim}")
```

### 运行环境

```python
# 重置环境
state, info = env.reset()

# 运行 10 步
for step in range(10):
    # 随机动作
    action = env.action_space.sample()
    
    # 执行
    next_state, reward, done, truncated, info = env.step(action)
    
    print(f"步数 {step+1}: 奖励={reward:.2f}")
    
    if done:
        break
```

### 向量化环境

```python
from env.building_env_wrapper import make_building_env

# 创建向量化环境
env, train_envs, test_envs = make_building_env(
    building_type='OfficeSmall',
    training_num=4,
    test_num=2
)

# 批量重置
states = train_envs.reset()

# 批量执行
import numpy as np
actions = np.array([train_envs.action_space.sample() for _ in range(4)])
next_states, rewards, dones, infos = train_envs.step(actions)
```

---

## 🏢 支持的建筑类型

| 建筑类型 | 代码 | 典型房间数 |
|---------|------|-----------|
| 小型办公楼 | `OfficeSmall` | 5-15 |
| 中型办公楼 | `OfficeMedium` | 10-30 |
| 大型办公楼 | `OfficeLarge` | 30-80 |
| 医院 | `Hospital` | 30-80 |
| 大型酒店 | `HotelLarge` | 40-100 |
| 小型酒店 | `HotelSmall` | 10-30 |
| 小学 | `SchoolPrimary` | 15-40 |
| 中学 | `SchoolSecondary` | 20-60 |
| 仓库 | `Warehouse` | 3-10 |

**完整列表**：见 `docs/BEAR_INTEGRATION_PLAN.md` 附录 A

---

## 🌍 支持的气候类型

| 气候类型 | 代码 | 代表城市 |
|---------|------|---------|
| 热干燥 | `Hot_Dry` | Tucson |
| 热湿润 | `Hot_Humid` | Tampa |
| 寒冷湿润 | `Cold_Humid` | Rochester |
| 温暖海洋性 | `Warm_Marine` | San Diego |
| 混合湿润 | `Mixed_Humid` | New York |

**完整列表**：见 `docs/BEAR_INTEGRATION_PLAN.md` 附录 B

---

## ⚙️ 常用参数

```python
env = BearEnvWrapper(
    # 建筑和气候
    building_type='OfficeSmall',      # 建筑类型
    weather_type='Hot_Dry',           # 气候类型
    location='Tucson',                # 地理位置
    
    # 控制目标
    target_temp=22.0,                 # 目标温度 (°C)
    temp_tolerance=2.0,               # 温度容差 (°C)
    
    # HVAC 参数
    max_power=8000,                   # 最大功率 (W)
    time_resolution=3600,             # 时间分辨率 (秒)
    
    # 奖励函数
    energy_weight=0.001,              # 能耗权重
    temp_weight=0.999,                # 温度偏差权重
    add_violation_penalty=False,      # 是否添加越界惩罚
    violation_penalty=100.0,          # 越界惩罚系数
    
    # 回合设置
    episode_length=None,              # 回合长度（None=完整年度）
)
```

---

## 📊 状态和动作空间

### 状态空间（维度：3n+3）

```python
state = [
    T_zone_1, ..., T_zone_n,    # 房间温度 (n)
    T_outdoor,                   # 室外温度 (1)
    GHI_1, ..., GHI_n,          # 太阳辐照度 (n)
    T_ground,                    # 地面温度 (1)
    Q_occ_1, ..., Q_occ_n       # 人员热负荷 (n)
]
```

### 动作空间（维度：n）

```python
action = [
    P_hvac_1, ..., P_hvac_n     # HVAC功率 [-1, 1]
]
# 负值 = 制冷，正值 = 制热
```

### 奖励函数

```
reward = -α * ||action||₂ - β * ||error||₂
```

- α: 能耗权重（默认 0.001）
- β: 温度偏差权重（默认 0.999）
- error: (当前温度 - 目标温度) × AC_map

---

## 🔧 常见问题

### Q1: 找不到 BEAR 模块？

**A**: 确保 `bear/` 文件夹在项目根目录，路径应该是：
```
DROPT/
├── bear/
│   └── BEAR/
│       ├── Env/
│       ├── Utils/
│       └── Data/
├── env/
│   └── building_env_wrapper.py
└── ...
```

### Q2: 缺少依赖？

**A**: 运行安装脚本：
```bash
python scripts/install_bear_deps.py
```

或手动安装：
```bash
pip install pvlib scikit-learn cvxpy gymnasium
```

### Q3: 找不到数据文件？

**A**: 确保 `bear/BEAR/Data/` 目录存在，包含：
- `.epw` 文件（天气数据）
- `.table.htm` 文件（建筑数据）

### Q4: 测试失败？

**A**: 查看详细测试指南：
```bash
# 查看文档
cat docs/BEAR_PHASE1_TESTING.md

# 运行测试
python scripts/test_building_env_basic.py
```

---

## 📚 更多资源

### 文档

- **集成方案**：`docs/BEAR_INTEGRATION_PLAN.md`
- **测试指南**：`docs/BEAR_PHASE1_TESTING.md`
- **完成总结**：`docs/BEAR_PHASE1_SUMMARY.md`
- **实现清单**：`docs/BEAR_IMPLEMENTATION_CHECKLIST.md`

### 代码

- **环境适配器**：`env/building_env_wrapper.py`
- **测试脚本**：`scripts/test_building_env_basic.py`
- **演示脚本**：`scripts/demo_building_env.py`

### BEAR 原始文档

- **GitHub**：https://github.com/chz056/BEAR
- **论文**：ACM e-Energy 2023

---

## 🎯 下一步

### 第二阶段：专家控制器

完成第一阶段后，可以进入第二阶段：

1. 创建 `env/building_expert_controller.py`
2. 实现 MPC、PID、规则控制器
3. 集成到训练流程
4. 测试行为克隆训练

### 第三阶段：训练脚本

1. 创建 `main_building.py`
2. 参数解析和配置
3. 完整训练流程
4. 性能评估

---

**准备好开始了吗？运行测试脚本：**

```bash
python scripts/test_building_env_basic.py
```

祝使用愉快！🚀

