# BEAR 集成 - 第一阶段测试指南

## 📋 第一阶段完成内容

已完成以下文件的创建：

1. **`env/building_env_wrapper.py`** (约 400 行)
   - `BearEnvWrapper` 类：BEAR 环境适配器
   - `make_building_env()` 函数：创建向量化环境
   - 状态空间和动作空间适配
   - 奖励函数适配（支持可选的温度越界惩罚）

2. **`scripts/test_building_env_basic.py`** (约 250 行)
   - 7 个基础功能测试
   - 自动化测试脚本

3. **`scripts/demo_building_env.py`** (约 200 行)
   - 3 个使用示例演示
   - 简单温度控制策略演示

---

## 🚀 快速测试

### 前置条件

确保已安装 BEAR 所需的依赖：

```bash
pip install pvlib scikit-learn cvxpy gymnasium
```

### 测试 1: 运行自动化测试脚本

```bash
cd c:\Users\21118\Desktop\research\DROPT
python scripts/test_building_env_basic.py
```

**预期输出**：
- 7 个测试全部通过
- 显示 "🎉 所有测试通过！环境基础功能正常。"

**测试内容**：
1. ✓ 环境创建
2. ✓ 状态/动作空间
3. ✓ reset() 方法
4. ✓ step() 方法
5. ✓ 多步运行 (24步)
6. ✓ 向量化环境
7. ✓ 不同建筑类型

### 测试 2: 运行使用示例

```bash
python scripts/demo_building_env.py
```

**预期输出**：
- 演示 1: 基本使用
- 演示 2: 简单温度控制策略（48步）
- 演示 3: 不同建筑类型对比
- 生成可视化图表（如果 matplotlib 可用）

---

## 🔍 详细测试说明

### 测试 1: 环境创建

**测试代码**：
```python
from env.building_env_wrapper import BearEnvWrapper

env = BearEnvWrapper(
    building_type='OfficeSmall',
    weather_type='Hot_Dry',
    location='Tucson'
)

print(f"房间数量: {env.roomnum}")
print(f"状态维度: {env.state_dim}")
print(f"动作维度: {env.action_dim}")
```

**预期结果**：
- 环境创建成功
- 显示房间数量（通常为 5-15）
- 状态维度 = 3 * 房间数 + 3
- 动作维度 = 房间数

### 测试 2: 状态空间和动作空间

**测试代码**：
```python
env = BearEnvWrapper()

# 状态空间
print(f"状态空间形状: {env.observation_space.shape}")
print(f"状态空间范围: [{env.observation_space.low[0]}, {env.observation_space.high[0]}]")

# 动作空间
print(f"动作空间形状: {env.action_space.shape}")
print(f"动作空间范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
```

**预期结果**：
- 状态空间：Box(3n+3,) 其中 n 是房间数
- 动作空间：Box(n,) 范围 [-1, 1]

### 测试 3: reset() 和 step()

**测试代码**：
```python
env = BearEnvWrapper()

# 重置
state, info = env.reset()
print(f"初始状态形状: {state.shape}")
print(f"初始状态: {state[:5]}...")

# 执行一步
action = env.action_space.sample()
next_state, reward, done, truncated, info = env.step(action)

print(f"奖励: {reward:.2f}")
print(f"done: {done}")
print(f"当前步数: {info['current_step']}")
```

**预期结果**：
- reset() 返回初始状态和信息字典
- step() 返回 (state, reward, done, truncated, info)
- 奖励为负值（能耗和温度偏差惩罚）
- done 初始为 False

### 测试 4: 多步运行

**测试代码**：
```python
env = BearEnvWrapper()
state, _ = env.reset()

for step in range(24):  # 24小时
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print(f"步数 {step+1}: 奖励={reward:.2f}")
    state = next_state
    if done:
        break
```

**预期结果**：
- 能够连续运行 24 步
- 每步返回有效的状态和奖励
- 不会出现异常或错误

### 测试 5: 向量化环境

**测试代码**：
```python
from env.building_env_wrapper import make_building_env

env, train_envs, test_envs = make_building_env(
    building_type='OfficeSmall',
    training_num=2,
    test_num=1
)

print(f"训练环境数量: {train_envs.env_num}")
print(f"测试环境数量: {test_envs.env_num}")

# 重置
states = train_envs.reset()
print(f"状态形状: {states.shape}")  # (2, state_dim)

# 执行一步
import numpy as np
actions = np.array([train_envs.action_space.sample() for _ in range(2)])
next_states, rewards, dones, infos = train_envs.step(actions)
print(f"奖励: {rewards}")
```

**预期结果**：
- 成功创建向量化环境
- 状态形状为 (env_num, state_dim)
- 可以批量执行 step()

### 测试 6: 不同建筑类型

**测试代码**：
```python
building_types = ['OfficeSmall', 'Hospital', 'SchoolPrimary']

for building in building_types:
    env = BearEnvWrapper(
        building_type=building,
        weather_type='Hot_Dry',
        location='Tucson'
    )
    print(f"{building}: 房间数={env.roomnum}, 状态维度={env.state_dim}")
```

**预期结果**：
- 不同建筑类型有不同的房间数
- OfficeSmall: 约 5-10 个房间
- Hospital: 约 30-80 个房间
- SchoolPrimary: 约 15-40 个房间

---

## ⚠️ 常见问题

### 问题 1: 找不到 BEAR 模块

**错误信息**：
```
ModuleNotFoundError: No module named 'BEAR'
```

**解决方案**：
1. 确保 `bear/` 文件夹在项目根目录
2. 检查 `bear/BEAR/` 路径是否正确
3. 尝试手动添加路径：
   ```python
   import sys
   sys.path.insert(0, 'bear')
   ```

### 问题 2: 缺少依赖

**错误信息**：
```
ModuleNotFoundError: No module named 'pvlib'
```

**解决方案**：
```bash
pip install pvlib scikit-learn cvxpy gymnasium
```

### 问题 3: 找不到数据文件

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'BEAR/Data/...'
```

**解决方案**：
1. 检查 `bear/BEAR/Data/` 目录是否存在
2. 确保数据文件（.epw, .table.htm）存在
3. 检查 `root` 参数是否正确设置为 `'bear/BEAR/Data/'`

### 问题 4: Gymnasium vs Gym 版本问题

**错误信息**：
```
AttributeError: module 'gym' has no attribute 'spaces'
```

**解决方案**：
BEAR 使用 `gymnasium`，而 DROPT 使用 `gym`。适配器已处理兼容性，但如果遇到问题：
```bash
pip install gymnasium gym
```

---

## ✅ 验收标准

第一阶段成功的标志：

1. ✓ 所有 7 个自动化测试通过
2. ✓ 能够创建不同建筑类型的环境
3. ✓ 能够正常执行 reset() 和 step()
4. ✓ 能够运行多步（至少 24 步）
5. ✓ 向量化环境正常工作
6. ✓ 状态和动作空间符合预期
7. ✓ 奖励计算正常

---

## 📊 性能基准

在标准配置下（OfficeSmall, Hot_Dry, Tucson）：

- **环境创建时间**: < 5 秒
- **reset() 时间**: < 0.1 秒
- **step() 时间**: < 0.01 秒
- **24 步运行时间**: < 1 秒

如果性能明显低于这些基准，可能需要检查：
1. 数据文件加载是否正常
2. 是否有不必要的计算
3. 是否有内存泄漏

---

## 🎯 下一步

第一阶段完成后，可以进入第二阶段：

**第二阶段：专家控制器集成**
- 创建 `env/building_expert_controller.py`
- 实现 MPC、PID、规则控制器
- 集成到 `BearEnvWrapper`
- 测试行为克隆训练

---

## 📝 测试报告模板

完成测试后，请提供以下信息：

```
测试环境：
- 操作系统: Windows 11
- Python 版本: 3.x
- 依赖版本: pvlib x.x, cvxpy x.x, gymnasium x.x

测试结果：
- 自动化测试: X/7 通过
- 使用示例: 成功/失败
- 性能基准: 符合/不符合

遇到的问题：
1. [问题描述]
   - 解决方案: [如何解决]

建议：
- [改进建议]
```

---

**准备好测试了吗？运行以下命令开始：**

```bash
python scripts/test_building_env_basic.py
```

祝测试顺利！🚀

