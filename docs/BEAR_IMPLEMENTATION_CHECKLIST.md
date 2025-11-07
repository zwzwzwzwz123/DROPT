# BEAR 集成实现清单

## 📋 总览

- **总文件数**: 6 个新文件 + 2 个修改
- **总代码量**: ~1550 行
- **预计时间**: 5-9 天
- **难度等级**: ⭐⭐⭐ (中等)

---

## 第一阶段：基础集成 (1-2天)

### ✅ 任务 1.1：安装依赖

**文件**: `requirements.txt` 或 `bear_requirements.txt`

**操作**:
```bash
pip install pvlib scikit-learn cvxpy
```

**验证**:
```python
import pvlib
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
print("依赖安装成功！")
```

**预计时间**: 10分钟

---

### ⬜ 任务 1.2：创建 BearEnvWrapper 基础类

**文件**: `env/building_env_wrapper.py` (新建)

**需要实现的方法**:
- [ ] `__init__()` - 初始化环境
  - [ ] 调用 `ParameterGenerator` 生成参数
  - [ ] 创建 `BuildingEnvReal` 实例
  - [ ] 设置状态和动作空间
  - [ ] 初始化计数器和标志

- [ ] `reset()` - 重置环境
  - [ ] 调用 BEAR 的 `reset()`
  - [ ] 适配状态格式
  - [ ] 返回初始状态和信息

- [ ] `step()` - 执行一步
  - [ ] 适配动作格式
  - [ ] 调用 BEAR 的 `step()`
  - [ ] 适配状态和奖励
  - [ ] 返回 (state, reward, done, info)

- [ ] `_adapt_observation_space()` - 适配状态空间
- [ ] `_adapt_action_space()` - 适配动作空间
- [ ] `_adapt_state()` - 适配状态向量
- [ ] `_adapt_action()` - 适配动作向量
- [ ] `_adapt_reward()` - 适配奖励值

**代码框架**:
```python
import sys
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional

# 添加 BEAR 到路径
sys.path.insert(0, 'bear')
from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator

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
        episode_length: Optional[int] = None,
        expert_type: str = 'mpc',
        **kwargs
    ):
        """初始化适配器"""
        # TODO: 实现初始化逻辑
        pass
    
    def reset(self, *, seed=None, options=None):
        """重置环境"""
        # TODO: 实现重置逻辑
        pass
    
    def step(self, action: np.ndarray):
        """执行一步"""
        # TODO: 实现步进逻辑
        pass
    
    # ... 其他方法
```

**预计时间**: 4-6 小时

---

### ⬜ 任务 1.3：创建环境创建函数

**文件**: `env/building_env_wrapper.py` (继续)

**需要实现的函数**:
- [ ] `make_building_env()` - 创建向量化环境
  - [ ] 创建单个环境实例
  - [ ] 创建训练环境向量 (DummyVectorEnv)
  - [ ] 创建测试环境向量 (DummyVectorEnv)
  - [ ] 返回 (env, train_envs, test_envs)

**代码框架**:
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
    
    # TODO: 实现环境创建逻辑
    pass
```

**预计时间**: 1-2 小时

---

### ⬜ 任务 1.4：基本功能测试

**文件**: `scripts/test_building_env.py` (新建)

**测试内容**:
- [ ] 环境创建测试
- [ ] 状态空间测试
- [ ] 动作空间测试
- [ ] reset() 测试
- [ ] step() 测试
- [ ] 多步运行测试

**测试脚本**:
```python
import numpy as np
from env.building_env_wrapper import BearEnvWrapper, make_building_env

def test_env_creation():
    """测试环境创建"""
    print("测试环境创建...")
    env = BearEnvWrapper(
        building_type='OfficeSmall',
        weather_type='Hot_Dry',
        location='Tucson'
    )
    print(f"✓ 环境创建成功")
    print(f"  状态维度: {env.observation_space.shape}")
    print(f"  动作维度: {env.action_space.shape}")

def test_reset():
    """测试重置"""
    print("\n测试重置...")
    env = BearEnvWrapper()
    state, info = env.reset()
    print(f"✓ 重置成功")
    print(f"  状态形状: {state.shape}")
    print(f"  状态范围: [{state.min():.2f}, {state.max():.2f}]")

def test_step():
    """测试步进"""
    print("\n测试步进...")
    env = BearEnvWrapper()
    state, _ = env.reset()
    
    for i in range(10):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(f"  Step {i}: reward={reward:.2f}, done={done}")
        if done:
            break
    print(f"✓ 步进测试成功")

if __name__ == '__main__':
    test_env_creation()
    test_reset()
    test_step()
    print("\n✅ 所有基本测试通过！")
```

**预计时间**: 1-2 小时

---

## 第二阶段：专家控制器 (2-3天)

### ⬜ 任务 2.1：创建专家控制器基类

**文件**: `env/building_expert_controller.py` (新建)

**需要实现的类**:
- [ ] `BaseBearController` - 基类
  - [ ] `__init__()` - 初始化
  - [ ] `get_action()` - 获取动作（抽象方法）
  - [ ] `reset()` - 重置控制器

**代码框架**:
```python
import numpy as np
from abc import ABC, abstractmethod

class BaseBearController(ABC):
    """BEAR专家控制器基类"""
    
    def __init__(self, bear_params: dict):
        """初始化控制器"""
        self.bear_params = bear_params
        self.roomnum = bear_params['roomnum']
        self.target = bear_params['target']
    
    @abstractmethod
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """获取控制动作"""
        pass
    
    def reset(self):
        """重置控制器"""
        pass
```

**预计时间**: 1 小时

---

### ⬜ 任务 2.2：包装 BEAR 的 MPC 控制器

**文件**: `env/building_expert_controller.py` (继续)

**需要实现的类**:
- [ ] `BearMPCWrapper` - MPC控制器包装器
  - [ ] `__init__()` - 初始化 MPCAgent
  - [ ] `get_action()` - 获取MPC动作
  - [ ] 动作归一化到 [-1, 1]

**代码框架**:
```python
import sys
sys.path.insert(0, 'bear')
from BEAR.Controller.MPC_Controller import MPCAgent

class BearMPCWrapper(BaseBearController):
    """BEAR MPC控制器包装器"""
    
    def __init__(self, bear_env, bear_params):
        """初始化MPC控制器"""
        super().__init__(bear_params)
        self.mpc = MPCAgent(
            environment=bear_env,
            gamma=bear_params['gamma'],
            planning_steps=1
        )
    
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """获取MPC动作"""
        # TODO: 实现MPC动作获取
        pass
```

**预计时间**: 2-3 小时

---

### ⬜ 任务 2.3：实现 PID 控制器

**文件**: `env/building_expert_controller.py` (继续)

**需要实现的类**:
- [ ] `BearPIDController` - PID控制器
  - [ ] `__init__()` - 初始化PID参数
  - [ ] `get_action()` - 计算PID控制动作
  - [ ] `reset()` - 重置积分项

**代码框架**:
```python
class BearPIDController(BaseBearController):
    """BEAR PID控制器"""
    
    def __init__(self, bear_params, kp=1.0, ki=0.1, kd=0.01):
        """初始化PID控制器"""
        super().__init__(bear_params)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 为每个房间初始化PID状态
        self.integral = np.zeros(self.roomnum)
        self.last_error = np.zeros(self.roomnum)
    
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """计算PID控制动作"""
        # TODO: 实现PID控制逻辑
        pass
    
    def reset(self):
        """重置PID状态"""
        self.integral = np.zeros(self.roomnum)
        self.last_error = np.zeros(self.roomnum)
```

**预计时间**: 2-3 小时

---

### ⬜ 任务 2.4：实现基于规则的控制器

**文件**: `env/building_expert_controller.py` (继续)

**需要实现的类**:
- [ ] `BearRuleBasedController` - 规则控制器
  - [ ] `__init__()` - 初始化规则参数
  - [ ] `get_action()` - 基于规则计算动作

**代码框架**:
```python
class BearRuleBasedController(BaseBearController):
    """BEAR基于规则的控制器"""
    
    def __init__(self, bear_params, deadband=1.0):
        """初始化规则控制器"""
        super().__init__(bear_params)
        self.deadband = deadband  # 温度死区
    
    def get_action(self, state: np.ndarray, env) -> np.ndarray:
        """基于规则计算动作"""
        # TODO: 实现规则控制逻辑
        # 简单规则：
        # - 温度 > 目标 + 死区 -> 制冷 (负值)
        # - 温度 < 目标 - 死区 -> 制热 (正值)
        # - 其他 -> 关闭 (0)
        pass
```

**预计时间**: 1-2 小时

---

### ⬜ 任务 2.5：集成专家控制器到 BearEnvWrapper

**文件**: `env/building_env_wrapper.py` (修改)

**需要修改的方法**:
- [ ] `__init__()` - 添加专家控制器创建
- [ ] `_create_expert_controller()` - 创建专家控制器
- [ ] `step()` - 在 info 中添加专家动作

**代码修改**:
```python
def _create_expert_controller(self, expert_type: str):
    """创建专家控制器"""
    from env.building_expert_controller import (
        BearMPCWrapper,
        BearPIDController,
        BearRuleBasedController
    )
    
    if expert_type == 'mpc':
        return BearMPCWrapper(self.bear_env, self.bear_params)
    elif expert_type == 'pid':
        return BearPIDController(self.bear_params)
    elif expert_type == 'rule_based':
        return BearRuleBasedController(self.bear_params)
    else:
        raise ValueError(f"Unknown expert type: {expert_type}")
```

**预计时间**: 1 小时

---

### ⬜ 任务 2.6：测试专家控制器

**文件**: `scripts/test_building_env.py` (修改)

**测试内容**:
- [ ] MPC控制器测试
- [ ] PID控制器测试
- [ ] 规则控制器测试
- [ ] 性能对比测试

**测试脚本**:
```python
def test_expert_controllers():
    """测试专家控制器"""
    print("\n测试专家控制器...")
    
    expert_types = ['mpc', 'pid', 'rule_based']
    
    for expert_type in expert_types:
        print(f"\n  测试 {expert_type} 控制器...")
        env = BearEnvWrapper(expert_type=expert_type)
        state, _ = env.reset()
        
        total_reward = 0
        for i in range(24):  # 24小时
            expert_action = env.expert_controller.get_action(state, env)
            next_state, reward, done, truncated, info = env.step(expert_action)
            total_reward += reward
            state = next_state
            if done:
                break
        
        print(f"    ✓ {expert_type}: 总奖励 = {total_reward:.2f}")
```

**预计时间**: 1-2 小时

---

## 第三阶段：训练集成 (1-2天)

### ⬜ 任务 3.1：创建训练脚本

**文件**: `main_building.py` (新建)

**需要实现的功能**:
- [ ] 参数解析
  - [ ] 建筑类型参数
  - [ ] 气候类型参数
  - [ ] 训练超参数
  - [ ] 设备参数

- [ ] 环境创建
  - [ ] 调用 `make_building_env()`
  - [ ] 设置向量化环境

- [ ] 网络初始化
  - [ ] Actor (Diffusion模型)
  - [ ] Critic (DoubleCritic)

- [ ] 训练循环
  - [ ] 使用 Tianshou 的 `offpolicy_trainer`
  - [ ] 日志记录
  - [ ] 模型保存

**代码框架**:
```python
import argparse
import torch
from env.building_env_wrapper import make_building_env
from policy.diffusion_opt import DiffusionOPT
# ... 其他导入

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    # 环境参数
    parser.add_argument('--building-type', type=str, default='OfficeSmall')
    parser.add_argument('--weather-type', type=str, default='Hot_Dry')
    parser.add_argument('--location', type=str, default='Tucson')
    # ... 更多参数
    
    return parser.parse_args()

def main():
    """主训练函数"""
    args = get_args()
    
    # 创建环境
    env, train_envs, test_envs = make_building_env(...)
    
    # 创建网络
    actor = ...
    critic = ...
    
    # 创建策略
    policy = DiffusionOPT(...)
    
    # 训练
    result = offpolicy_trainer(...)
    
if __name__ == '__main__':
    main()
```

**预计时间**: 4-6 小时

---

### ⬜ 任务 3.2：端到端训练测试

**操作**:
```bash
# 快速测试（1000 epochs）
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --bc-coef \
    --expert-type mpc \
    --epoch 1000 \
    --device cpu
```

**验证内容**:
- [ ] 训练能正常启动
- [ ] 损失正常下降
- [ ] 奖励正常提升
- [ ] 模型能正常保存

**预计时间**: 2-3 小时

---

## 第四阶段：优化和文档 (1-2天)

### ⬜ 任务 4.1：创建配置文件

**文件**: `env/building_config.py` (新建)

**内容**:
- [ ] 预定义建筑配置
- [ ] 训练超参数推荐
- [ ] 环境参数模板

**预计时间**: 1-2 小时

---

### ⬜ 任务 4.2：编写使用文档

**文件**: `docs/BEAR_INTEGRATION_GUIDE.md` (新建)

**内容**:
- [ ] 快速开始指南
- [ ] API 文档
- [ ] 使用示例
- [ ] 常见问题

**预计时间**: 2-3 小时

---

### ⬜ 任务 4.3：性能优化

**优化项**:
- [ ] 数据加载优化
- [ ] 缓存机制
- [ ] 并行化
- [ ] 内存优化

**预计时间**: 2-4 小时

---

### ⬜ 任务 4.4：修改 env/__init__.py

**文件**: `env/__init__.py` (修改)

**操作**:
```python
# 添加以下导入
from .building_env_wrapper import BearEnvWrapper, make_building_env
from .building_expert_controller import (
    BearMPCWrapper,
    BearPIDController,
    BearRuleBasedController
)
```

**预计时间**: 5 分钟

---

## 📊 进度追踪

### 总体进度

- [ ] 第一阶段：基础集成 (0/4)
- [ ] 第二阶段：专家控制器 (0/6)
- [ ] 第三阶段：训练集成 (0/2)
- [ ] 第四阶段：优化文档 (0/4)

**总进度**: 0/16 任务完成

---

## 🎯 里程碑

- [ ] **里程碑 1**: 基础环境可运行 (第1阶段完成)
- [ ] **里程碑 2**: 专家控制器可用 (第2阶段完成)
- [ ] **里程碑 3**: 端到端训练成功 (第3阶段完成)
- [ ] **里程碑 4**: 完整功能和文档 (第4阶段完成)

---

## 📝 注意事项

1. **路径问题**
   - 确保 `bear` 文件夹在项目根目录
   - 使用 `sys.path.insert(0, 'bear')` 添加到路径

2. **依赖版本**
   - `pvlib >= 0.9.0`
   - `cvxpy >= 1.2.0`
   - `scikit-learn >= 1.0.0`

3. **测试策略**
   - 每完成一个任务立即测试
   - 不要等到所有代码写完再测试
   - 使用小规模数据快速验证

4. **代码风格**
   - 遵循 PEP 8 规范
   - 添加类型注解
   - 编写详细的中文注释

---

**准备好开始实现了吗？建议从任务 1.2 开始！** 🚀


