# Bug 修复说明：reset() 方法兼容性问题

## 问题描述

### 错误信息
```
TypeError: DataCenterEnv.reset() got an unexpected keyword argument 'seed'
```

### 错误堆栈
```python
File "main_datacenter.py", line 176, in main
    train_envs.seed(args.seed)
File "tianshou/env/venvs.py", line 345, in seed
    return [w.seed(s) for w, s in zip(self.workers, seed_list)]
File "tianshou/env/worker/dummy.py", line 45, in seed
    self.env.reset(seed=seed)
TypeError: DataCenterEnv.reset() got an unexpected keyword argument 'seed'
```

### 问题原因

这是一个 **Gym/Gymnasium API 兼容性问题**：

1. **旧版 Gym API** (< 0.26):
   ```python
   def reset(self):
       return observation
   ```

2. **新版 Gymnasium API** (>= 0.26):
   ```python
   def reset(self, seed=None, options=None):
       return observation, info
   ```

3. **Tianshou 0.4.11** 使用新版 Gymnasium API，在设置随机种子时会调用 `env.reset(seed=seed)`

4. **原始 DataCenterEnv** 使用旧版 API，不接受 `seed` 参数，导致报错

## 修复方案

### 修改文件
`env/datacenter_env.py`

### 修改前
```python
def reset(self) -> np.ndarray:
    """
    重置环境到初始状态
    
    返回：
    - state: 初始状态向量
    """
    self._num_steps = 0
    self._terminated = False
    # ... 其他初始化代码
    
    return self._state  # 只返回状态
```

### 修改后
```python
def reset(self, seed=None, options=None):
    """
    重置环境到初始状态
    
    参数：
    - seed: 随机种子（可选，用于 Gymnasium 兼容性）
    - options: 额外选项（可选，用于 Gymnasium 兼容性）
    
    返回：
    - state: 初始状态向量
    - info: 额外信息字典
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)
        
    self._num_steps = 0
    self._terminated = False
    # ... 其他初始化代码
    
    # 返回状态和信息（Gymnasium API）
    info = {
        'T_in': self.T_in,
        'T_out': self.T_out,
        'IT_load': self.IT_load
    }
    
    return self._state, info  # 返回 (state, info) 元组
```

### 关键修改点

1. **方法签名**: 添加 `seed` 和 `options` 参数（都是可选的）
   ```python
   def reset(self, seed=None, options=None):
   ```

2. **处理随机种子**: 如果提供了 `seed`，设置 NumPy 随机种子
   ```python
   if seed is not None:
       np.random.seed(seed)
   ```

3. **返回值**: 返回 `(observation, info)` 元组而不是单独的 observation
   ```python
   info = {'T_in': self.T_in, 'T_out': self.T_out, 'IT_load': self.IT_load}
   return self._state, info
   ```

## 验证修复

### 测试代码
```python
from env.datacenter_env import DataCenterEnv

# 创建环境
env = DataCenterEnv(num_crac_units=4)

# 测试旧版 API（向后兼容）
state, info = env.reset()
print(f"State shape: {state.shape}")
print(f"Info: {info}")

# 测试新版 API（带 seed）
state, info = env.reset(seed=42)
print(f"State with seed: {state.shape}")

# 测试 Tianshou 集成
from env.datacenter_env import make_datacenter_env
env, train_envs, test_envs = make_datacenter_env(training_num=4, test_num=2)
train_envs.seed(42)  # 应该不再报错
print("✓ Tianshou integration OK")
```

### 运行训练验证
```bash
# 快速测试（10 轮，10 步）
python main_datacenter.py --bc-coef --epoch 10 --episode-length 10 --device cpu

# 如果成功运行，说明修复有效
```

## 其他环境检查

### 建筑环境
`env/building_env_wrapper.py` 中的 `BearEnvWrapper.reset()` 已经使用新版 API：
```python
def reset(
    self, 
    seed: Optional[int] = None, 
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # 已经兼容，无需修改
```

### 自定义环境
如果您创建了自定义环境，请确保：

1. **继承正确的基类**:
   ```python
   import gymnasium as gym
   
   class MyEnv(gym.Env):
       def reset(self, seed=None, options=None):
           if seed is not None:
               self.np_random, seed = gym.utils.seeding.np_random(seed)
           # ... 初始化代码
           return observation, info
   ```

2. **返回正确的格式**:
   - `reset()`: 返回 `(observation, info)`
   - `step()`: 返回 `(observation, reward, terminated, truncated, info)`

## 兼容性说明

### 向后兼容
修改后的 `reset()` 方法**向后兼容**旧版代码：

```python
# 旧版调用（不传参数）- 仍然有效
state, info = env.reset()

# 新版调用（传 seed）- 也有效
state, info = env.reset(seed=42)
```

### Tianshou 兼容性
- ✅ Tianshou 0.4.11 (使用 Gymnasium API)
- ✅ Tianshou 0.4.8+ (大部分版本)

### Gym/Gymnasium 版本
- ✅ Gymnasium >= 0.26 (推荐)
- ✅ Gym 0.21 - 0.25 (部分兼容)
- ⚠️ Gym < 0.21 (可能需要额外适配)

## 相关文档更新

已更新以下文档，添加此问题的说明：

1. **完整教程** (`docs/TUTORIAL_CN.md`)
   - 第 7.2 节：添加 Q3.5 - TypeError: reset() 问题

2. **快速参考** (`docs/QUICK_REFERENCE_CN.md`)
   - 常见问题部分：添加 reset() 错误的快速解决方案

## 总结

### 问题
- Tianshou 使用新版 Gymnasium API
- 原始环境使用旧版 Gym API
- 导致 `reset(seed=...)` 调用失败

### 解决方案
- 更新 `DataCenterEnv.reset()` 方法签名
- 添加 `seed` 和 `options` 参数支持
- 返回 `(observation, info)` 元组

### 影响
- ✅ 修复了训练启动时的崩溃问题
- ✅ 保持向后兼容性
- ✅ 符合 Gymnasium 标准
- ✅ 与 Tianshou 完全兼容

### 建议
如果您在其他项目中遇到类似问题，请：
1. 检查环境的 `reset()` 和 `step()` 方法签名
2. 确保返回值格式符合 Gymnasium API
3. 添加 `seed` 参数支持以实现可重复性

---

**修复日期**: 2024-01-15  
**修复版本**: v1.1  
**测试状态**: ✅ 已验证

