# Bug 修复说明：MLP 参数不匹配问题

## 问题描述

### 错误信息
```
TypeError: MLP.__init__() got an unexpected keyword argument 'hidden_sizes'
```

### 错误堆栈
```python
File "main_datacenter.py", line 181, in main
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_sizes=args.hidden_sizes
    )
TypeError: MLP.__init__() got an unexpected keyword argument 'hidden_sizes'
```

### 问题原因

这是一个**参数命名不一致**的问题：

1. **MLP 类定义** (`diffusion/model.py`):
   ```python
   class MLP(nn.Module):
       def __init__(
           self,
           state_dim,
           action_dim,
           hidden_dim=256,  # 单个整数
           t_dim=16,
           activation='mish'
       ):
   ```
   - 接受 `hidden_dim` 参数（单个整数）
   - 表示隐藏层的维度大小

2. **main_datacenter.py 原始代码**:
   ```python
   parser.add_argument('--hidden-sizes', type=int, nargs='+', 
                       default=[256, 256, 256])
   
   actor_net = MLP(
       state_dim=args.state_shape,
       action_dim=args.action_shape,
       hidden_sizes=args.hidden_sizes  # 错误：传递列表
   )
   ```
   - 使用 `hidden_sizes` 参数（列表）
   - 与 MLP 类的参数名不匹配

3. **main_building.py 正确代码**:
   ```python
   parser.add_argument('--hidden-dim', type=int, default=256)
   
   actor_net = MLP(
       state_dim=args.state_shape,
       action_dim=args.action_shape,
       hidden_dim=args.hidden_dim  # 正确：传递整数
   )
   ```
   - 使用 `hidden_dim` 参数（整数）
   - 与 MLP 类的参数名匹配

## 修复方案

### 修改文件
`main_datacenter.py`

### 修改 1: 参数定义

**修改前**:
```python
# ========== 网络架构参数 ==========
parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 256],
                    help='MLP隐藏层大小')
```

**修改后**:
```python
# ========== 网络架构参数 ==========
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='MLP隐藏层维度')
```

**关键变化**:
- 参数名: `--hidden-sizes` → `--hidden-dim`
- 类型: `nargs='+'` (列表) → 单个整数
- 默认值: `[256, 256, 256]` → `256`

### 修改 2: MLP 初始化

**修改前**:
```python
actor_net = MLP(
    state_dim=args.state_shape,
    action_dim=args.action_shape,
    hidden_sizes=args.hidden_sizes  # 错误的参数名
)
```

**修改后**:
```python
actor_net = MLP(
    state_dim=args.state_shape,
    action_dim=args.action_shape,
    hidden_dim=args.hidden_dim  # 正确的参数名
)
```

**关键变化**:
- 参数名: `hidden_sizes` → `hidden_dim`
- 值类型: 列表 → 整数

## 设计说明

### 为什么使用 `hidden_dim` 而不是 `hidden_sizes`？

#### MLP 网络结构
当前的 MLP 实现是一个**固定结构**的网络：

```python
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, ...):
        # 状态编码器: state_dim → hidden_dim → hidden_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间编码器: t_dim → t_dim*2 → t_dim
        self.time_mlp = nn.Sequential(...)
        
        # 融合层: (hidden_dim + action_dim + t_dim) → hidden_dim → hidden_dim → action_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim)
        )
```

**特点**:
- 所有隐藏层使用**相同的维度** `hidden_dim`
- 网络深度是固定的（2层状态编码 + 3层融合）
- 只需要一个参数控制网络容量

#### 如果需要灵活的层数和维度

如果未来需要支持不同的层数和维度，可以修改 MLP 类：

```python
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], ...):
        """
        hidden_sizes: 隐藏层维度列表，例如 [256, 256, 128]
        """
        # 动态构建网络
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Mish())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
```

但对于当前的扩散模型实现，固定结构已经足够。

## 验证修复

### 测试代码
```python
from diffusion.model import MLP
import torch

# 测试 MLP 初始化
state_dim = 13
action_dim = 4
hidden_dim = 256

# 正确的初始化方式
model = MLP(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=hidden_dim
)

print(f"✓ MLP initialized successfully")
print(f"  State dim: {state_dim}")
print(f"  Action dim: {action_dim}")
print(f"  Hidden dim: {hidden_dim}")

# 测试前向传播
batch_size = 32
x = torch.randn(batch_size, action_dim)
time = torch.randint(0, 5, (batch_size,))
state = torch.randn(batch_size, state_dim)

output = model(x, time, state)
print(f"✓ Forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == (batch_size, action_dim)
```

### 运行训练验证
```bash
# 快速测试（使用默认 hidden_dim=256）
python main_datacenter.py --bc-coef --epoch 10 --episode-length 10 --device cpu

# 自定义隐藏层维度
python main_datacenter.py --bc-coef --hidden-dim 128 --epoch 10 --device cpu
python main_datacenter.py --bc-coef --hidden-dim 512 --epoch 10 --device cpu
```

## 命令行使用

### 正确的命令
```bash
# 使用默认隐藏层维度（256）
python main_datacenter.py --bc-coef --epoch 50000

# 自定义隐藏层维度
python main_datacenter.py --bc-coef --hidden-dim 128 --epoch 50000
python main_datacenter.py --bc-coef --hidden-dim 512 --epoch 50000

# 小网络（快速训练，性能较低）
python main_datacenter.py --bc-coef --hidden-dim 64 --epoch 50000

# 大网络（慢速训练，性能较高）
python main_datacenter.py --bc-coef --hidden-dim 1024 --epoch 50000
```

### 错误的命令（已修复）
```bash
# 这些命令在旧版本中会报错
python main_datacenter.py --hidden-sizes 256 256 256  # 错误：参数名不存在
```

## 与 main_building.py 的一致性

修复后，两个主程序使用相同的参数名：

### main_datacenter.py
```python
parser.add_argument('--hidden-dim', type=int, default=256)
actor_net = MLP(..., hidden_dim=args.hidden_dim)
```

### main_building.py
```python
parser.add_argument('--hidden-dim', type=int, default=256)
actor_net = MLP(..., hidden_dim=args.hidden_dim)
```

**优点**:
- ✅ 参数命名一致
- ✅ 使用方式相同
- ✅ 文档更清晰
- ✅ 减少混淆

## 性能建议

### 隐藏层维度选择

| hidden_dim | 参数量 | 训练速度 | 性能 | 适用场景 |
|------------|--------|----------|------|----------|
| 64 | ~10K | 很快 | 低 | 快速原型验证 |
| 128 | ~40K | 快 | 中低 | 简单任务 |
| 256 | ~160K | 中等 | 中高 | **推荐默认** |
| 512 | ~650K | 慢 | 高 | 复杂任务 |
| 1024 | ~2.6M | 很慢 | 很高 | 大规模问题 |

### 选择建议

1. **快速验证**: `--hidden-dim 64` 或 `128`
2. **标准训练**: `--hidden-dim 256`（默认）
3. **追求性能**: `--hidden-dim 512`
4. **大规模问题**: `--hidden-dim 1024`（需要更多 GPU 内存）

### 与其他参数的配合

```bash
# 小网络 + 快速训练
python main_datacenter.py \
    --hidden-dim 128 \
    --batch-size 128 \
    --diffusion-steps 3 \
    --epoch 30000

# 大网络 + 高性能训练
python main_datacenter.py \
    --hidden-dim 512 \
    --batch-size 512 \
    --diffusion-steps 8 \
    --epoch 100000 \
    --device cuda:0
```

## 相关文档更新

已更新以下文档：

1. **完整教程** (`docs/TUTORIAL_CN.md`)
   - 第 4.1.2 节：更新参数说明，使用 `--hidden-dim`
   - 第 7.2 节：添加 Q3.6 - MLP 参数错误问题

2. **快速参考** (`docs/QUICK_REFERENCE_CN.md`)
   - 训练参数部分：更新为 `--hidden-dim`
   - 常见问题部分：添加 MLP 参数错误的快速解决方案

## 总结

### 问题
- `main_datacenter.py` 使用 `hidden_sizes`（列表）
- MLP 类期望 `hidden_dim`（整数）
- 参数名不匹配导致初始化失败

### 解决方案
- 统一使用 `hidden_dim` 参数名
- 传递单个整数而不是列表
- 与 `main_building.py` 保持一致

### 影响
- ✅ 修复了 MLP 初始化错误
- ✅ 统一了两个主程序的参数命名
- ✅ 简化了命令行使用
- ✅ 更符合当前 MLP 实现的设计

### 建议
- 使用默认值 `--hidden-dim 256` 进行标准训练
- 根据任务复杂度和硬件资源调整隐藏层维度
- 大网络需要更多训练时间和 GPU 内存

---

**修复日期**: 2024-01-15  
**修复版本**: v1.1  
**测试状态**: ✅ 已验证

