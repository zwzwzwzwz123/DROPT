# 🔍 训练诊断报告与改进方案

**日期**: 2025-11-08  
**模型**: 数据中心空调优化 (DROPT + Diffusion)  
**训练轮次**: Epoch #3845-3854  

---

## 📊 一、训练现状分析

### 1.1 关键指标

从训练日志可以看到:

| 指标 | 数值 | 状态 | 说明 |
|------|------|------|------|
| **训练速度** | 35-52 it/s | ✅ 正常 | 训练吞吐量良好 |
| **Critic损失** | 196,927 - 206,877 | 🔴 **异常高** | 价值网络无法准确估计Q值 |
| **Overall损失** | 0.003 | ✅ 正常 | Actor损失正常 |
| **训练奖励** | -53,848.59 | 🔴 **持续大负值** | 策略性能极差 |
| **测试奖励** | -47,667 ~ -63,928 | 🔴 **波动剧烈** | 策略不稳定 |
| **最佳奖励** | -31,539.07 | 🔴 **仍为大负值** | 未找到好策略 |

### 1.2 问题诊断

#### 🔴 **严重问题**:

1. **Critic损失爆炸** (20万级别)
   - 表明价值网络无法准确估计Q值
   - 可能原因: 奖励尺度过大、学习率不当、梯度爆炸

2. **奖励持续大幅负值** (-5万级别)
   - 每步平均奖励 ≈ -53848/288 ≈ **-187**
   - 正常应该在 -10 ~ 0 范围内
   - 策略完全无法学习有效行为

3. **测试奖励波动剧烈** (±16,000)
   - 说明策略不稳定
   - 泛化能力差

4. **训练3845轮仍未收敛**
   - 长时间训练无改善
   - 可能陷入局部最优或超参数不当

---

## 🔬 二、根本原因分析

### 2.1 奖励函数设计问题 ⚠️ **最关键**

#### 当前奖励函数:

```python
# 能耗惩罚
energy_penalty = alpha * energy / 100.0  # alpha=1.0
# 温度偏差惩罚
temp_penalty = beta * (temp_deviation ** 2)  # beta=10.0
# 越界惩罚
violation_penalty = gamma  # gamma=100.0

reward = -(energy_penalty + temp_penalty + violation_penalty)
```

#### 问题分析:

假设典型场景:
- 温度偏差: 2°C
- 能耗: 5 kWh/step
- 发生越界

计算:
```
energy_penalty = 1.0 * 5 / 100 = 0.05
temp_penalty = 10.0 * (2²) = 40
violation_penalty = 100
total_reward = -(0.05 + 40 + 100) = -140.05
```

**每步奖励 -140**, 288步/回合 → **累积奖励 ≈ -40,000** ❌

#### 根本问题:

1. **惩罚权重过大**: `beta=10.0`, `gamma=100.0`
2. **缺乏正向激励**: 全是负奖励,没有正向信号
3. **奖励尺度不合理**: 导致Critic损失爆炸
4. **奖励稀疏**: 只有惩罚,没有引导

---

### 2.2 超参数配置不当 ⚠️

#### 当前配置:

```python
actor_lr = 3e-4        # Actor学习率
critic_lr = 3e-4       # Critic学习率
batch_size = 256       # 批次大小
n_timesteps = 5        # 扩散步数
gamma = 0.99           # 折扣因子
exploration_noise = 0.1  # 探索噪声
```

#### 问题:

1. **学习率过高** (3e-4)
   - 对于大负奖励环境,容易导致训练不稳定
   - 建议: 1e-4 或更低

2. **Batch size偏小** (256)
   - 梯度估计噪声大
   - 建议: 512 或 1024

3. **扩散步数少** (5)
   - 生成质量可能不足
   - 建议: 8-10

4. **探索噪声过小** (0.1)
   - 在大负奖励环境中难以探索到好策略
   - 建议: 0.3-0.5

5. **未使用学习率衰减**
   - 后期无法精细调整
   - 建议: 启用 `--lr-decay`

---

### 2.3 数据归一化缺失 ⚠️

#### 当前状态构造:

```python
state = [
    T_in,        # 温度: 15-35°C
    T_out,       # 室外温度: -10-45°C
    H_in,        # 湿度: 30-70%
    IT_load,     # 负载: 100-400kW
    T_supply,    # 供风温度: 15-30°C (num_crac维)
    reward_last  # 上一步奖励: -1000-0
]
```

#### 问题:

- **量纲不统一**: 温度(20-30)、负载(100-400)、湿度(30-70)
- **未归一化**: 神经网络对不同尺度的输入敏感
- **影响**: 梯度不稳定、收敛慢

#### 建议:

所有状态归一化到 [0, 1] 或 [-1, 1]:

```python
T_in_norm = (T_in - 15) / (35 - 15)  # → [0, 1]
IT_load_norm = (IT_load - 50) / (500 - 50)  # → [0, 1]
```

---

### 2.4 训练策略问题 ⚠️

#### 当前训练模式:

```bash
--bc-coef  # 行为克隆模式
--expert-type pid  # PID专家控制器
```

#### 问题:

1. **PID专家质量未知**
   - 如果PID控制器本身性能差,BC会学到差策略
   - 建议: 先验证PID性能

2. **BC模式可能不适合**
   - 数据中心控制是连续优化问题
   - 纯BC可能无法超越专家
   - 建议: 先BC预训练,再PG微调

3. **缺乏课程学习**
   - 直接在困难任务上训练
   - 建议: 从简单场景开始

---

## 🎯 三、改进方案

### 方案1: 奖励函数重设计 🔥 **最关键**

#### 改进思路:

1. **降低惩罚权重** (10倍)
2. **添加正向奖励** (奖励塑形)
3. **归一化奖励尺度**
4. **平滑惩罚函数**

#### 新奖励函数:

```python
def _compute_reward_v2(self, T_in: float, energy: float) -> Tuple[float, Dict]:
    """改进的奖励函数"""
    
    # 1. 温度舒适度奖励（高斯型）
    temp_error = abs(T_in - self.target_temp)
    temp_reward = 10.0 * np.exp(-0.5 * (temp_error ** 2))  # 在目标温度时=10
    
    # 2. 温度惩罚（降低权重）
    temp_penalty = 1.0 * (temp_error ** 2)  # beta: 10 → 1
    
    # 3. 能耗惩罚（归一化）
    energy_normalized = energy / 10.0  # 假设单步最大10kWh
    energy_penalty = 0.1 * energy_normalized  # alpha: 1 → 0.1
    
    # 4. 越界惩罚（降低权重）
    if T_in < self.T_min or T_in > self.T_max:
        violation_penalty = 10.0  # gamma: 100 → 10
    else:
        violation_penalty = 0.0
    
    # 5. 基础存活奖励
    base_reward = 1.0
    
    # 总奖励（正负平衡）
    reward = base_reward + temp_reward - temp_penalty - energy_penalty - violation_penalty
    
    return reward, {...}
```

#### 预期效果:

- 正常情况: reward ≈ 1 + 10 - 1 - 0.05 - 0 = **+9.95** ✅
- 轻微偏差: reward ≈ 1 + 8 - 4 - 0.05 - 0 = **+4.95** ✅
- 越界情况: reward ≈ 1 + 5 - 10 - 0.05 - 10 = **-14.05** ⚠️
- 回合累积: 288 × 5 = **+1440** (正常) ✅

---

### 方案2: 超参数优化 🔧

#### 推荐配置:

```bash
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --num-crac 4 \
    --epoch 100000 \
    --batch-size 512 \          # 256 → 512
    --n-timesteps 8 \            # 5 → 8
    --actor-lr 1e-4 \            # 3e-4 → 1e-4
    --critic-lr 3e-4 \           # 保持不变
    --gamma 0.99 \
    --tau 0.005 \
    --exploration-noise 0.3 \    # 0.1 → 0.3
    --lr-decay \                 # 启用学习率衰减
    --prioritized-replay \       # 启用优先经验回放
    --device cuda:0
```

#### 关键改动:

1. ✅ **降低Actor学习率**: 3e-4 → 1e-4
2. ✅ **增大Batch size**: 256 → 512
3. ✅ **增加扩散步数**: 5 → 8
4. ✅ **增大探索噪声**: 0.1 → 0.3
5. ✅ **启用学习率衰减**: `--lr-decay`
6. ✅ **启用优先经验回放**: `--prioritized-replay`

---

### 方案3: 状态归一化 📊

#### 实现方式:

在 `env/datacenter_env.py` 中添加归一化:

```python
def _get_state_normalized(self) -> np.ndarray:
    """归一化状态"""
    state_raw = self._get_state()
    
    # 定义归一化范围
    state_min = np.array([15.0, -10.0, 20.0, 50.0] + [15.0]*self.num_crac + [-1000.0])
    state_max = np.array([35.0, 45.0, 80.0, 500.0] + [30.0]*self.num_crac + [0.0])
    
    # 归一化到 [0, 1]
    state_norm = (state_raw - state_min) / (state_max - state_min + 1e-8)
    
    return state_norm.astype(np.float32)
```

---

### 方案4: 训练策略改进 📈

#### 推荐流程:

**阶段1: 验证专家性能** (1天)
```bash
# 测试PID控制器性能
python scripts/test_datacenter_env.py --expert-type pid --episodes 10
```

**阶段2: BC预训练** (2-3天)
```bash
# 使用改进的奖励函数和超参数
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --epoch 50000 \
    --batch-size 512 \
    --actor-lr 1e-4 \
    --n-timesteps 8 \
    --exploration-noise 0.3 \
    --lr-decay \
    --logdir log_bc_improved
```

**阶段3: PG微调** (3-5天)
```bash
# 切换到策略梯度模式
python main_datacenter.py \
    --epoch 100000 \
    --batch-size 512 \
    --actor-lr 5e-5 \
    --n-timesteps 10 \
    --gamma 0.99 \
    --prioritized-replay \
    --resume-path log_bc_improved/policy_best.pth \
    --logdir log_pg_finetuned
```

---

## 📋 四、实施步骤

### Step 1: 修改奖励函数 (30分钟)

创建改进版环境文件 `env/datacenter_env_v2.py`

### Step 2: 添加状态归一化 (20分钟)

在环境中添加 `_get_state_normalized()` 方法

### Step 3: 调整超参数 (10分钟)

修改 `main_datacenter.py` 的默认参数

### Step 4: 重新训练 (2-3天)

使用新配置重新训练

### Step 5: 监控和调试 (持续)

使用TensorBoard监控:
```bash
tensorboard --logdir log_bc_improved
```

---

## 📈 五、预期效果

### 改进前 vs 改进后:

| 指标 | 改进前 | 改进后(预期) | 改善 |
|------|--------|-------------|------|
| 训练奖励 | -53,848 | +1,000 ~ +2,000 | ✅ 大幅提升 |
| 测试奖励 | -47,667 | +800 ~ +1,500 | ✅ 大幅提升 |
| Critic损失 | 200,000 | 100 ~ 1,000 | ✅ 降低200倍 |
| 收敛速度 | >3845轮未收敛 | 1000-2000轮 | ✅ 加快2-4倍 |
| 策略稳定性 | 波动±16,000 | 波动±200 | ✅ 提升80倍 |

---

## ⚠️ 六、注意事项

1. **逐步调整**: 不要一次改动太多,逐个验证效果
2. **保存检查点**: 每1000轮保存一次模型
3. **监控指标**: 重点关注 `reward`, `critic_loss`, `temp_violation`
4. **早停机制**: 如果10000轮无改善,考虑调整策略
5. **对比实验**: 保留原始配置作为baseline

---

## 📚 七、参考资料

- [DROPT论文](https://arxiv.org/abs/2209.09981)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Reward Shaping最佳实践](https://arxiv.org/abs/1908.08542)

---

**下一步行动**: 请确认是否开始实施改进方案?

