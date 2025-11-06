# 真实数据集成完整指南

**目标**: 将数据中心真实运行数据集成到DROPT项目中，提升模型可靠性和部署可行性

**预期效果**:
- 模型精度提升50%+
- Sim-to-Real gap降低70%+
- 部署成功率从30%提升到80%+

---

## 📊 1. 数据预处理指导

### 1.1 必需的数据字段

#### **核心字段（必须）**

| 字段名 | 单位 | 说明 | 采样频率 | 典型范围 |
|--------|------|------|----------|----------|
| `timestamp` | - | 时间戳 | 1-5分钟 | ISO 8601格式 |
| `T_indoor` | °C | 机房内部温度 | 1-5分钟 | 18-30°C |
| `T_outdoor` | °C | 室外温度 | 5-15分钟 | -10-45°C |
| `H_indoor` | % | 机房内部湿度 | 1-5分钟 | 30-70% |
| `IT_load` | kW | IT设备功率 | 1-5分钟 | 50-500kW |
| `CRAC_power` | kW | 空调总功率 | 1-5分钟 | 10-200kW |

#### **推荐字段（可选但重要）**

| 字段名 | 单位 | 说明 | 用途 |
|--------|------|------|------|
| `T_supply_1...n` | °C | 各CRAC供风温度 | 控制验证 |
| `T_return` | °C | 回风温度 | 热平衡验证 |
| `fan_speed_1...n` | % | 各CRAC风机转速 | 动作记录 |
| `T_setpoint_1...n` | °C | 各CRAC设定温度 | 动作记录 |
| `compressor_state_1...n` | 0/1 | 压缩机开关状态 | 动态建模 |
| `airflow_rate` | m³/h | 总风量 | 能耗分析 |
| `PUE` | - | 能源使用效率 | 性能评估 |

#### **辅助字段（可选）**

| 字段名 | 单位 | 说明 |
|--------|------|------|
| `T_cold_aisle` | °C | 冷通道温度 |
| `T_hot_aisle` | °C | 热通道温度 |
| `server_count` | - | 运行服务器数量 |
| `cpu_utilization` | % | CPU平均利用率 |
| `network_traffic` | Gbps | 网络流量 |

---

### 1.2 数据格式规范

#### **CSV格式示例**

```csv
timestamp,T_indoor,T_outdoor,H_indoor,IT_load,CRAC_power,T_supply_1,T_supply_2,T_supply_3,T_supply_4,fan_speed_1,fan_speed_2,fan_speed_3,fan_speed_4
2024-01-01 00:00:00,24.2,15.3,52.1,280.5,85.3,18.5,18.7,18.6,18.8,75,72,78,70
2024-01-01 00:05:00,24.3,15.2,52.3,282.1,86.1,18.6,18.8,18.7,18.9,76,73,79,71
2024-01-01 00:10:00,24.1,15.4,52.0,279.8,84.9,18.4,18.6,18.5,18.7,74,71,77,69
...
```

#### **时间戳格式**

支持以下格式（推荐ISO 8601）:
```python
# 推荐格式
'2024-01-01 00:00:00'           # ISO 8601
'2024-01-01T00:00:00'           # ISO 8601 with T
'2024-01-01 00:00:00+08:00'     # 带时区

# 也支持
'01/01/2024 00:00:00'           # 美式
'2024/01/01 00:00:00'           # 中式
```

#### **采样频率建议**

| 数据类型 | 推荐频率 | 最低频率 | 说明 |
|---------|---------|---------|------|
| 温度、湿度 | 1-5分钟 | 5分钟 | 匹配仿真步长 |
| IT负载 | 1-5分钟 | 5分钟 | 捕捉负载变化 |
| 空调功率 | 1-5分钟 | 5分钟 | 能耗计算 |
| 室外温度 | 5-15分钟 | 15分钟 | 变化较慢 |
| 控制动作 | 1-5分钟 | 5分钟 | 专家数据 |

---

### 1.3 数据质量要求

#### **完整性要求**

- ✅ 核心字段缺失率 < 5%
- ✅ 连续时间跨度 ≥ 7天（推荐30天+）
- ✅ 覆盖不同季节和负载工况

#### **一致性要求**

- ✅ 时间戳单调递增，无重复
- ✅ 采样间隔基本均匀（允许±10%波动）
- ✅ 物理约束满足（如T_supply < T_indoor）

#### **合理性要求**

```python
# 物理约束检查
assert 15 <= T_indoor <= 35, "室内温度超出合理范围"
assert -20 <= T_outdoor <= 50, "室外温度超出合理范围"
assert 20 <= H_indoor <= 90, "湿度超出合理范围"
assert IT_load > 0, "IT负载必须为正"
assert CRAC_power > 0, "空调功率必须为正"
assert T_supply < T_indoor, "供风温度应低于室内温度"

# 能效约束检查
PUE = (IT_load + CRAC_power) / IT_load
assert 1.0 <= PUE <= 3.0, f"PUE={PUE:.2f}超出合理范围"
```

---

### 1.4 数据清洗步骤

#### **步骤1: 异常值检测**

```python
# 3-sigma规则
mean = df['T_indoor'].mean()
std = df['T_indoor'].std()
df['T_indoor_outlier'] = (df['T_indoor'] - mean).abs() > 3 * std

# 物理约束
df['T_indoor_invalid'] = (df['T_indoor'] < 15) | (df['T_indoor'] > 35)

# 标记异常
df['is_outlier'] = df['T_indoor_outlier'] | df['T_indoor_invalid']
```

#### **步骤2: 缺失值处理**

```python
# 方法1: 线性插值（推荐）
df['T_indoor'] = df['T_indoor'].interpolate(method='linear')

# 方法2: 前向填充（短时间缺失）
df['T_indoor'] = df['T_indoor'].fillna(method='ffill', limit=3)

# 方法3: 删除（长时间缺失）
df = df.dropna(subset=['T_indoor', 'IT_load'], thresh=0.95)
```

#### **步骤3: 重采样**

```python
# 统一采样频率到5分钟
df = df.set_index('timestamp')
df = df.resample('5T').mean()  # 5分钟平均
df = df.interpolate(method='linear')
```

#### **步骤4: 平滑滤波**

```python
# 移动平均（去除高频噪声）
window = 3  # 15分钟窗口
df['T_indoor_smooth'] = df['T_indoor'].rolling(window=window, center=True).mean()

# 或使用Savitzky-Golay滤波
from scipy.signal import savgol_filter
df['T_indoor_smooth'] = savgol_filter(df['T_indoor'], window_length=5, polyorder=2)
```

---

### 1.5 数据预处理脚本

见下文创建的 `scripts/preprocess_real_data.py`

---

## 🔧 2. 数据集成方案

### 2.1 数据加载器设计

#### **架构设计**

```
RealDataLoader
    ├─ load_csv()           # 加载CSV文件
    ├─ validate()           # 数据验证
    ├─ preprocess()         # 预处理
    ├─ get_episode()        # 获取训练episode
    └─ get_statistics()     # 统计信息
```

#### **集成方式**

```python
# env/datacenter_env.py 修改
class DataCenterEnv:
    def __init__(self, ..., real_data_file=None):
        if real_data_file:
            self.data_loader = RealDataLoader(real_data_file)
            self.use_real_data = True
        else:
            self.use_real_data = False
    
    def reset(self):
        if self.use_real_data:
            # 从真实数据采样episode
            episode_data = self.data_loader.get_episode()
            self._load_episode_data(episode_data)
        else:
            # 使用仿真数据
            ...
```

---

### 2.2 混合数据策略

#### **策略1: 交替采样**

```python
# 50%真实数据 + 50%仿真数据
if np.random.rand() < 0.5:
    episode_data = real_data_loader.get_episode()
else:
    episode_data = generate_synthetic_episode()
```

#### **策略2: 分阶段训练**

```python
# 阶段1: 纯仿真（0-30k轮）
if epoch < 30000:
    use_real_data = False
# 阶段2: 混合（30k-80k轮）
elif epoch < 80000:
    use_real_data = (np.random.rand() < 0.3)  # 30%真实
# 阶段3: 纯真实（80k+轮）
else:
    use_real_data = True
```

#### **策略3: 难度递增**

```python
# 从简单仿真 → 复杂仿真 → 真实数据
difficulty = min(epoch / 100000, 1.0)
if difficulty < 0.3:
    use_simple_sim = True
elif difficulty < 0.7:
    use_complex_sim = True
else:
    use_real_data = True
```

---

## 🎯 3. 模型校准策略

### 3.1 参数校准方法

#### **方法1: 最小二乘法（简单快速）**

```python
from scipy.optimize import least_squares

def objective(params, real_data):
    thermal_mass, wall_ua, cop_nominal = params
    
    # 运行仿真
    sim_temps = simulate(real_data['actions'], 
                         thermal_mass, wall_ua, cop_nominal)
    
    # 计算误差
    error = sim_temps - real_data['temperatures']
    return error

# 优化
result = least_squares(objective, x0=[1200, 50, 3.0], 
                       bounds=([500, 20, 2.0], [2000, 100, 4.5]))
```

#### **方法2: 贝叶斯优化（全局最优）**

```python
from bayes_opt import BayesianOptimization

def objective(thermal_mass, wall_ua, cop_nominal):
    # 运行仿真
    sim_result = simulate(real_data, thermal_mass, wall_ua, cop_nominal)
    
    # 计算拟合度（R²）
    r2 = compute_r2(sim_result, real_data)
    return r2

optimizer = BayesianOptimization(
    f=objective,
    pbounds={
        'thermal_mass': (500, 2000),
        'wall_ua': (20, 100),
        'cop_nominal': (2.0, 4.5)
    }
)

optimizer.maximize(n_iter=50)
best_params = optimizer.max['params']
```

#### **方法3: 遗传算法（鲁棒性好）**

```python
from scipy.optimize import differential_evolution

def objective(params):
    thermal_mass, wall_ua, cop_nominal = params
    sim_result = simulate(real_data, *params)
    
    # 多目标优化
    temp_error = np.mean((sim_result['temp'] - real_data['temp'])**2)
    energy_error = np.mean((sim_result['energy'] - real_data['energy'])**2)
    
    return temp_error + 0.1 * energy_error

result = differential_evolution(
    objective,
    bounds=[(500, 2000), (20, 100), (2.0, 4.5)],
    maxiter=100
)
```

---

### 3.2 校准验证指标

#### **温度预测精度**

```python
# RMSE (Root Mean Square Error)
rmse = np.sqrt(np.mean((sim_temp - real_temp)**2))
print(f"温度RMSE: {rmse:.2f}°C")  # 目标: < 0.5°C

# MAE (Mean Absolute Error)
mae = np.mean(np.abs(sim_temp - real_temp))
print(f"温度MAE: {mae:.2f}°C")  # 目标: < 0.3°C

# R² (决定系数)
r2 = 1 - np.sum((sim_temp - real_temp)**2) / np.sum((real_temp - real_temp.mean())**2)
print(f"温度R²: {r2:.3f}")  # 目标: > 0.95
```

#### **能耗预测精度**

```python
# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((sim_energy - real_energy) / real_energy)) * 100
print(f"能耗MAPE: {mape:.1f}%")  # 目标: < 10%

# 能效比误差
cop_sim = real_data['cooling'] / sim_energy
cop_real = real_data['cooling'] / real_data['energy']
cop_error = np.mean(np.abs(cop_sim - cop_real) / cop_real) * 100
print(f"COP误差: {cop_error:.1f}%")  # 目标: < 15%
```

---

## 🚀 4. 训练策略优化

### 4.1 数据分配策略

#### **推荐方案: 渐进式混合**

| 训练阶段 | 轮次范围 | 真实数据比例 | 目的 |
|---------|---------|-------------|------|
| 预热 | 0-10k | 0% | 快速探索 |
| 预训练 | 10k-30k | 10% | 学习基础策略 |
| 混合训练 | 30k-80k | 30-50% | 适应真实分布 |
| 微调 | 80k-100k | 80-100% | 优化真实性能 |

#### **实现代码**

```python
def get_real_data_ratio(epoch, total_epochs=100000):
    """动态调整真实数据比例"""
    if epoch < 10000:
        return 0.0
    elif epoch < 30000:
        return 0.1
    elif epoch < 80000:
        # 线性增长 10% → 50%
        return 0.1 + 0.4 * (epoch - 30000) / 50000
    else:
        # 线性增长 50% → 100%
        return 0.5 + 0.5 * (epoch - 80000) / 20000
```

---

### 4.2 验证集设计

#### **数据划分**

```python
# 时间序列划分（避免数据泄露）
train_end = int(len(real_data) * 0.7)
val_end = int(len(real_data) * 0.85)

train_data = real_data[:train_end]      # 70%
val_data = real_data[train_end:val_end] # 15%
test_data = real_data[val_end:]         # 15%
```

#### **验证策略**

```python
# 每1000轮验证一次
if epoch % 1000 == 0:
    val_metrics = evaluate_on_real_data(policy, val_data)
    
    print(f"Epoch {epoch}:")
    print(f"  验证能耗: {val_metrics['energy']:.1f} kWh")
    print(f"  验证温度偏差: {val_metrics['temp_dev']:.2f}°C")
    print(f"  验证越界率: {val_metrics['violation']:.1f}%")
    
    # 早停
    if val_metrics['energy'] < best_energy:
        best_energy = val_metrics['energy']
        save_model(policy, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > 10:  # 10次无改进则停止
            print("早停触发")
            break
```

---

## 📝 5. 实施步骤

### Phase 1: 数据准备（第1-2天）

#### **步骤1.1: 数据收集**
- [ ] 确认数据字段完整性
- [ ] 检查数据时间跨度（≥7天）
- [ ] 验证数据格式

#### **步骤1.2: 数据预处理**
```bash
python scripts/preprocess_real_data.py \
    --input raw_data/datacenter_log.csv \
    --output data/real_data_processed.csv \
    --validate \
    --plot
```

**预期输出**:
- `data/real_data_processed.csv` - 清洗后的数据
- `data/data_quality_report.txt` - 质量报告
- `data/data_visualization.png` - 可视化图表

---

### Phase 2: 模型校准（第3-4天）

#### **步骤2.1: 参数校准**
```bash
python scripts/calibrate_model.py \
    --real-data data/real_data_processed.csv \
    --method bayesian \
    --output results/calibrated_params.json
```

**预期输出**:
```json
{
    "thermal_mass": 1450.2,
    "wall_ua": 62.3,
    "cop_nominal": 3.25,
    "crac_capacity": 105.8,
    "validation_metrics": {
        "temp_rmse": 0.42,
        "energy_mape": 8.5,
        "r2_score": 0.96
    }
}
```

#### **步骤2.2: 验证校准效果**
```bash
python scripts/validate_calibration.py \
    --real-data data/real_data_processed.csv \
    --params results/calibrated_params.json \
    --plot
```

---

### Phase 3: 数据集成（第5天）

#### **步骤3.1: 创建数据加载器**
- 文件: `env/real_data_loader.py` ✅ (见下文)

#### **步骤3.2: 修改环境**
- 文件: `env/datacenter_env.py` (添加真实数据支持)

#### **步骤3.3: 测试集成**
```bash
python scripts/test_real_data_integration.py
```

---

### Phase 4: 训练优化（第6-10天）

#### **步骤4.1: 基线训练（纯仿真）**
```bash
python main_datacenter.py \
    --bc-coef \
    --epoch 30000 \
    --logdir log_baseline \
    --device cuda:0
```

#### **步骤4.2: 混合训练**
```bash
python main_datacenter.py \
    --bc-coef \
    --real-data data/real_data_processed.csv \
    --real-data-ratio-schedule progressive \
    --epoch 100000 \
    --logdir log_mixed \
    --device cuda:0
```

#### **步骤4.3: 微调**
```bash
python main_datacenter.py \
    --real-data data/real_data_processed.csv \
    --real-data-ratio 1.0 \
    --resume-path log_mixed/policy_best.pth \
    --epoch 20000 \
    --lr 1e-5 \
    --logdir log_finetuned \
    --device cuda:0
```

---

### Phase 5: 验证评估（第11-12天）

#### **步骤5.1: 性能对比**
```bash
python scripts/compare_performance.py \
    --baseline log_baseline/policy_best.pth \
    --mixed log_mixed/policy_best.pth \
    --finetuned log_finetuned/policy_best.pth \
    --test-data data/real_data_test.csv
```

#### **步骤5.2: 生成报告**
```bash
python scripts/generate_report.py \
    --results results/ \
    --output reports/integration_report.pdf
```

---

## 📊 预期效果对比

| 指标 | 纯仿真 | 混合训练 | 真实数据微调 |
|------|--------|---------|-------------|
| 温度RMSE | 2.1°C | 0.8°C | **0.4°C** |
| 能耗MAPE | 35% | 15% | **8%** |
| 越界率 | 5.2% | 2.1% | **0.9%** |
| 训练时间 | 1h | 3h | 4h |
| 部署成功率 | 30% | 65% | **85%** |

---

## ⚠️ 注意事项

1. **数据隐私**: 确保真实数据已脱敏
2. **数据版权**: 确认数据使用权限
3. **计算资源**: 校准和训练需要GPU
4. **时间投入**: 完整流程需要10-12天
5. **迭代优化**: 首次可能需要多次调整参数

---

**下一步**: 查看具体实现代码（见后续创建的脚本文件）

