# 真实数据集成快速开始指南

**目标**: 5分钟内开始使用真实数据训练数据中心空调优化模型

---

## 📋 前提条件

### 1. 环境准备

确保已安装所有依赖：
```bash
pip install pandas numpy scipy matplotlib bayesian-optimization
```

### 2. 数据准备

您需要准备包含以下字段的CSV文件：

#### **必需字段**（最低要求）
- `timestamp`: 时间戳
- `T_indoor`: 室内温度 (°C)
- `T_outdoor`: 室外温度 (°C)
- `H_indoor`: 室内湿度 (%)
- `IT_load`: IT设备功率 (kW)

#### **推荐字段**（提升精度）
- `CRAC_power`: 空调总功率 (kW)
- `T_supply_1...n`: 各CRAC供风温度 (°C)
- `fan_speed_1...n`: 各CRAC风机转速 (%)
- `T_setpoint_1...n`: 各CRAC设定温度 (°C)

#### **数据格式示例**

参考 `data/data_format_template.csv`:
```csv
timestamp,T_indoor,T_outdoor,H_indoor,IT_load,CRAC_power
2024-01-01 00:00:00,24.2,15.3,52.1,280.5,85.3
2024-01-01 00:05:00,24.3,15.2,52.3,282.1,86.1
...
```

#### **数据要求**
- ✅ 采样频率: 1-5分钟
- ✅ 时间跨度: ≥7天（推荐30天+）
- ✅ 缺失率: <5%
- ✅ 覆盖不同工况（工作日/周末、不同季节）

---

## 🚀 快速开始（3种方式）

### 方式1: 一键运行（推荐）

#### **Linux/Mac**
```bash
# 1. 将您的数据放到 raw_data/datacenter_log.csv
cp your_data.csv raw_data/datacenter_log.csv

# 2. 运行完整工作流（从项目根目录执行）
bash scripts/example_real_data_workflow.sh
```

#### **Windows**
```cmd
REM 1. 将您的数据放到 raw_data\datacenter_log.csv
copy your_data.csv raw_data\datacenter_log.csv

REM 2. 运行完整工作流（从项目根目录执行）
scripts\example_real_data_workflow.bat
```

**工作流包含**:
- ✅ 数据预处理和验证
- ✅ 模型参数校准
- ✅ 基线训练（纯仿真）
- ✅ 混合训练（仿真+真实）
- ✅ 真实数据微调

**预计时间**: 6-12小时（取决于GPU性能）

---

### 方式2: 分步执行（灵活控制）

#### **步骤1: 数据预处理**
```bash
python scripts/preprocess_real_data.py \
    --input raw_data/datacenter_log.csv \
    --output data/real_data_processed.csv \
    --validate \
    --plot
```

**输出**:
- `data/real_data_processed.csv` - 清洗后的数据
- `data/data_quality_report.txt` - 质量报告
- `data/data_visualization.png` - 可视化图表

**检查点**: 查看质量报告，确保数据质量良好

---

#### **步骤2: 模型校准**
```bash
python scripts/calibrate_model.py \
    --real-data data/real_data_processed.csv \
    --method bayesian \
    --output results/calibrated_params.json
```

**输出**:
- `results/calibrated_params.json` - 校准后的参数

**示例输出**:
```json
{
  "parameters": {
    "thermal_mass": 1450.2,
    "wall_ua": 62.3,
    "cop_nominal": 3.25,
    "crac_capacity": 105.8
  },
  "validation_metrics": {
    "temp_rmse": 0.42,
    "energy_mape": 8.5,
    "r2_score": 0.96
  }
}
```

**检查点**: 确保 `temp_rmse < 1.0°C` 且 `r2_score > 0.9`

---

#### **步骤3: 训练模型**

##### **3a. 基线训练（纯仿真）**
```bash
python main_datacenter.py \
    --bc-coef \
    --epoch 30000 \
    --calibrated-params results/calibrated_params.json \
    --logdir log_baseline
```

##### **3b. 混合训练（推荐）**
```bash
python main_datacenter.py \
    --bc-coef \
    --real-data data/real_data_processed.csv \
    --real-data-ratio-schedule progressive \
    --data-augmentation \
    --epoch 100000 \
    --calibrated-params results/calibrated_params.json \
    --logdir log_mixed
```

##### **3c. 微调（可选）**
```bash
python main_datacenter.py \
    --real-data data/real_data_processed.csv \
    --real-data-ratio 1.0 \
    --resume-path log_mixed/policy_best.pth \
    --epoch 20000 \
    --lr 1e-5 \
    --calibrated-params results/calibrated_params.json \
    --logdir log_finetuned
```

---

#### **步骤4: 查看结果**
```bash
# 启动TensorBoard
tensorboard --logdir log_mixed

# 浏览器访问
# http://localhost:6006
```

---

### 方式3: 最小化测试（快速验证）

如果只想快速验证流程：

```bash
# 1. 数据预处理（使用示例数据）
python scripts/preprocess_real_data.py \
    --input data/data_format_template.csv \
    --output data/test_processed.csv

# 2. 快速校准（少量迭代）
python scripts/calibrate_model.py \
    --real-data data/test_processed.csv \
    --method least_squares \
    --output results/test_params.json

# 3. 短时训练（验证流程）
python main_datacenter.py \
    --bc-coef \
    --real-data data/test_processed.csv \
    --real-data-ratio 0.5 \
    --epoch 1000 \
    --calibrated-params results/test_params.json \
    --logdir log_test
```

**预计时间**: 10-20分钟

---

## 📊 预期效果

### 性能对比

| 指标 | 纯仿真 | 混合训练 | 真实数据微调 |
|------|--------|---------|-------------|
| **温度RMSE** | 2.1°C | 0.8°C | **0.4°C** |
| **能耗MAPE** | 35% | 15% | **8%** |
| **越界率** | 5.2% | 2.1% | **0.9%** |
| **训练时间** | 1h | 3h | 4h |

### 训练曲线示例

**纯仿真**:
- 快速收敛但性能受限
- 能耗估算误差大

**混合训练**:
- 平衡训练效率和性能
- 逐步适应真实分布

**真实数据微调**:
- 最佳真实性能
- 可能过拟合（需验证集监控）

---

## 🔧 常见问题

### Q1: 数据预处理失败，提示缺少字段

**A**: 确保您的数据至少包含5个必需字段：
```
timestamp, T_indoor, T_outdoor, H_indoor, IT_load
```

如果缺少某些字段，可以：
1. 从其他数据源补充
2. 使用合理的默认值
3. 修改 `preprocess_real_data.py` 中的字段检查

---

### Q2: 模型校准结果不理想（RMSE > 2°C）

**可能原因**:
1. 数据质量差（噪声大、缺失多）
2. 数据不具代表性（时间跨度短）
3. 模型简化假设不适用

**解决方案**:
1. 增加数据量（≥30天）
2. 改进数据清洗
3. 调整校准方法（尝试genetic或bayesian）
4. 考虑使用更复杂的热力学模型

---

### Q3: 训练过程中GPU内存不足

**解决方案**:
```bash
# 减小batch size
python main_datacenter.py --batch-size 128 ...

# 减小网络规模
python main_datacenter.py --hidden-sizes 256 256 ...

# 使用CPU（慢）
python main_datacenter.py --device cpu ...
```

---

### Q4: 混合训练效果不如纯仿真

**可能原因**:
1. 真实数据质量差
2. 真实数据比例过高（早期训练）
3. 模型未校准

**解决方案**:
1. 使用 `progressive` 调度策略
2. 降低初始真实数据比例
3. 确保先完成模型校准
4. 启用数据增强 `--data-augmentation`

---

### Q5: 如何判断模型是否过拟合？

**监控指标**:
```bash
# 在TensorBoard中对比
- train/reward vs test/reward
- train/energy vs test/energy
- train/violation vs test/violation
```

**过拟合特征**:
- 训练集性能持续提升
- 测试集性能停滞或下降
- 训练集和测试集差距大

**解决方案**:
1. 早停（patience=10）
2. 增加数据量
3. 数据增强
4. 正则化（dropout、weight decay）

---

## 📚 进阶使用

### 自定义数据比例调度

编辑 `main_datacenter.py`，添加自定义调度函数：

```python
def custom_schedule(epoch, total_epochs):
    """自定义真实数据比例调度"""
    if epoch < 10000:
        return 0.0  # 纯仿真
    elif epoch < 50000:
        return 0.2  # 20%真实
    else:
        return 0.5  # 50%真实
```

### 多数据源融合

如果有多个数据中心的数据：

```bash
# 1. 分别预处理
python scripts/preprocess_real_data.py --input dc1.csv --output dc1_processed.csv
python scripts/preprocess_real_data.py --input dc2.csv --output dc2_processed.csv

# 2. 合并数据
cat dc1_processed.csv dc2_processed.csv > combined.csv

# 3. 使用合并数据训练
python main_datacenter.py --real-data combined.csv ...
```

### 在线学习

如果需要持续学习新数据：

```bash
# 1. 初始训练
python main_datacenter.py --real-data old_data.csv --logdir log_v1

# 2. 增量训练
python main_datacenter.py \
    --real-data new_data.csv \
    --resume-path log_v1/policy_best.pth \
    --epoch 10000 \
    --lr 1e-5 \
    --logdir log_v2
```

---

## 📞 获取帮助

如果遇到问题：

1. **查看文档**: `REAL_DATA_INTEGRATION_GUIDE.md`
2. **查看示例**: `scripts/example_real_data_workflow.sh`
3. **检查日志**: 训练日志在 `logdir/` 目录
4. **提交Issue**: 附上错误信息和数据统计

---

## ✅ 检查清单

使用前确认：

- [ ] 数据包含所有必需字段
- [ ] 数据时间跨度 ≥7天
- [ ] 数据缺失率 <5%
- [ ] 已安装所有依赖
- [ ] GPU可用（推荐）
- [ ] 磁盘空间充足（≥10GB）

训练后确认：

- [ ] 温度RMSE <1°C
- [ ] 能耗MAPE <15%
- [ ] 越界率 <2%
- [ ] 训练曲线平滑收敛
- [ ] 测试集性能良好

---

**祝您使用愉快！** 🎉

