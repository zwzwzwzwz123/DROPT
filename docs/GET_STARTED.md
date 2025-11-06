# 🚀 快速上手指南

欢迎使用DROPT数据中心空调优化系统！本指南将帮助您在5分钟内开始使用。

---

## 📋 前置要求

### 必需
- Python 3.7+
- PyTorch 1.8+
- Tianshou 0.4.8+
- Gym 0.21+
- NumPy, Pandas

### 可选
- CUDA（用于GPU加速）
- TensorBoard（用于可视化）
- Matplotlib（用于绘图）

---

## ⚡ 5分钟快速开始

### 方式1: 一键启动（推荐）

#### Linux/Mac
```bash
# 赋予执行权限
chmod +x scripts/quick_start.sh

# 运行启动脚本
bash scripts/quick_start.sh
```

#### Windows
```cmd
# 双击运行或命令行执行
scripts\quick_start.bat
```

脚本会自动：
1. ✅ 检查环境和依赖
2. ✅ 创建必要目录
3. ✅ 生成模拟数据
4. ✅ 测试环境
5. ✅ 提供训练选项

### 方式2: 手动步骤

#### Step 1: 安装依赖
```bash
pip install torch tianshou gym numpy pandas tensorboard matplotlib
```

#### Step 2: 生成数据
```bash
python scripts/generate_data.py
```

#### Step 3: 测试环境
```bash
python scripts/test_datacenter_env.py
```

#### Step 4: 开始训练
```bash
# 快速演示（5分钟）
python main_datacenter.py --bc-coef --epoch 1000 --device cpu

# 标准训练（1小时）
python main_datacenter.py --bc-coef --epoch 50000 --device cuda:0
```

---

## 📚 文档导航

根据您的需求选择合适的文档：

### 🎯 我想快速使用
→ **本文档** (`GET_STARTED.md`)
→ **使用手册** (`README_DATACENTER.md`)

### 🔧 我想理解系统架构
→ **架构文档** (`ARCHITECTURE.md`)
→ **总结文档** (`DATACENTER_SUMMARY.md`)

### 🚀 我想进行迁移开发
→ **迁移指南** (`MIGRATION_GUIDE.md`)

### 📊 我想查看原始DROPT
→ **原始README** (`README.md`)

---

## 🎓 训练模式选择

### 模式1: 快速演示（推荐新手）
**目标**: 快速验证系统可用性
**时间**: ~5分钟
**命令**:
```bash
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --epoch 1000 \
    --batch-size 128 \
    --n-timesteps 3 \
    --episode-length 50 \
    --device cpu
```

### 模式2: 标准训练（推荐）
**目标**: 获得可用的控制模型
**时间**: ~1小时
**命令**:
```bash
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --num-crac 4 \
    --epoch 50000 \
    --batch-size 256 \
    --n-timesteps 5 \
    --device cuda:0
```

### 模式3: 高性能训练
**目标**: 追求最优性能
**时间**: ~6小时
**命令**:
```bash
python main_datacenter.py \
    --expert-type pid \
    --num-crac 4 \
    --epoch 200000 \
    --batch-size 512 \
    --n-timesteps 8 \
    --gamma 0.99 \
    --prioritized-replay \
    --device cuda:0
```

---

## 📊 查看训练结果

### 启动TensorBoard
```bash
tensorboard --logdir=log_datacenter
```
然后在浏览器打开: http://localhost:6006

### 测试训练好的模型
```bash
python main_datacenter.py \
    --watch \
    --resume-path log_datacenter/default/datacenter_pid_crac4_t5/XXX/policy_best.pth
```

---

## 🔍 常见问题速查

### Q1: 如何查看所有可用参数？
```bash
python main_datacenter.py --help
```

### Q2: 训练太慢怎么办？
- 减少epoch数: `--epoch 10000`
- 减少扩散步数: `--n-timesteps 3`
- 使用GPU: `--device cuda:0`
- 增加并行环境: `--training-num 16`

### Q3: 如何调整数据中心规模？
```bash
# 小型（2个CRAC）
python main_datacenter.py --num-crac 2

# 中型（4个CRAC，默认）
python main_datacenter.py --num-crac 4

# 大型（8个CRAC）
python main_datacenter.py --num-crac 8
```

### Q4: 如何调整优化目标？
```bash
# 更重视节能
python main_datacenter.py --energy-weight 2.0 --temp-weight 5.0

# 更重视温度稳定（推荐）
python main_datacenter.py --energy-weight 1.0 --temp-weight 10.0

# 严格温度控制
python main_datacenter.py --energy-weight 0.5 --temp-weight 20.0 --violation-penalty 200.0
```

### Q5: 如何使用真实数据？
```bash
# 准备CSV文件（参考scripts/generate_data.py的格式）
python main_datacenter.py \
    --weather-file data/your_weather.csv \
    --workload-file data/your_workload.csv
```

---

## 📁 项目结构速览

```
DROPT/
├── 📖 文档
│   ├── GET_STARTED.md              ← 你在这里
│   ├── README_DATACENTER.md        ← 详细使用手册
│   ├── MIGRATION_GUIDE.md          ← 迁移开发指南
│   ├── ARCHITECTURE.md             ← 系统架构
│   └── DATACENTER_SUMMARY.md       ← 项目总结
│
├── 🔧 核心代码
│   ├── main_datacenter.py          ← 训练主程序
│   ├── env/
│   │   ├── datacenter_env.py       ← 数据中心环境
│   │   ├── thermal_model.py        ← 热力学模型
│   │   ├── expert_controller.py    ← 专家控制器
│   │   └── datacenter_config.py    ← 配置文件
│   ├── diffusion/                  ← 扩散模型（复用）
│   └── policy/                     ← 策略（复用）
│
├── 🛠️ 工具脚本
│   └── scripts/
│       ├── generate_data.py        ← 数据生成
│       ├── test_datacenter_env.py  ← 环境测试
│       ├── quick_start.sh          ← Linux启动脚本
│       └── quick_start.bat         ← Windows启动脚本
│
└── 📊 数据和日志
    ├── data/                       ← 数据文件
    └── log_datacenter/             ← 训练日志
```

---

## 🎯 典型使用流程

### 场景1: 研究人员快速验证
```bash
# 1. 快速演示
bash scripts/quick_start.sh
# 选择选项1（快速演示）

# 2. 查看结果
tensorboard --logdir=log_datacenter
```

### 场景2: 工程师部署应用
```bash
# 1. 生成数据
python scripts/generate_data.py

# 2. 标准训练
python main_datacenter.py --bc-coef --epoch 50000

# 3. 测试模型
python main_datacenter.py --watch --resume-path <MODEL_PATH>

# 4. 集成到系统
# 参考README_DATACENTER.md的"部署"章节
```

### 场景3: 开发者迁移框架
```bash
# 1. 阅读迁移指南
cat MIGRATION_GUIDE.md

# 2. 测试环境
python scripts/test_datacenter_env.py

# 3. 修改代码
# 参考MIGRATION_GUIDE.md的"实施步骤"

# 4. 验证修改
python main_datacenter.py --epoch 1000
```

---

## 💡 最佳实践

### 训练建议
1. **先BC后PG**: 先用BC训练获得稳定基础，再用PG微调
2. **逐步增加难度**: 从短回合、少CRAC开始，逐步增加复杂度
3. **监控关键指标**: 重点关注温度越界率和能耗
4. **保存检查点**: 定期保存模型，避免训练中断

### 调参建议
1. **学习率**: BC用3e-4，PG用1e-4，微调用5e-5
2. **扩散步数**: 训练用5-6步，推理用8步
3. **奖励权重**: 从保守型(α=0.5, β=20)开始，逐步调整
4. **批次大小**: GPU内存允许的情况下越大越好

### 调试建议
1. **先测试环境**: 确保`test_datacenter_env.py`全部通过
2. **短时间训练**: 先用1000轮验证流程
3. **查看日志**: TensorBoard是最好的调试工具
4. **对比专家**: 性能应该接近或超过专家控制器

---

## 🆘 获取帮助

### 遇到问题？

1. **查看文档**
   - 使用问题 → `README_DATACENTER.md`
   - 开发问题 → `MIGRATION_GUIDE.md`
   - 架构问题 → `ARCHITECTURE.md`

2. **运行测试**
   ```bash
   python scripts/test_datacenter_env.py
   ```

3. **查看示例**
   - 所有文档中都有完整的命令示例
   - 参考`scripts/quick_start.sh`中的命令

4. **检查日志**
   - 终端输出
   - TensorBoard可视化
   - `log_datacenter/`目录

---

## 🎉 下一步

完成快速开始后，您可以：

1. **深入学习**: 阅读`README_DATACENTER.md`了解详细功能
2. **优化性能**: 参考"超参数调优指南"章节
3. **扩展功能**: 参考`MIGRATION_GUIDE.md`进行定制开发
4. **部署应用**: 将训练好的模型集成到实际系统

---

## 📞 联系方式

- **项目主页**: [DROPT GitHub]
- **问题反馈**: 提交Issue
- **文档更新**: 2025-11-06

---

**祝您使用愉快！** 🚀

如有任何问题，请参考详细文档或提交Issue。

