# DROPT 快速参考卡片

## 🚀 一键启动

### 最快开始（5分钟）
```bash
# 数据中心 - 快速演示
python main_datacenter.py --bc-coef --epoch 1000 --device cpu

# 建筑环境 - 快速演示（需要先安装 BEAR 依赖）
python scripts/install_bear_deps.py
python main_building.py --building-type OfficeSmall --epoch 1000 --device cpu
```

---

## 📦 环境配置

### 创建环境
```bash
# Conda
conda create --name dropt python=3.8
conda activate dropt

# venv
python -m venv dropt_env
source dropt_env/bin/activate  # Linux/Mac
dropt_env\Scripts\activate     # Windows
```

### 安装依赖
```bash
# 核心依赖
pip install torch==1.13.1 tianshou==0.4.11 numpy pandas scipy matplotlib gym gymnasium tensorboard

# BEAR 依赖（建筑环境）
pip install pvlib scikit-learn cvxpy
# 或使用脚本
python scripts/install_bear_deps.py
```

### 验证安装
```bash
# 测试数据中心环境
python scripts/test_datacenter_env.py

# 测试建筑环境
python scripts/test_building_env_basic.py
```

---

## 🎯 训练命令速查

### 数据中心训练

#### 行为克隆模式（推荐入门）
```bash
# 基础训练（PID 专家）
python main_datacenter.py --bc-coef --expert-type pid --epoch 50000 --device cuda:0

# 高级训练（MPC 专家）
python main_datacenter.py --bc-coef --expert-type mpc --epoch 50000 --device cuda:0

# 快速验证
python main_datacenter.py --bc-coef --epoch 1000 --episode-length 50 --device cpu
```

#### 策略梯度模式（追求性能）
```bash
# 标准训练
python main_datacenter.py --epoch 200000 --batch-size 512 --device cuda:0

# 高性能配置
python main_datacenter.py \
    --epoch 200000 \
    --batch-size 512 \
    --diffusion-steps 8 \
    --gamma 0.99 \
    --actor-lr 1e-4 \
    --prioritized-replay \
    --device cuda:0
```

#### 混合模式（最佳实践）
```bash
# 阶段 1: BC 预训练
python main_datacenter.py \
    --bc-coef --expert-type mpc \
    --epoch 30000 \
    --log-prefix bc_pretrain \
    --device cuda:0

# 阶段 2: PG 精调
python main_datacenter.py \
    --resume-path log/bc_pretrain_*/policy_best.pth \
    --epoch 100000 \
    --batch-size 512 \
    --log-prefix pg_finetune \
    --device cuda:0
```

### 建筑环境训练

#### 基础训练
```bash
# 小型办公楼
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --location Tucson \
    --epoch 10000 \
    --device cuda:0

# 医院建筑
python main_building.py \
    --building-type Hospital \
    --weather-type Cold_Humid \
    --location Rochester \
    --epoch 20000 \
    --device cuda:0
```

#### 使用专家控制器
```bash
# MPC 专家
python main_building.py \
    --building-type OfficeSmall \
    --expert-type mpc \
    --bc-coef \
    --epoch 50000 \
    --device cuda:0

# PID 专家
python main_building.py \
    --building-type OfficeSmall \
    --expert-type pid \
    --bc-coef \
    --epoch 50000 \
    --device cuda:0
```

---

## 📊 监控与评估

### TensorBoard
```bash
# 启动 TensorBoard
tensorboard --logdir log

# 指定端口
tensorboard --logdir log --port 6007

# 监控特定训练
tensorboard --logdir log/datacenter_20240115_143022
```

### 模型评估
```bash
# 评估模式（不训练）
python main_datacenter.py \
    --watch \
    --resume-path log/datacenter_*/policy_best.pth \
    --test-num 20

# 从检查点恢复训练
python main_datacenter.py \
    --resume-path log/datacenter_*/checkpoint_100.pth \
    --epoch 200000 \
    --device cuda:0
```

---

## ⚙️ 关键参数速查

### 环境参数

#### 数据中心
```bash
--num-crac 4                    # CRAC 单元数量
--target-temp 24.0              # 目标温度 (°C)
--temp-tolerance 2.0            # 温度容差 (°C)
--episode-length 288            # 回合长度（步数）
--energy-weight 1.0             # 能耗权重
--temp-weight 10.0              # 温度权重
--violation-penalty 100.0       # 越界惩罚
```

#### 建筑环境
```bash
--building-type OfficeSmall     # 建筑类型
--weather-type Hot_Dry          # 气候类型
--location Tucson               # 地理位置
--target-temp 22.0              # 目标温度 (°C)
--temp-tolerance 2.0            # 温度容差 (°C)
--max-power 8000                # HVAC 最大功率 (W)
--time-resolution 3600          # 时间分辨率 (秒)
```

### 训练参数
```bash
--epoch 50000                   # 训练轮数
--batch-size 256                # 批次大小
--actor-lr 3e-4                 # Actor 学习率
--critic-lr 3e-4                # Critic 学习率
--gamma 0.99                    # 折扣因子
--tau 0.005                     # 软更新系数
--n-step 3                      # N步TD学习
--training-num 4                # 并行训练环境数
--test-num 2                    # 测试环境数
--buffer-size 1000000           # 经验回放缓冲区大小
--step-per-epoch 5000           # 每轮步数
--step-per-collect 100          # 每次收集步数
```

### 扩散模型参数
```bash
--diffusion-steps 5             # 扩散步数（3-10）
--beta-schedule vp              # 噪声调度（vp/linear/cosine）
--exploration-noise 0.1         # 探索噪声标准差
```

### 训练模式
```bash
--bc-coef                       # 启用行为克隆
--expert-type pid               # 专家类型（pid/mpc/rule_based）
--prioritized-replay            # 启用优先经验回放
--lr-decay                      # 启用学习率衰减
```

### 其他参数
```bash
--device cuda:0                 # 计算设备（cuda:0/cpu）
--seed 42                       # 随机种子
--logdir log                    # 日志目录
--log-prefix datacenter         # 日志前缀
--save-interval 10              # 保存间隔（轮数）
--resume-path path/to/model.pth # 恢复训练路径
--watch                         # 评估模式（不训练）
```

---

## 🏗️ 建筑类型和气候

### 建筑类型 (`--building-type`)
- `OfficeSmall` - 小型办公楼
- `Hospital` - 医院
- `SchoolPrimary` - 小学
- `Hotel` - 酒店
- `Warehouse` - 仓库

### 气候类型 (`--weather-type`)
- `Hot_Dry` - 炎热干燥
- `Hot_Humid` - 炎热潮湿
- `Cold_Humid` - 寒冷潮湿
- `Mixed_Humid` - 混合潮湿

### 地理位置 (`--location`)
- `Tucson` - 图森（亚利桑那）
- `Tampa` - 坦帕（佛罗里达）
- `Rochester` - 罗切斯特（纽约）

---

## 🔧 常用工具脚本

### 数据生成
```bash
# 生成模拟数据（数据中心）
python scripts/generate_data.py
```

### 测试脚本
```bash
# 测试数据中心环境
python scripts/test_datacenter_env.py

# 测试建筑环境
python scripts/test_building_env_basic.py
python scripts/test_building_expert.py

# 快速测试
python scripts/quick_test.py
```

### 演示脚本
```bash
# 建筑环境演示
python scripts/demo_building_env.py
```

---

## 🐛 常见问题快速解决

### ModuleNotFoundError
```bash
# 激活环境
conda activate dropt

# 验证安装
python -c "import tianshou; print('OK')"
```

### TypeError: reset() got unexpected keyword argument 'seed'
```bash
# 此问题已修复，确保使用最新代码
# 如果仍有问题，检查 env/datacenter_env.py 中的 reset() 方法
```

### TypeError: MLP.__init__() got unexpected keyword argument 'hidden_sizes'
```bash
# 此问题已修复，使用 --hidden-dim 参数
python main_datacenter.py --hidden-dim 256  # 正确
```

### RuntimeError: Numpy is not available
```bash
# 此问题已修复，确保使用最新代码
# 问题原因：混用了 NumPy 和 PyTorch 操作
# 解决方案：所有 tensor 操作都使用 torch.* 函数
```

### CUDA out of memory
```bash
# 减小批次大小
--batch-size 128

# 减少并行环境
--training-num 2

# 减小网络规模
--hidden-dim 128

# 使用 CPU
--device cpu
```

### 训练不收敛
```bash
# 降低学习率
--actor-lr 1e-4 --critic-lr 1e-4

# 使用 BC 预训练
--bc-coef --expert-type pid

# 增加温度权重
--temp-weight 20.0

# 延长训练
--epoch 100000
```

### 动作卡在边界
```bash
# 增加探索噪声
--exploration-noise 0.2

# 增加扩散步数
--diffusion-steps 8
```

### 训练速度慢
```bash
# 使用 GPU
--device cuda:0

# 减少扩散步数
--diffusion-steps 3

# 增加并行环境
--training-num 8
```

---

## 📁 文件位置速查

### 主程序
- 数据中心: `main_datacenter.py`
- 建筑环境: `main_building.py`

### 配置文件
- 数据中心配置: `env/datacenter_config.py`
- 建筑环境包装器: `env/building_env_wrapper.py`

### 策略实现
- DiffusionOPT: `policy/diffusion_opt.py`
- 扩散模型: `diffusion/diffusion.py`
- 神经网络: `diffusion/model.py`

### 环境实现
- 数据中心环境: `env/datacenter_env.py`
- 建筑环境: `env/building_env_wrapper.py`
- 专家控制器: `env/expert_controller.py`, `env/building_expert_controller.py`

### 工具脚本
- 数据生成: `scripts/generate_data.py`
- 测试脚本: `scripts/test_*.py`
- 演示脚本: `scripts/demo_*.py`

### 日志和模型
- 训练日志: `log/`
- 最佳模型: `log/*/policy_best.pth`
- 最终模型: `log/*/policy_final.pth`
- 检查点: `log/*/checkpoint_*.pth`

### 文档
- 完整教程: `docs/TUTORIAL_CN.md`
- 快速开始: `docs/GET_STARTED.md`
- 架构文档: `docs/ARCHITECTURE.md`
- 数据中心总结: `docs/DATACENTER_SUMMARY.md`
- BEAR 快速开始: `docs/BEAR_QUICKSTART.md`

---

## 📚 推荐训练流程

### 新手流程
1. **安装依赖**: `pip install torch tianshou numpy pandas gym tensorboard`
2. **快速验证**: `python main_datacenter.py --bc-coef --epoch 1000 --device cpu`
3. **查看结果**: `tensorboard --logdir log`
4. **标准训练**: `python main_datacenter.py --bc-coef --epoch 50000 --device cuda:0`

### 进阶流程
1. **BC 预训练**: `python main_datacenter.py --bc-coef --expert-type mpc --epoch 30000`
2. **PG 精调**: `python main_datacenter.py --resume-path log/*/policy_best.pth --epoch 100000`
3. **性能优化**: 调整参数（学习率、批次大小、扩散步数）
4. **模型评估**: `python main_datacenter.py --watch --resume-path log/*/policy_best.pth`

### 建筑环境流程
1. **安装 BEAR 依赖**: `python scripts/install_bear_deps.py`
2. **测试环境**: `python scripts/test_building_env_basic.py`
3. **运行演示**: `python scripts/demo_building_env.py`
4. **开始训练**: `python main_building.py --building-type OfficeSmall --epoch 10000`

---

## 🎓 性能基准参考

| 配置 | 模式 | Epoch | 时间 | 预期奖励 | 能耗节省 |
|------|------|-------|------|----------|----------|
| 快速验证 | BC | 1,000 | 5分钟 | ~800 | ~10% |
| 标准训练 | BC | 50,000 | 1小时 | ~1200 | ~20% |
| 高性能 | PG | 200,000 | 6小时 | ~1500 | ~30% |
| 混合模式 | BC→PG | 130,000 | 3小时 | ~1600 | ~35% |

*注: 实际性能取决于具体配置和硬件*

---

## 💡 最佳实践

### 训练建议
1. **先小后大**: 先用小配置验证，再扩大规模
2. **BC 起步**: 新手建议从行为克隆开始
3. **监控训练**: 始终使用 TensorBoard 监控
4. **定期保存**: 使用 `--save-interval` 定期保存检查点
5. **多次实验**: 使用不同随机种子 `--seed` 进行多次实验

### 参数调优
1. **学习率**: 从 3e-4 开始，不收敛则降低到 1e-4
2. **批次大小**: GPU 内存允许的情况下尽量大（256-512）
3. **扩散步数**: 平衡精度和速度（5-8 步）
4. **奖励权重**: 根据目标调整能耗和温度权重比例

### 调试技巧
1. **小规模测试**: `--epoch 100 --episode-length 10`
2. **CPU 调试**: `--device cpu` 避免 CUDA 错误
3. **查看日志**: 使用 TensorBoard 分析训练曲线
4. **运行测试**: 使用 `scripts/test_*.py` 验证环境

---

## 🔗 相关资源

- **完整教程**: `docs/TUTORIAL_CN.md`
- **项目文档**: `docs/README.md`
- **论文**: [ArXiv](https://arxiv.org/abs/2308.05384)
- **BEAR 项目**: [GitHub](https://github.com/chz056/BEAR)
- **Tianshou 文档**: [ReadTheDocs](https://tianshou.readthedocs.io/)

---

**提示**: 将此文件保存为书签，方便随时查阅！

