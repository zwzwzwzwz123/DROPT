# DROPT 项目详细使用教程

## 📋 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [配置文件](#4-配置文件)
5. [开始训练](#5-开始训练)
6. [训练过程监控](#6-训练过程监控)
7. [常见问题](#7-常见问题)

---

## 1. 项目概述

### 1.1 项目简介

**DROPT (Diffusion-based Reinforcement learning OPTimization)** 是一个基于扩散模型和深度强化学习的网络优化框架。本项目包含两个主要应用场景：

1. **建筑环境 HVAC 优化** (BEAR 集成)
   - 优化建筑物的供暖、通风和空调系统
   - 平衡能耗和室内温度舒适度
   - 支持多种建筑类型和气候条件

2. **数据中心空调优化**
   - 优化数据中心的 CRAC 空调单元控制
   - 降低能耗同时保持服务器温度在安全范围
   - 支持不同规模的数据中心配置

### 1.2 核心技术

- **扩散模型 (Diffusion Model)**: 通过迭代去噪过程生成最优动作
- **Actor-Critic 架构**: 
  - Actor: 扩散模型（生成控制策略）
  - Critic: 双 Q 网络（评估动作价值）
- **两种训练模式**:
  - **行为克隆 (BC)**: 从专家控制器学习（快速收敛）
  - **策略梯度 (PG)**: 通过环境交互学习（更高性能）

### 1.3 项目结构

```
DROPT/
├── main_building.py          # 建筑环境训练主程序
├── main_datacenter.py        # 数据中心训练主程序
├── policy/                   # 策略实现
│   ├── diffusion_opt.py     # DiffusionOPT 策略类
│   └── helpers.py           # 辅助函数
├── diffusion/               # 扩散模型
│   ├── diffusion.py        # 扩散过程实现
│   ├── model.py            # 神经网络模型
│   └── helpers.py          # 辅助函数
├── env/                     # 环境定义
│   ├── building_env_wrapper.py    # BEAR 环境包装器
│   ├── building_expert_controller.py  # 建筑专家控制器
│   ├── datacenter_env.py          # 数据中心环境
│   ├── datacenter_config.py       # 数据中心配置
│   └── expert_controller.py       # 数据中心专家控制器
├── scripts/                 # 工具脚本
│   ├── generate_data.py    # 数据生成脚本
│   ├── test_*.py           # 测试脚本
│   └── install_bear_deps.py  # 依赖安装脚本
├── log/                     # 训练日志（自动创建）
└── data/                    # 数据目录（自动创建）
```

---

## 2. 环境配置

### 2.1 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python 版本**: 3.7 - 3.10 (推荐 3.8)
- **硬件要求**:
  - CPU: 4 核以上
  - 内存: 8GB 以上
  - GPU: 可选，NVIDIA GPU with CUDA 支持（训练加速）

### 2.2 创建 Python 环境

#### 使用 Conda（推荐）

```bash
# 创建新环境
conda create --name dropt python=3.8

# 激活环境
conda activate dropt
```

#### 使用 venv

```bash
# 创建虚拟环境
python -m venv dropt_env

# 激活环境 (Windows)
dropt_env\Scripts\activate

# 激活环境 (Linux/Mac)
source dropt_env/bin/activate
```

### 2.3 安装依赖

#### 核心依赖（必需）

```bash
# PyTorch (根据您的 CUDA 版本选择)
# CPU 版本
pip install torch==1.13.1 torchvision torchaudio

# CUDA 11.7 版本
pip install torch==1.13.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 强化学习框架
pip install tianshou==0.4.11

# 基础科学计算库
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scipy==1.10.1
pip install matplotlib==3.7.3

# Gym 环境接口
pip install gym==0.21.0
pip install gymnasium==0.28.1

# TensorBoard 可视化
pip install tensorboard==2.13.0
```

#### BEAR 建筑环境依赖（如果使用建筑环境）

```bash
# 方式 1: 使用安装脚本（推荐）
python scripts/install_bear_deps.py

# 方式 2: 手动安装
pip install pvlib==0.9.5
pip install scikit-learn==1.3.0
pip install cvxpy==1.3.2
```

### 2.4 验证安装

```bash
# 测试核心依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tianshou; print(f'Tianshou: {tianshou.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 测试 CUDA（如果使用 GPU）
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 测试 BEAR 依赖（如果安装）
python -c "import pvlib; import cvxpy; print('BEAR dependencies OK')"
```

### 2.5 快速验证脚本

```bash
# 测试数据中心环境
python scripts/test_datacenter_env.py

# 测试建筑环境（需要 BEAR 依赖）
python scripts/test_building_env_basic.py
```

---

## 3. 数据准备

### 3.1 数据中心场景

#### 3.1.1 使用模拟数据（推荐入门）

数据中心环境**不需要外部数据**，可以直接使用内置的模拟数据：

```bash
# 直接开始训练，环境会自动生成模拟数据
python main_datacenter.py --bc-coef --epoch 1000
```

#### 3.1.2 生成自定义模拟数据（可选）

如果需要更真实的气象和负载数据：

```bash
# 生成一年的气象和负载轨迹数据
python scripts/generate_data.py
```

这将在 `data/` 目录下生成：
- `weather_data.csv`: 气象数据（温度、湿度）
- `workload_trace.csv`: IT 负载轨迹

**数据格式示例**:

`weather_data.csv`:
```csv
timestamp,temperature,humidity
2024-01-01 00:00:00,15.2,65.3
2024-01-01 00:05:00,15.1,65.5
...
```

`workload_trace.csv`:
```csv
timestamp,load
2024-01-01 00:00:00,180.5
2024-01-01 00:05:00,182.3
...
```

#### 3.1.3 使用真实数据

如果有真实的数据中心数据，按照上述格式准备 CSV 文件，然后在训练时指定：

```bash
python main_datacenter.py \
    --use-real-weather \
    --weather-file data/your_weather.csv \
    --workload-file data/your_workload.csv \
    --bc-coef \
    --epoch 50000
```

### 3.2 建筑环境场景

#### 3.2.1 使用 BEAR 内置数据（推荐）

BEAR 环境自带了多种建筑和气候的真实数据，**无需额外准备**：

```bash
# 直接使用内置数据训练
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --location Tucson \
    --epoch 10000
```

#### 3.2.2 支持的建筑类型

BEAR 数据位于 `BEAR/BEAR/Data/` 目录，包含：

**建筑类型** (`--building-type`):
- `OfficeSmall`: 小型办公楼
- `Hospital`: 医院
- `SchoolPrimary`: 小学
- `Hotel`: 酒店
- `Warehouse`: 仓库

**气候类型** (`--weather-type`):
- `Hot_Dry`: 炎热干燥（如亚利桑那）
- `Hot_Humid`: 炎热潮湿（如佛罗里达）
- `Cold_Humid`: 寒冷潮湿（如纽约）
- `Mixed_Humid`: 混合潮湿

**地理位置** (`--location`):
- `Tucson`: 图森（亚利桑那）
- `Tampa`: 坦帕（佛罗里达）
- `Rochester`: 罗切斯特（纽约）
- 等等

### 3.3 数据目录结构

```
DROPT/
├── data/                    # 数据中心数据（可选）
│   ├── weather_data.csv
│   └── workload_trace.csv
├── BEAR/BEAR/Data/          # BEAR 建筑数据（内置）
│   ├── OfficeSmall/
│   ├── Hospital/
│   └── ...
└── log/                     # 训练日志和模型（自动创建）
    ├── datacenter_*/
    └── building_*/
```

---

## 4. 配置文件

### 4.1 数据中心配置

#### 4.1.1 预定义配置

项目提供了三种预定义配置（在 `env/datacenter_config.py`）：

**小型数据中心** (`SMALL_DATACENTER`):
```python
num_crac = 2              # 2 个 CRAC 单元
target_temp = 24.0        # 目标温度 24°C
it_load_max = 100.0       # 最大负载 100kW
```

**中型数据中心** (`MEDIUM_DATACENTER`):
```python
num_crac = 4              # 4 个 CRAC 单元
target_temp = 24.0        # 目标温度 24°C
it_load_max = 500.0       # 最大负载 500kW
```

**大型数据中心** (`LARGE_DATACENTER`):
```python
num_crac = 8              # 8 个 CRAC 单元
target_temp = 24.0        # 目标温度 24°C
it_load_max = 2000.0      # 最大负载 2MW
```

#### 4.1.2 关键参数说明

**环境参数**:
- `--num-crac`: CRAC 空调单元数量（默认 4）
- `--target-temp`: 目标温度，单位°C（默认 24.0）
- `--temp-tolerance`: 温度容差，单位°C（默认 2.0）
- `--episode-length`: 回合长度，单位步数（默认 288 = 24小时）

**奖励函数权重**:
- `--energy-weight`: 能耗权重 α（默认 1.0）
- `--temp-weight`: 温度偏差权重 β（默认 10.0）
- `--violation-penalty`: 温度越界惩罚 γ（默认 100.0）

**训练参数**:
- `--epoch`: 训练轮数（BC: 50000, PG: 200000）
- `--batch-size`: 批次大小（BC: 256, PG: 512）
- `--actor-lr`: Actor 学习率（默认 3e-4）
- `--critic-lr`: Critic 学习率（默认 3e-4）
- `--gamma`: 折扣因子（默认 0.99）
- `--tau`: 目标网络软更新系数（默认 0.005）

**扩散模型参数**:
- `--diffusion-steps`: 扩散步数（默认 5，越大越精确但越慢）
- `--beta-schedule`: 噪声调度（'vp'/'linear'/'cosine'）
- `--exploration-noise`: 探索噪声标准差（默认 0.1）

**训练模式**:
- `--bc-coef`: 启用行为克隆模式（需要专家控制器）
- `--expert-type`: 专家类型（'pid'/'mpc'/'rule_based'）

### 4.2 建筑环境配置

#### 4.2.1 关键参数说明

**建筑和气候**:
- `--building-type`: 建筑类型（见 3.2.2 节）
- `--weather-type`: 气候类型（见 3.2.2 节）
- `--location`: 地理位置（见 3.2.2 节）

**HVAC 参数**:
- `--target-temp`: 目标温度，单位°C（默认 22.0）
- `--temp-tolerance`: 温度容差，单位°C（默认 2.0）
- `--max-power`: HVAC 最大功率，单位 W（默认 8000）
- `--time-resolution`: 时间分辨率，单位秒（默认 3600 = 1小时）

**奖励函数**:
- `--energy-weight`: 能耗权重（默认 1.0）
- `--temp-weight`: 温度偏差权重（默认 10.0）
- `--add-violation-penalty`: 是否添加越界惩罚（默认 True）
- `--violation-penalty`: 越界惩罚值（默认 100.0）

**训练参数**: 与数据中心相同

### 4.3 配置示例

#### 示例 1: 快速演示配置
```bash
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --epoch 1000 \
    --batch-size 128 \
    --diffusion-steps 3 \
    --episode-length 50 \
    --device cpu
```

#### 示例 2: 标准训练配置
```bash
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --num-crac 4 \
    --epoch 50000 \
    --batch-size 256 \
    --diffusion-steps 5 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --device cuda:0
```

#### 示例 3: 高性能配置
```bash
python main_datacenter.py \
    --num-crac 4 \
    --epoch 200000 \
    --batch-size 512 \
    --diffusion-steps 8 \
    --gamma 0.99 \
    --actor-lr 1e-4 \
    --critic-lr 3e-4 \
    --prioritized-replay \
    --device cuda:0
```

---

## 5. 开始训练

### 5.1 数据中心训练

#### 5.1.1 行为克隆模式（推荐入门）

**特点**: 从专家控制器学习，收敛快，适合快速验证

```bash
# 基础训练（使用 PID 专家）
python main_datacenter.py \
    --bc-coef \
    --expert-type pid \
    --epoch 50000 \
    --device cuda:0

# 使用 MPC 专家（更优但更慢）
python main_datacenter.py \
    --bc-coef \
    --expert-type mpc \
    --epoch 50000 \
    --device cuda:0
```

#### 5.1.2 策略梯度模式（追求性能）

**特点**: 通过环境交互学习，训练时间长，性能更好

```bash
python main_datacenter.py \
    --epoch 200000 \
    --batch-size 512 \
    --diffusion-steps 8 \
    --gamma 0.99 \
    --device cuda:0
```

#### 5.1.3 混合模式（推荐）

**策略**: 先用 BC 快速收敛，再用 PG 精调

```bash
# 阶段 1: BC 预训练
python main_datacenter.py \
    --bc-coef \
    --expert-type mpc \
    --epoch 30000 \
    --log-prefix bc_pretrain \
    --device cuda:0

# 阶段 2: PG 精调（加载预训练模型）
python main_datacenter.py \
    --resume-path log/bc_pretrain_*/policy_best.pth \
    --epoch 100000 \
    --batch-size 512 \
    --log-prefix pg_finetune \
    --device cuda:0
```

### 5.2 建筑环境训练

#### 5.2.1 基础训练

```bash
# 小型办公楼 + 炎热干燥气候
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --location Tucson \
    --episode-length 288 \
    --epoch 10000 \
    --device cuda:0
```

#### 5.2.2 使用专家控制器

```bash
# 使用 MPC 专家进行行为克隆
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --expert-type mpc \
    --bc-coef \
    --epoch 50000 \
    --device cuda:0
```

#### 5.2.3 多场景训练

```bash
# 医院 + 寒冷潮湿气候
python main_building.py \
    --building-type Hospital \
    --weather-type Cold_Humid \
    --location Rochester \
    --target-temp 22.0 \
    --temp-tolerance 1.5 \
    --epoch 20000 \
    --device cuda:0
```

### 5.3 命令行参数完整示例

```bash
python main_datacenter.py \
    --num-crac 4 \                    # 4 个 CRAC 单元
    --target-temp 24.0 \              # 目标温度 24°C
    --temp-tolerance 2.0 \            # 容差 ±2°C
    --episode-length 288 \            # 24 小时回合
    --energy-weight 1.0 \             # 能耗权重
    --temp-weight 10.0 \              # 温度权重
    --violation-penalty 100.0 \       # 越界惩罚
    --bc-coef \                       # 启用行为克隆
    --expert-type pid \               # 使用 PID 专家
    --epoch 50000 \                   # 训练 50000 轮
    --batch-size 256 \                # 批次大小 256
    --actor-lr 3e-4 \                 # Actor 学习率
    --critic-lr 3e-4 \                # Critic 学习率
    --gamma 0.99 \                    # 折扣因子
    --tau 0.005 \                     # 软更新系数
    --diffusion-steps 5 \             # 扩散步数
    --beta-schedule vp \              # 噪声调度
    --hidden-dim 256 \                # 隐藏层维度
    --training-num 4 \                # 4 个并行训练环境
    --test-num 2 \                    # 2 个测试环境
    --buffer-size 1000000 \           # 经验回放缓冲区大小
    --step-per-epoch 5000 \           # 每轮步数
    --step-per-collect 100 \          # 每次收集步数
    --save-interval 10 \              # 每 10 轮保存一次
    --logdir log \                    # 日志目录
    --log-prefix datacenter \         # 日志前缀
    --device cuda:0 \                 # 使用 GPU 0
    --seed 42                         # 随机种子
```

### 5.4 训练脚本入口文件

- **数据中心**: `main_datacenter.py`
- **建筑环境**: `main_building.py`

两个脚本的参数大部分相同，主要区别在于环境特定参数。

---

## 6. 训练过程监控

### 6.1 TensorBoard 可视化

#### 6.1.1 启动 TensorBoard

```bash
# 监控所有训练日志
tensorboard --logdir log

# 监控特定训练
tensorboard --logdir log/datacenter_20240115_143022

# 指定端口
tensorboard --logdir log --port 6007
```

然后在浏览器打开: `http://localhost:6006`

#### 6.1.2 关键指标

**训练指标**:
- `train/reward`: 训练奖励（越高越好）
- `train/length`: 回合长度
- `loss/critic`: Critic 损失（应逐渐下降）
- `loss/actor` 或 `overall_loss`: Actor 损失

**测试指标**:
- `test/reward`: 测试奖励（评估性能）
- `test/reward_std`: 奖励标准差（评估稳定性）

**环境指标**（如果记录）:
- `env/energy_consumption`: 能耗
- `env/temperature_violation`: 温度违规次数
- `env/average_temperature`: 平均温度

### 6.2 模型检查点

#### 6.2.1 保存位置

训练过程中，模型会自动保存到日志目录：

```
log/
└── datacenter_20240115_143022/      # 训练会话目录
    ├── events.out.tfevents.*        # TensorBoard 日志
    ├── policy_best.pth              # 最佳模型（测试奖励最高）
    ├── policy_final.pth             # 最终模型
    ├── checkpoint_10.pth            # 定期检查点
    ├── checkpoint_20.pth
    └── ...
```

#### 6.2.2 检查点内容

```python
checkpoint = {
    'model': policy.state_dict(),           # 策略网络参数
    'optim_actor': actor_optim.state_dict(),   # Actor 优化器状态
    'optim_critic': critic_optim.state_dict(), # Critic 优化器状态
}
```

#### 6.2.3 加载模型

```bash
# 从检查点恢复训练
python main_datacenter.py \
    --resume-path log/datacenter_*/policy_best.pth \
    --epoch 100000 \
    --device cuda:0

# 评估模型
python main_datacenter.py \
    --watch \
    --resume-path log/datacenter_*/policy_best.pth \
    --test-num 10
```

### 6.3 命令行输出

#### 6.3.1 训练开始信息

```
======================================================================
数据中心空调优化 - 基于扩散模型的强化学习
======================================================================

[1/6] 创建数据中心环境...
  ✓ 环境创建成功
  - CRAC 单元数: 4
  - 状态维度: 13
  - 动作维度: 4
  - 目标温度: 24.0°C ± 2.0°C

[2/6] 创建神经网络...
  ✓ Actor (扩散模型): MLP(state_dim=13, action_dim=4, hidden_dim=256)
  ✓ Critic (双Q网络): DoubleCritic(state_dim=13, action_dim=4)

[3/6] 初始化专家控制器...
  ✓ 专家类型: PID Controller

[4/6] 创建经验回放缓冲区...
  ✓ 缓冲区大小: 1000000

[5/6] 初始化DiffusionOPT策略...
  ✓ 训练模式: 行为克隆 (BC)
  ✓ 扩散步数: 5
  ✓ 噪声调度: vp

[6/6] 开始训练...
```

#### 6.3.2 训练过程输出

```
Epoch #1: 5000it [01:23, 59.88it/s, env_step=5000, len=288, loss=0.245, n/ep=17, n/st=5000, rew=1234.56]
Epoch #2: 5000it [01:22, 60.12it/s, env_step=10000, len=288, loss=0.198, n/ep=17, n/st=5000, rew=1456.78]
...
```

**字段说明**:
- `env_step`: 总环境步数
- `len`: 平均回合长度
- `loss`: 平均损失
- `n/ep`: 本轮回合数
- `n/st`: 本轮步数
- `rew`: 平均奖励

### 6.4 日志文件

#### 6.4.1 日志目录结构

```
log/
├── datacenter_20240115_143022/
│   ├── events.out.tfevents.1705305022.hostname
│   ├── policy_best.pth
│   ├── policy_final.pth
│   └── checkpoint_*.pth
└── building_OfficeSmall_Hot_Dry_20240115_150000/
    └── ...
```

#### 6.4.2 查看日志

```bash
# 列出所有训练会话
ls -lt log/

# 查看最新训练
ls -lt log/ | head -n 2

# 查找最佳模型
find log/ -name "policy_best.pth"
```

### 6.5 实时监控脚本

创建一个简单的监控脚本 `monitor_training.py`:

```python
import os
import time
from tensorboard.backend.event_processing import event_accumulator

def monitor_latest_run(logdir='log'):
    # 找到最新的训练目录
    runs = sorted([os.path.join(logdir, d) for d in os.listdir(logdir)], 
                  key=os.path.getmtime, reverse=True)
    latest_run = runs[0]
    
    print(f"监控训练: {latest_run}")
    
    ea = event_accumulator.EventAccumulator(latest_run)
    ea.Reload()
    
    while True:
        ea.Reload()
        
        # 获取最新指标
        if 'train/reward' in ea.Tags()['scalars']:
            rewards = ea.Scalars('train/reward')
            if rewards:
                latest = rewards[-1]
                print(f"Step {latest.step}: Reward = {latest.value:.2f}")
        
        time.sleep(10)  # 每 10 秒更新一次

if __name__ == '__main__':
    monitor_latest_run()
```

---

## 7. 常见问题

### 7.1 安装问题

#### Q0: ModuleNotFoundError: No module named 'tianshou'

**问题**: 运行训练时提示找不到模块

**解决方案**:
```bash
# 确保激活了正确的 conda 环境
conda activate dropt

# 或者如果使用 venv
source dropt_env/bin/activate  # Linux/Mac
dropt_env\Scripts\activate     # Windows

# 验证环境
python -c "import tianshou; print('OK')"

# 如果仍然报错，重新安装依赖
pip install tianshou==0.4.11
```

#### Q1: PyTorch 安装失败

**问题**: `pip install torch` 下载速度慢或失败

**解决方案**:
```bash
# 使用清华镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用官方 CUDA 版本链接
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```

#### Q2: Tianshou 版本不兼容

**问题**: `ImportError: cannot import name 'BasePolicy'`

**解决方案**:
```bash
# 确保使用正确版本
pip uninstall tianshou
pip install tianshou==0.4.11
```

#### Q3: BEAR 依赖安装失败

**问题**: `cvxpy` 或 `pvlib` 安装失败

**解决方案**:
```bash
# 先安装编译依赖
pip install --upgrade pip setuptools wheel

# 再安装 BEAR 依赖
pip install pvlib scikit-learn cvxpy

# 如果仍然失败，尝试 conda
conda install -c conda-forge cvxpy pvlib-python
```

### 7.2 训练问题

#### Q3.5: TypeError: DataCenterEnv.reset() got an unexpected keyword argument 'seed'

**问题**: 运行训练时提示 `reset()` 方法不接受 `seed` 参数

**原因**: 这是 Gym/Gymnasium API 兼容性问题。Tianshou 使用新版 Gymnasium API，需要环境的 `reset()` 方法支持 `seed` 参数。

**解决方案**:
此问题已在最新代码中修复。如果您仍然遇到此问题，请确保 `env/datacenter_env.py` 中的 `reset()` 方法签名如下：

```python
def reset(self, seed=None, options=None):
    """重置环境"""
    if seed is not None:
        np.random.seed(seed)
    # ... 其他代码
    return self._state, info  # 返回 (state, info) 元组
```

如果您修改了环境代码，请确保：
1. `reset()` 方法接受 `seed` 和 `options` 参数（可选）
2. 返回 `(observation, info)` 元组，而不是单独的 observation

#### Q3.6: TypeError: MLP.__init__() got an unexpected keyword argument 'hidden_sizes'

**问题**: 运行训练时提示 MLP 初始化参数错误

**原因**: MLP 类的 `__init__` 方法接受 `hidden_dim`（单个整数），而不是 `hidden_sizes`（列表）。

**解决方案**:
此问题已在最新代码中修复。如果您仍然遇到此问题：

1. 确保使用 `--hidden-dim` 参数（而不是 `--hidden-sizes`）：
```bash
python main_datacenter.py --hidden-dim 256  # 正确
```

2. 如果您修改了代码，确保 MLP 调用使用正确的参数名：
```python
actor_net = MLP(
    state_dim=args.state_shape,
    action_dim=args.action_shape,
    hidden_dim=args.hidden_dim  # 使用 hidden_dim，不是 hidden_sizes
)
```

#### Q3.7: RuntimeError: Numpy is not available

**问题**: 运行训练时提示 `RuntimeError: Numpy is not available` 或类似的 NumPy/PyTorch 转换错误

**原因**: 在 `diffusion/diffusion.py` 中混用了 NumPy 和 PyTorch 操作。新版 PyTorch 不允许直接对 GPU tensor 使用 NumPy 函数。

**解决方案**:
此问题已在最新代码中修复。如果您仍然遇到此问题，请确保所有 tensor 操作都使用 PyTorch 函数：

```python
# 错误：对 PyTorch tensor 使用 np.sqrt()
coef = betas * np.sqrt(alphas_cumprod_prev)  # ❌

# 正确：使用 torch.sqrt()
coef = betas * torch.sqrt(alphas_cumprod_prev)  # ✅
```

如果您修改了扩散模型代码，请检查：
1. 所有数学运算使用 `torch.*` 而不是 `np.*`
2. 确保 tensor 在正确的设备上（CPU 或 GPU）
3. 避免在 GPU tensor 上调用 `.numpy()`

#### Q4: CUDA out of memory

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python main_datacenter.py --batch-size 128  # 默认 256

# 减少并行环境数
python main_datacenter.py --training-num 2  # 默认 4

# 减小网络规模
python main_datacenter.py --hidden-dim 128  # 默认 256

# 使用 CPU
python main_datacenter.py --device cpu
```

#### Q5: 训练不收敛

**问题**: 奖励不增长或震荡

**解决方案**:

1. **检查学习率**:
```bash
# 降低学习率
python main_datacenter.py --actor-lr 1e-4 --critic-lr 1e-4
```

2. **使用行为克隆预训练**:
```bash
# 先用 BC 训练
python main_datacenter.py --bc-coef --expert-type pid --epoch 30000
```

3. **调整奖励权重**:
```bash
# 增加温度权重
python main_datacenter.py --temp-weight 20.0
```

4. **增加训练步数**:
```bash
# 延长训练
python main_datacenter.py --epoch 100000
```

#### Q6: 动作卡在边界

**问题**: 动作总是 -1 或 1

**解决方案**:

1. **调整探索噪声**:
```bash
python main_datacenter.py --exploration-noise 0.2  # 默认 0.1
```

2. **增加扩散步数**:
```bash
python main_datacenter.py --diffusion-steps 8  # 默认 5
```

3. **检查奖励函数**: 确保奖励函数设计合理

### 7.3 环境问题

#### Q7: 建筑环境创建失败

**问题**: `RuntimeError: 生成 BEAR 参数失败`

**解决方案**:

1. **检查 BEAR 数据**:
```bash
# 确保 BEAR 数据存在
ls BEAR/BEAR/Data/

# 重新克隆 BEAR
cd BEAR
git pull
```

2. **检查参数组合**:
```bash
# 使用已知有效的组合
python main_building.py \
    --building-type OfficeSmall \
    --weather-type Hot_Dry \
    --location Tucson
```

#### Q8: 数据中心环境状态异常

**问题**: 温度或能耗值异常

**解决方案**:

1. **检查配置**:
```python
# 查看环境配置
python -c "from env.datacenter_config import get_config; print(get_config('medium'))"
```

2. **运行测试**:
```bash
python scripts/test_datacenter_env.py
```

### 7.4 性能问题

#### Q9: 训练速度慢

**问题**: 训练速度低于预期

**解决方案**:

1. **使用 GPU**:
```bash
python main_datacenter.py --device cuda:0
```

2. **减少扩散步数**:
```bash
python main_datacenter.py --diffusion-steps 3  # 默认 5
```

3. **增加并行环境**:
```bash
python main_datacenter.py --training-num 8  # 默认 4
```

4. **减少测试频率**:
```bash
# 修改 main_datacenter.py 中的 test_in_train=False
```

#### Q10: TensorBoard 占用内存过大

**问题**: TensorBoard 内存占用高

**解决方案**:
```bash
# 只加载最近的日志
tensorboard --logdir log/datacenter_latest --reload_interval 30

# 限制加载的数据点
tensorboard --logdir log --samples_per_plugin scalars=1000
```

### 7.5 模型评估问题

#### Q11: 如何评估训练好的模型

**解决方案**:
```bash
# 评估模式（不训练，只测试）
python main_datacenter.py \
    --watch \
    --resume-path log/datacenter_*/policy_best.pth \
    --test-num 20 \
    --device cuda:0
```

#### Q12: 如何比较不同模型

**解决方案**:

创建评估脚本 `evaluate_models.py`:
```python
import torch
from env.datacenter_env import make_datacenter_env
from policy import DiffusionOPT

def evaluate_model(model_path, num_episodes=10):
    # 加载模型
    policy = torch.load(model_path)
    
    # 创建环境
    env, _, _ = make_datacenter_env(test_num=1)
    
    # 评估
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = policy.forward(obs).act
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards

# 比较多个模型
models = [
    'log/model1/policy_best.pth',
    'log/model2/policy_best.pth',
]

for model_path in models:
    print(f"\n评估: {model_path}")
    evaluate_model(model_path)
```

### 7.6 其他问题

#### Q13: 如何调试代码

**解决方案**:

1. **使用小规模配置快速测试**:
```bash
python main_datacenter.py \
    --epoch 100 \
    --episode-length 10 \
    --batch-size 32 \
    --device cpu
```

2. **启用详细日志**:
```python
# 在代码中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **使用 Python 调试器**:
```bash
python -m pdb main_datacenter.py --epoch 100
```

#### Q14: 如何获取帮助

**解决方案**:

1. **查看命令行帮助**:
```bash
python main_datacenter.py --help
python main_building.py --help
```

2. **查看文档**:
```bash
# 查看所有文档
ls docs/

# 阅读相关文档
cat docs/DATACENTER_SUMMARY.md
cat docs/ARCHITECTURE.md
```

3. **运行测试脚本**:
```bash
# 测试环境
python scripts/test_datacenter_env.py
python scripts/test_building_env_basic.py

# 快速测试
python scripts/quick_test.py
```

4. **查看示例**:
```bash
# 运行演示
python scripts/demo_building_env.py
```

---

## 附录

### A. 完整依赖列表

```txt
# 核心依赖
torch>=1.8.0
tianshou==0.4.11
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
gym==0.21.0
gymnasium>=0.28.0
tensorboard>=2.8.0

# BEAR 建筑环境依赖（可选）
pvlib>=0.9.0
scikit-learn>=1.0.0
cvxpy>=1.2.0
```

### B. 推荐训练配置

| 场景 | 模式 | Epoch | Batch Size | 扩散步数 | 预计时间 | 预期性能 |
|------|------|-------|------------|----------|----------|----------|
| 快速验证 | BC | 1,000 | 128 | 3 | 5 分钟 | 低 |
| 标准训练 | BC | 50,000 | 256 | 5 | 1 小时 | 中 |
| 高性能 | PG | 200,000 | 512 | 8 | 6 小时 | 高 |
| 混合模式 | BC→PG | 30k+100k | 256→512 | 5→8 | 3 小时 | 最高 |

### C. 参考资源

- **论文**: [Enhancing Deep Reinforcement Learning: A Tutorial on Generative Diffusion Models in Network Optimization](https://arxiv.org/abs/2308.05384)
- **BEAR 项目**: [https://github.com/chz056/BEAR](https://github.com/chz056/BEAR)
- **Tianshou 文档**: [https://tianshou.readthedocs.io/](https://tianshou.readthedocs.io/)
- **项目文档**: `docs/` 目录下的其他文档

---

**祝您训练顺利！如有问题，请参考常见问题部分或查看项目文档。**

