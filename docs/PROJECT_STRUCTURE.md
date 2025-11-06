# DROPT 项目结构说明

**最后更新**: 2025-11-06  
**版本**: 1.0

---

## 📁 项目目录结构

```
DROPT/
│
├── README.md                          # 项目主README
├── main.py                            # 原始DROPT训练程序（无线网络功率分配）
├── main_datacenter.py                 # 数据中心空调优化训练程序
├── __init__.py                        # Python包初始化文件
├── index.html                         # 项目网页（如果有）
│
├── docs/                              # 📚 文档中心
│   ├── README.md                      # 文档导航
│   ├── PROJECT_STRUCTURE.md           # 本文档：项目结构说明
│   │
│   ├── GET_STARTED.md                 # 快速开始指南
│   ├── REAL_DATA_QUICK_START.md       # 真实数据快速开始
│   │
│   ├── DATACENTER_SUMMARY.md          # 数据中心项目总结
│   ├── ARCHITECTURE.md                # 系统架构详细说明
│   ├── README_DATACENTER.md           # 数据中心完整使用手册
│   ├── MIGRATION_GUIDE.md             # 迁移指南
│   │
│   ├── REAL_DATA_INTEGRATION_SUMMARY.md      # 真实数据集成总结
│   ├── REAL_DATA_INTEGRATION_GUIDE.md        # 真实数据集成完整指南
│   ├── DATA_RELIABILITY_ANALYSIS.md          # 数据可靠性分析
│   │
│   └── DELIVERY_CHECKLIST.md          # 项目交付清单
│
├── env/                               # 🌍 环境代码
│   ├── __init__.py
│   ├── env.py                         # 原始AIGC网络环境
│   ├── utility.py                     # 水注入算法等工具
│   │
│   ├── datacenter_env.py              # 数据中心环境
│   ├── datacenter_env_robust.py       # 鲁棒性增强环境
│   ├── datacenter_config.py           # 数据中心配置
│   ├── thermal_model.py               # 热力学模型
│   ├── expert_controller.py           # 专家控制器（PID/MPC/RBC）
│   └── real_data_loader.py            # 真实数据加载器
│
├── scripts/                           # 🛠️ 工具脚本
│   ├── generate_data.py               # 数据生成工具
│   ├── preprocess_real_data.py        # 真实数据预处理
│   ├── calibrate_model.py             # 模型参数校准
│   ├── sensitivity_analysis.py        # 敏感性分析
│   ├── fetch_real_weather.py          # 真实气象数据获取
│   ├── test_datacenter_env.py         # 环境测试脚本
│   │
│   ├── quick_start.sh                 # 快速开始脚本（Linux/Mac）
│   ├── quick_start.bat                # 快速开始脚本（Windows）
│   ├── example_real_data_workflow.sh  # 真实数据工作流（Linux/Mac）
│   └── example_real_data_workflow.bat # 真实数据工作流（Windows）
│
├── policy/                            # 🎯 策略代码
│   ├── __init__.py
│   ├── diffusion_opt.py               # DiffusionOPT策略（核心）
│   ├── helpers.py                     # 辅助函数
│   └── random.py                      # 随机策略（基线）
│
├── diffusion/                         # 🌊 扩散模型
│   ├── __init__.py
│   ├── diffusion.py                   # 扩散模型核心实现（DDPM）
│   ├── model.py                       # 网络架构（MLP, DoubleCritic）
│   ├── helpers.py                     # 辅助函数
│   └── utils.py                       # 工具函数
│
├── data/                              # 💾 数据文件
│   ├── data_format_template.csv       # 数据格式模板
│   ├── real_data_processed.csv        # 处理后的真实数据（生成）
│   └── realistic_weather.csv          # 真实感气象数据（生成）
│
├── Software/                          # 🖥️ GUI版本
│   ├── main.py                        # GUI主程序
│   └── parameter_gui.py               # 参数配置界面
│
├── images/                            # 🖼️ 图片资源
│   ├── 0.jpg                          # 教程结构图
│   ├── 1.png                          # GDM训练方法图
│   ├── 3.png                          # 优化问题公式
│   ├── 7.png                          # TensorBoard示例
│   └── show.png                       # GUI界面截图
│
├── static/                            # 🌐 静态资源（网页）
│   ├── css/
│   ├── images/
│   └── js/
│
├── log/                               # 📊 训练日志
│   └── default/                       # 默认日志目录
│
└── __pycache__/                       # Python缓存（自动生成）
```

---

## 📂 核心目录说明

### **1. `docs/` - 文档中心**

所有项目文档集中存放，便于查找和维护。

**文档分类**:
- **快速开始**: GET_STARTED.md, REAL_DATA_QUICK_START.md
- **核心文档**: DATACENTER_SUMMARY.md, ARCHITECTURE.md, README_DATACENTER.md
- **迁移开发**: MIGRATION_GUIDE.md, DELIVERY_CHECKLIST.md
- **真实数据**: REAL_DATA_INTEGRATION_*.md, DATA_RELIABILITY_ANALYSIS.md

**导航**: 查看 [docs/README.md](README.md) 获取完整文档导航

---

### **2. `env/` - 环境代码**

包含所有环境相关的代码。

**原始环境**:
- `env.py` - AIGC网络环境（无线网络功率分配）
- `utility.py` - 水注入算法等工具

**数据中心环境**:
- `datacenter_env.py` - 标准数据中心环境
- `datacenter_env_robust.py` - 鲁棒性增强版本（域随机化、噪声等）
- `datacenter_config.py` - 配置管理
- `thermal_model.py` - 物理驱动的热力学模型
- `expert_controller.py` - 专家控制器（PID/MPC/RBC）
- `real_data_loader.py` - 真实数据加载器

---

### **3. `scripts/` - 工具脚本**

各种实用工具脚本。

**数据处理**:
- `generate_data.py` - 生成合成数据
- `preprocess_real_data.py` - 预处理真实数据
- `fetch_real_weather.py` - 获取真实气象数据

**模型工具**:
- `calibrate_model.py` - 校准模型参数
- `sensitivity_analysis.py` - 敏感性分析
- `test_datacenter_env.py` - 测试环境

**工作流**:
- `quick_start.sh/.bat` - 快速开始
- `example_real_data_workflow.sh/.bat` - 真实数据完整工作流

---

### **4. `policy/` - 策略代码**

强化学习策略实现。

- `diffusion_opt.py` - DiffusionOPT策略（核心）
  - Actor: 扩散模型
  - Critic: 双Q网络
  - 支持BC（行为克隆）和PG（策略梯度）

- `random.py` - 随机策略（基线对比）

---

### **5. `diffusion/` - 扩散模型**

扩散模型核心实现。

- `diffusion.py` - DDPM（去噪扩散概率模型）
  - 前向扩散过程
  - 反向去噪过程
  - 损失函数计算

- `model.py` - 网络架构
  - MLP: 多层感知机（Actor网络）
  - DoubleCritic: 双Q网络（Critic网络）

---

### **6. `data/` - 数据文件**

数据文件存放目录。

- `data_format_template.csv` - 数据格式模板
- 其他数据文件（训练过程中生成）

---

### **7. `Software/` - GUI版本**

带图形界面的训练程序。

- `main.py` - GUI主程序
- `parameter_gui.py` - 参数配置界面

**使用方式**: 将这两个文件复制到项目根目录，替换原 main.py

---

## 🔗 文件关联关系

### **训练流程**

```
main_datacenter.py
    ├─> env/datacenter_env.py
    │       ├─> env/thermal_model.py
    │       ├─> env/expert_controller.py
    │       └─> env/real_data_loader.py (可选)
    │
    ├─> policy/diffusion_opt.py
    │       ├─> diffusion/diffusion.py
    │       │       └─> diffusion/model.py (MLP)
    │       └─> diffusion/model.py (DoubleCritic)
    │
    └─> scripts/calibrate_model.py (可选，预先校准)
```

### **数据处理流程**

```
原始数据 (your_data.csv)
    ↓
scripts/preprocess_real_data.py
    ↓
处理后数据 (data/real_data_processed.csv)
    ↓
scripts/calibrate_model.py
    ↓
校准参数 (results/calibrated_params.json)
    ↓
main_datacenter.py (训练)
```

---

## 📝 使用建议

### **新手用户**

1. 阅读 [docs/GET_STARTED.md](GET_STARTED.md)
2. 运行 `scripts/quick_start.sh` 或 `scripts/quick_start.bat`
3. 查看 TensorBoard 监控训练

### **有真实数据的用户**

1. 阅读 [docs/REAL_DATA_QUICK_START.md](REAL_DATA_QUICK_START.md)
2. 运行 `scripts/example_real_data_workflow.sh` 或 `.bat`
3. 查看 [docs/REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) 了解详细步骤

### **开发者**

1. 阅读 [docs/ARCHITECTURE.md](ARCHITECTURE.md) 了解架构
2. 阅读 [docs/MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) 了解迁移过程
3. 根据需求修改代码

---

## 🔧 维护说明

### **添加新文档**

1. 在 `docs/` 目录下创建新的 `.md` 文件
2. 更新 `docs/README.md` 添加导航链接
3. 如需要，更新根目录 `README.md`

### **添加新脚本**

1. 在 `scripts/` 目录下创建新的 `.py` 文件
2. 添加详细的中文注释
3. 在相关文档中添加使用说明

### **添加新环境**

1. 在 `env/` 目录下创建新的环境文件
2. 继承 `gym.Env` 或现有环境类
3. 在 `main_datacenter.py` 中添加支持
4. 编写测试脚本

---

## ⚠️ 注意事项

### **不要修改的文件**

- `__pycache__/` - Python自动生成的缓存
- `log/` - 训练日志（可以清理但不要删除目录）
- `.gitignore` - Git忽略规则

### **可以安全删除的文件**

- `__pycache__/` 目录（会自动重新生成）
- `log/` 目录下的旧日志
- `data/` 目录下的临时数据文件

### **重要文件备份**

建议定期备份：
- 训练好的模型（`log/*/policy_best.pth`）
- 校准参数（`results/calibrated_params.json`）
- 真实数据（`data/real_data_processed.csv`）

---

## 📞 获取帮助

如果对项目结构有疑问：

1. 查看 [docs/README.md](README.md) 文档导航
2. 查看相关文档的详细说明
3. 提交 Issue 或联系项目维护者

---

**返回**: [文档中心](README.md) | [项目主页](../README.md)

