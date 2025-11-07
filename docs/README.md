# DROPT 项目文档中心

欢迎来到DROPT项目文档中心！本目录包含了项目的所有详细文档。

---

## 📚 文档导航

### 🚀 快速开始

| 文档 | 说明 | 适用人群 |
|------|------|---------|
| [GET_STARTED.md](GET_STARTED.md) | 5分钟快速开始指南 | 新手用户 |
| [REAL_DATA_QUICK_START.md](REAL_DATA_QUICK_START.md) | 真实数据集成快速开始 | 有真实数据的用户 |

---

### 📖 核心文档

#### **项目概览**

| 文档 | 说明 |
|------|------|
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目结构说明（推荐首先阅读） |
| [DATACENTER_SUMMARY.md](DATACENTER_SUMMARY.md) | 数据中心空调优化项目总结 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构详细说明 |
| [README_DATACENTER.md](README_DATACENTER.md) | 数据中心项目完整使用手册 |

#### **迁移与开发**

| 文档 | 说明 |
|------|------|
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | 从无线网络到数据中心的迁移指南 |
| [DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md) | 项目交付清单 |

---

### 🔬 真实数据集成（已移除功能，文档仅供参考）

> ⚠️ **注意**: 项目已简化，不再使用真实数据集成功能。以下文档仅作为历史参考保留。

| 文档 | 说明 | 状态 |
|------|------|------|
| [REAL_DATA_INTEGRATION_SUMMARY.md](REAL_DATA_INTEGRATION_SUMMARY.md) | 真实数据集成方案总结 | 📚 参考 |
| [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) | 真实数据集成完整指南 | 📚 参考 |
| [REAL_DATA_QUICK_START.md](REAL_DATA_QUICK_START.md) | 真实数据快速开始 | 📚 参考 |
| [DATA_RELIABILITY_ANALYSIS.md](DATA_RELIABILITY_ANALYSIS.md) | 数据可靠性深度分析 | 📚 参考 |

---

### 🔧 代码质量与优化

| 文档 | 说明 | 适用场景 |
|------|------|---------|
| [THERMAL_MODEL_REVIEW.md](THERMAL_MODEL_REVIEW.md) | 热力学模型代码审查报告 | 了解问题和改进点 |
| [THERMAL_MODEL_OPTIMIZATION_SUMMARY.md](THERMAL_MODEL_OPTIMIZATION_SUMMARY.md) | 热力学模型优化总结 | 查看优化效果 |
| [PROJECT_CLEANUP_REPORT.md](PROJECT_CLEANUP_REPORT.md) | 项目清理报告 | 了解项目简化过程 |

---

## 🎯 按使用场景选择文档

### 场景1: 我是新手，想快速了解项目

**推荐阅读顺序**:
1. [GET_STARTED.md](GET_STARTED.md) - 5分钟快速开始
2. [DATACENTER_SUMMARY.md](DATACENTER_SUMMARY.md) - 项目概览
3. [README_DATACENTER.md](README_DATACENTER.md) - 详细使用手册

---

### 场景2: 我有真实数据，想集成到项目中

**推荐阅读顺序**:
1. [REAL_DATA_QUICK_START.md](REAL_DATA_QUICK_START.md) - 快速开始
2. [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) - 详细指南
3. [DATA_RELIABILITY_ANALYSIS.md](DATA_RELIABILITY_ANALYSIS.md) - 可靠性分析

**关键步骤**:
```bash
# 1. 数据预处理
python scripts/preprocess_real_data.py --input your_data.csv --output data/processed.csv

# 2. 模型校准
python scripts/calibrate_model.py --real-data data/processed.csv --output results/params.json

# 3. 训练
python main_datacenter.py --real-data data/processed.csv --calibrated-params results/params.json
```

---

### 场景3: 我想了解技术架构和实现细节

**推荐阅读顺序**:
1. [ARCHITECTURE.md](ARCHITECTURE.md) - 系统架构
2. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - 迁移过程
3. [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) - 实现细节

---

### 场景4: 我想评估数据质量和模型可靠性

**推荐阅读顺序**:
1. [DATA_RELIABILITY_ANALYSIS.md](DATA_RELIABILITY_ANALYSIS.md) - 可靠性分析
2. [REAL_DATA_INTEGRATION_GUIDE.md](REAL_DATA_INTEGRATION_GUIDE.md) - 校准方法
3. 运行敏感性分析工具

**关键工具**:
```bash
# 敏感性分析
python scripts/sensitivity_analysis.py --param all

# 数据质量验证
python scripts/preprocess_real_data.py --input your_data.csv --validate
```

---

## 📊 文档概览表

| 文档 | 页数 | 主要内容 | 更新日期 |
|------|------|---------|---------|
| GET_STARTED.md | 8 | 快速开始指南 | 2025-11-06 |
| DATACENTER_SUMMARY.md | 10 | 项目总结 | 2025-11-06 |
| ARCHITECTURE.md | 12 | 系统架构 | 2025-11-06 |
| README_DATACENTER.md | 15 | 完整使用手册 | 2025-11-06 |
| MIGRATION_GUIDE.md | 12 | 迁移指南 | 2025-11-06 |
| DELIVERY_CHECKLIST.md | 8 | 交付清单 | 2025-11-06 |
| REAL_DATA_INTEGRATION_SUMMARY.md | 10 | 真实数据集成总结 | 2025-11-06 |
| REAL_DATA_INTEGRATION_GUIDE.md | 15 | 真实数据集成指南 | 2025-11-06 |
| REAL_DATA_QUICK_START.md | 12 | 真实数据快速开始 | 2025-11-06 |
| DATA_RELIABILITY_ANALYSIS.md | 12 | 数据可靠性分析 | 2025-11-06 |

---

## 🔗 相关资源

### 代码文件

- **环境代码**: `../env/`
  - `datacenter_env.py` - 数据中心环境
  - `thermal_model.py` - 热力学模型
  - `expert_controller.py` - 专家控制器
  - `real_data_loader.py` - 真实数据加载器

- **训练脚本**: `../scripts/`
  - `preprocess_real_data.py` - 数据预处理
  - `calibrate_model.py` - 模型校准
  - `sensitivity_analysis.py` - 敏感性分析
  - `generate_data.py` - 数据生成

- **主程序**:
  - `../main_datacenter.py` - 数据中心训练主程序
  - `../main.py` - 原始DROPT训练程序

### 工作流脚本

- `../scripts/example_real_data_workflow.sh` - Linux/Mac完整工作流
- `../scripts/example_real_data_workflow.bat` - Windows完整工作流
- `../scripts/quick_start.sh` - 快速开始脚本（Linux/Mac）
- `../scripts/quick_start.bat` - 快速开始脚本（Windows）

---

## 💡 使用建议

### 新手用户
1. 从 [GET_STARTED.md](GET_STARTED.md) 开始
2. 运行快速开始脚本体验
3. 阅读 [README_DATACENTER.md](README_DATACENTER.md) 了解详细功能

### 进阶用户
1. 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 理解架构
2. 参考 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) 了解设计思路
3. 根据需求调整超参数和模型结构

### 研究人员
1. 阅读 [DATA_RELIABILITY_ANALYSIS.md](DATA_RELIABILITY_ANALYSIS.md)
2. 使用敏感性分析工具评估模型
3. 参考真实数据集成方案进行实验

---

## 📞 获取帮助

如果您在使用过程中遇到问题：

1. **查看文档**: 先查看相关文档的常见问题部分
2. **运行测试**: 使用 `scripts/test_datacenter_env.py` 测试环境
3. **查看日志**: 检查 `log/` 目录下的训练日志
4. **提交Issue**: 在GitHub上提交问题（附上错误信息和环境配置）

---

## 📝 文档维护

- **最后更新**: 2025-11-06
- **版本**: 1.0
- **维护者**: DROPT项目团队

如发现文档错误或有改进建议，欢迎提交PR或Issue。

---

**返回**: [项目根目录](../README.md)

