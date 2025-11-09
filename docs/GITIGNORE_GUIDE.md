# .gitignore 配置指南

## 📋 概述

本文档说明 DROPT 项目的 `.gitignore` 配置，帮助你理解哪些文件会被 Git 忽略，哪些会被跟踪。

---

## 🎯 设计原则

### ✅ 应该被跟踪的文件

1. **源代码**: 所有 `.py` 文件
2. **配置文件**: `*.yaml`, `*.json`, `*.toml`, `requirements.txt`
3. **文档**: `*.md` 文件和 `docs/` 目录
4. **脚本**: `scripts/` 目录下的所有脚本
5. **项目结构**: `__init__.py`, `README.md`, `LICENSE`
6. **示例数据**: 模板和示例文件

### ❌ 应该被忽略的文件

1. **训练输出**: 模型检查点、日志、TensorBoard 文件
2. **生成的数据**: CSV 数据文件、专家轨迹
3. **Python 缓存**: `__pycache__/`, `*.pyc`
4. **IDE 配置**: `.vscode/`, `.idea/`
5. **临时文件**: `*.tmp`, `*.log`, `*.bak`
6. **环境配置**: `venv/`, `.env`

---

## 📂 详细分类

### 1. Python 相关

```gitignore
# 字节码和缓存
__pycache__/
*.py[cod]
*$py.class

# 虚拟环境
venv/
env/
.venv/
.conda/

# 包管理
*.egg-info/
dist/
build/
```

**说明**: 这些是 Python 运行时生成的文件，不应该提交到版本控制。

---

### 2. 深度学习和训练相关

```gitignore
# 模型文件
*.pth
*.pt
*.ckpt
*.pkl

# TensorBoard 日志
events.out.tfevents.*

# 训练日志目录
log/
log_*/
```

**说明**: 
- 模型文件通常很大（几百 MB 到几 GB），不适合放在 Git 中
- 建议使用 Git LFS 或云存储（如 Google Drive、AWS S3）来管理模型
- TensorBoard 日志文件也很大，且可以重新生成

**示例**:
```
log_building/
├── default_OfficeSmall_Hot_Dry_20251108_202639/
│   ├── events.out.tfevents.* ← 被忽略
│   └── policy_best.pth        ← 被忽略
```

---

### 3. 数据文件

```gitignore
# 大型数据集
data/
*.csv
*.h5
*.npy

# 但保留示例和模板
!*template*.csv
!*example*.csv
```

**说明**:
- 大型数据集不应该放在 Git 中
- 使用 `!` 前缀可以排除特定文件（即使它们匹配了忽略规则）
- 示例数据和模板应该被保留

**示例**:
```
data/
├── weather_data.csv          ← 被忽略（生成的）
├── workload_trace.csv        ← 被忽略（生成的）
└── data_format_template.csv  ← 被跟踪（模板）
```

---

### 4. IDE 和编辑器

```gitignore
# VSCode
.vscode/

# PyCharm
.idea/

# Vim
*.swp
*~
```

**说明**: 
- IDE 配置通常是个人偏好，不应该强制给团队
- 如果需要共享某些配置，可以使用 `!.vscode/settings.json` 排除

---

### 5. 操作系统

```gitignore
# macOS
.DS_Store

# Windows
Thumbs.db
Desktop.ini

# Linux
*~
```

**说明**: 这些是操作系统自动生成的文件，对项目没有意义。

---

### 6. 项目特定

```gitignore
# DROPT 训练输出
log/default/
log_building/default_*/
log_datacenter/default_*/

# 专家控制器生成的数据
expert_*.csv
expert_*.npy

# 实验结果
experiment_results/
comparison_results/
```

**说明**: 这些是 DROPT 项目特有的输出文件。

---

## 🔧 使用方法

### 初始化 Git 仓库

```bash
# 1. 初始化 Git 仓库
git init

# 2. 添加 .gitignore
git add .gitignore

# 3. 添加所有应该跟踪的文件
git add .

# 4. 查看状态
git status

# 5. 提交
git commit -m "Initial commit"
```

---

### 检查特定文件是否被忽略

```bash
# 检查单个文件
git check-ignore -v log/default/events.out.tfevents.123

# 输出示例:
# .gitignore:142:events.out.tfevents.*    log/default/events.out.tfevents.123
```

**说明**:
- `-v` 参数显示详细信息，包括匹配的规则和行号
- 如果文件被忽略，命令返回 0；否则返回 1

---

### 验证 .gitignore 配置

```bash
# 运行验证脚本
python scripts/verify_gitignore.py
```

**输出示例**:
```
======================================================================
  .gitignore 配置验证
======================================================================

✓ .gitignore 文件存在

[1/3] 扫描项目文件...
  ✓ 找到 156 个被忽略的文件
  ✓ 找到 89 个被跟踪的文件

[2/3] 分析被忽略的文件...

被忽略的文件类型:
  • 模型文件: 13 个
    - log_building/default_OfficeSmall_Hot_Dry_20251108_204524/policy_best.pth
    - log_building/default_OfficeSmall_Hot_Dry_20251108_204850/policy_best.pth
    ... 还有 11 个
  • 日志文件: 143 个
    - log_building/default_OfficeSmall_Hot_Dry_20251108_202639/events.out.tfevents.*
    ...

[3/3] 分析被跟踪的文件...

被跟踪的文件类型:
  • Python源码: 45 个
    - main_datacenter.py
    - main_building.py
    - env/datacenter_env.py
    ...
  • 文档: 32 个
    - README.md
    - docs/GET_STARTED.md
    ...
```

---

## 📊 文件统计

### 典型的 DROPT 项目

| 类型 | 数量 | 状态 | 说明 |
|------|------|------|------|
| Python 源码 | ~50 | ✅ 跟踪 | 核心代码 |
| 文档 | ~30 | ✅ 跟踪 | Markdown 文档 |
| 配置文件 | ~5 | ✅ 跟踪 | YAML/JSON 配置 |
| 模型文件 | ~20 | ❌ 忽略 | .pth 检查点 |
| 日志文件 | ~100+ | ❌ 忽略 | TensorBoard 日志 |
| 数据文件 | ~10 | ❌ 忽略 | CSV 数据 |
| Python 缓存 | ~50 | ❌ 忽略 | __pycache__ |

---

## 🚨 常见问题

### Q1: 如何跟踪一个被忽略的文件？

**A**: 使用 `!` 前缀在 `.gitignore` 中排除它：

```gitignore
# 忽略所有 .csv 文件
*.csv

# 但保留这个特定文件
!important_data.csv
```

---

### Q2: 如何忽略整个目录但保留目录结构？

**A**: 使用 `.gitkeep` 文件：

```bash
# 1. 在 .gitignore 中忽略目录内容
log/
!log/.gitkeep

# 2. 创建 .gitkeep 文件
touch log/.gitkeep

# 3. 添加到 Git
git add log/.gitkeep
```

---

### Q3: 已经提交的文件如何从 Git 中移除但保留本地？

**A**: 使用 `git rm --cached`：

```bash
# 移除单个文件
git rm --cached log/default/policy_best.pth

# 移除整个目录
git rm -r --cached log/

# 提交更改
git commit -m "Remove log files from Git"
```

---

### Q4: 如何查看所有被忽略的文件？

**A**: 使用 `git status --ignored`：

```bash
git status --ignored

# 或者只显示被忽略的文件
git status --ignored --short | grep '^!!'
```

---

### Q5: 模型文件太大，如何管理？

**A**: 有几种方案：

1. **Git LFS** (Large File Storage):
   ```bash
   # 安装 Git LFS
   git lfs install
   
   # 跟踪大文件
   git lfs track "*.pth"
   
   # 添加和提交
   git add .gitattributes
   git add model.pth
   git commit -m "Add model with LFS"
   ```

2. **云存储**:
   - Google Drive
   - AWS S3
   - Azure Blob Storage
   - 在 README 中提供下载链接

3. **模型仓库**:
   - Hugging Face Model Hub
   - PyTorch Hub
   - TensorFlow Hub

---

## 📝 最佳实践

### 1. 定期检查 Git 状态

```bash
# 查看当前状态
git status

# 查看被忽略的文件
git status --ignored
```

---

### 2. 提交前验证

```bash
# 查看将要提交的文件
git diff --cached --name-only

# 确保没有大文件
git diff --cached --stat
```

---

### 3. 使用 .gitignore 模板

GitHub 提供了各种语言和框架的 `.gitignore` 模板：
- https://github.com/github/gitignore

---

### 4. 团队协作

- 在项目初期就设置好 `.gitignore`
- 定期更新和维护
- 在 README 中说明哪些文件需要单独获取
- 使用 `requirements.txt` 管理依赖

---

## 🔗 相关资源

- [Git 官方文档 - gitignore](https://git-scm.com/docs/gitignore)
- [GitHub .gitignore 模板](https://github.com/github/gitignore)
- [Git LFS 文档](https://git-lfs.github.com/)
- [验证脚本](../scripts/verify_gitignore.py)

---

## ✅ 检查清单

在提交代码前，确保：

- [ ] `.gitignore` 文件已创建并添加到 Git
- [ ] 所有源代码文件都被跟踪
- [ ] 所有文档文件都被跟踪
- [ ] 模型文件和日志被忽略
- [ ] Python 缓存被忽略
- [ ] IDE 配置被忽略（或只保留必要的）
- [ ] 运行 `git status` 检查没有意外的文件
- [ ] 运行 `scripts/verify_gitignore.py` 验证配置

---

**最后更新**: 2025-11-09  
**维护者**: DROPT Team

