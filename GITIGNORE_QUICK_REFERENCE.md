# .gitignore 快速参考

## 🚀 快速开始

### 1. 初始化 Git 仓库

```bash
# 初始化 Git
git init

# 添加 .gitignore
git add .gitignore

# 添加所有文件
git add .

# 查看状态
git status

# 提交
git commit -m "Initial commit with .gitignore"
```

---

## 📋 被忽略的文件类型

### ❌ 训练输出（最重要）

```
log/                              # 所有训练日志
log_building/                     # 建筑环境训练日志
log_datacenter/                   # 数据中心训练日志
*.pth                             # PyTorch 模型
*.pt                              # PyTorch 模型
*.ckpt                            # 检查点
events.out.tfevents.*             # TensorBoard 日志
```

**原因**: 这些文件很大（几百 MB 到几 GB），不适合放在 Git 中。

---

### ❌ Python 缓存

```
__pycache__/                      # Python 缓存目录
*.pyc                             # 字节码文件
*.pyo                             # 优化的字节码
*.egg-info/                       # 包信息
```

**原因**: 这些是自动生成的，每次运行都会重新创建。

---

### ❌ 数据文件

```
data/                             # 数据目录
*.csv                             # CSV 数据
*.h5                              # HDF5 数据
*.npy                             # NumPy 数组
weather_data.csv                  # 生成的天气数据
workload_trace.csv                # 生成的负载轨迹
```

**原因**: 数据文件通常很大，且可以重新生成。

**例外**: 模板和示例文件会被保留（如 `*template*.csv`）。

---

### ❌ IDE 配置

```
.vscode/                          # VSCode 配置
.idea/                            # PyCharm 配置
*.swp                             # Vim 临时文件
```

**原因**: IDE 配置是个人偏好，不应该强制给团队。

---

### ❌ 环境和临时文件

```
venv/                             # 虚拟环境
env/                              # 虚拟环境
.env                              # 环境变量
*.log                             # 日志文件
*.tmp                             # 临时文件
```

---

## ✅ 被跟踪的文件类型

### ✅ 源代码

```
*.py                              # 所有 Python 文件
main_datacenter.py                # 训练脚本
main_building.py                  # 训练脚本
env/*.py                          # 环境代码
policy/*.py                       # 策略代码
diffusion/*.py                    # 扩散模型代码
```

---

### ✅ 配置文件

```
requirements.txt                  # Python 依赖
*.yaml                            # YAML 配置
*.json                            # JSON 配置
*.toml                            # TOML 配置
```

---

### ✅ 文档

```
README.md                         # 项目说明
docs/*.md                         # 所有文档
*.txt                             # 文本文件
```

---

### ✅ 脚本和工具

```
scripts/*.py                      # 所有脚本
scripts/*.sh                      # Shell 脚本
scripts/*.bat                     # Windows 批处理
```

---

## 🔧 常用命令

### 检查文件是否被忽略

```bash
# 检查单个文件
git check-ignore -v log/default/policy_best.pth

# 输出示例:
# .gitignore:142:*.pth    log/default/policy_best.pth
```

---

### 查看所有被忽略的文件

```bash
# 查看被忽略的文件
git status --ignored

# 只显示被忽略的文件（简洁）
git status --ignored --short | grep '^!!'
```

---

### 移除已提交的文件（但保留本地）

```bash
# 移除单个文件
git rm --cached log/policy_best.pth

# 移除整个目录
git rm -r --cached log/

# 提交更改
git commit -m "Remove log files from Git"
```

---

### 强制添加被忽略的文件

```bash
# 使用 -f 强制添加
git add -f important_model.pth

# 或者在 .gitignore 中排除
# 在 .gitignore 中添加: !important_model.pth
```

---

## 🚨 常见问题

### Q: 为什么我的 Python 文件没有被跟踪？

**A**: 检查文件名是否正确，确保没有拼写错误。运行：
```bash
git check-ignore -v your_file.py
```

---

### Q: 如何保留空目录？

**A**: 在目录中创建 `.gitkeep` 文件：
```bash
touch log/.gitkeep
git add log/.gitkeep
```

---

### Q: 模型文件太大怎么办？

**A**: 有几种方案：
1. **使用 Git LFS**: `git lfs track "*.pth"`
2. **使用云存储**: Google Drive, AWS S3
3. **使用模型仓库**: Hugging Face, PyTorch Hub

---

### Q: 如何验证 .gitignore 配置？

**A**: 运行验证脚本：
```bash
python scripts/verify_gitignore.py
```

---

## 📊 项目文件统计

### 典型的 DROPT 项目

| 类型 | 数量 | 大小 | 状态 |
|------|------|------|------|
| Python 源码 | ~50 | ~500 KB | ✅ 跟踪 |
| 文档 | ~30 | ~2 MB | ✅ 跟踪 |
| 配置文件 | ~5 | ~50 KB | ✅ 跟踪 |
| **总计（跟踪）** | **~85** | **~2.5 MB** | **✅** |
| 模型文件 | ~20 | ~2 GB | ❌ 忽略 |
| 日志文件 | ~100+ | ~500 MB | ❌ 忽略 |
| 数据文件 | ~10 | ~100 MB | ❌ 忽略 |
| Python 缓存 | ~50 | ~50 MB | ❌ 忽略 |
| **总计（忽略）** | **~180** | **~2.6 GB** | **❌** |

**结论**: `.gitignore` 帮助你只跟踪 ~2.5 MB 的核心代码，而忽略 ~2.6 GB 的生成文件。

---

## ✅ 提交前检查清单

在提交代码前，确保：

- [ ] 运行 `git status` 检查状态
- [ ] 确认所有 `.py` 文件都被跟踪
- [ ] 确认所有 `.md` 文档都被跟踪
- [ ] 确认 `requirements.txt` 被跟踪
- [ ] 确认没有 `.pth` 模型文件被跟踪
- [ ] 确认没有 `log/` 目录内容被跟踪
- [ ] 确认没有 `__pycache__/` 被跟踪
- [ ] 运行 `python scripts/verify_gitignore.py` 验证

---

## 📚 更多信息

- **详细指南**: [docs/GITIGNORE_GUIDE.md](docs/GITIGNORE_GUIDE.md)
- **验证脚本**: [scripts/verify_gitignore.py](scripts/verify_gitignore.py)
- **Git 官方文档**: https://git-scm.com/docs/gitignore

---

## 🎯 一键设置

```bash
# 1. 初始化 Git（如果还没有）
git init

# 2. 添加所有文件
git add .

# 3. 查看状态（确认正确）
git status

# 4. 提交
git commit -m "Initial commit: DROPT project with proper .gitignore"

# 5. 验证配置
python scripts/verify_gitignore.py
```

---

**提示**: 如果你已经提交了不应该跟踪的文件，使用 `git rm --cached` 移除它们，然后重新提交。

---

**最后更新**: 2025-11-09  
**版本**: v1.0

