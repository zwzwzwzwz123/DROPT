#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 .gitignore 配置

检查哪些文件会被 Git 忽略，哪些会被跟踪
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_ignore(file_path):
    """
    检查文件是否被 .gitignore 忽略
    
    参数:
        file_path: 文件路径
    
    返回:
        True: 被忽略
        False: 不被忽略
    """
    try:
        result = subprocess.run(
            ['git', 'check-ignore', file_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"警告: 无法检查 {file_path}: {e}")
        return None

def scan_directory(root_dir='.', max_depth=3):
    """
    扫描目录并分类文件
    
    参数:
        root_dir: 根目录
        max_depth: 最大扫描深度
    
    返回:
        (ignored_files, tracked_files): 被忽略和被跟踪的文件列表
    """
    ignored_files = []
    tracked_files = []
    
    root_path = Path(root_dir)
    
    # 遍历目录
    for item in root_path.rglob('*'):
        # 跳过 .git 目录
        if '.git' in item.parts:
            continue
        
        # 检查深度
        depth = len(item.relative_to(root_path).parts)
        if depth > max_depth:
            continue
        
        # 只检查文件
        if item.is_file():
            rel_path = str(item.relative_to(root_path))
            is_ignored = check_git_ignore(rel_path)
            
            if is_ignored is True:
                ignored_files.append(rel_path)
            elif is_ignored is False:
                tracked_files.append(rel_path)
    
    return ignored_files, tracked_files

def categorize_files(files):
    """
    按类型分类文件
    
    参数:
        files: 文件列表
    
    返回:
        dict: 分类后的文件字典
    """
    categories = {
        'Python源码': [],
        '模型文件': [],
        '日志文件': [],
        '数据文件': [],
        '文档': [],
        '配置': [],
        '其他': []
    }
    
    for file in files:
        if file.endswith('.py'):
            categories['Python源码'].append(file)
        elif file.endswith(('.pth', '.pt', '.ckpt', '.pkl')):
            categories['模型文件'].append(file)
        elif 'log' in file.lower() or file.endswith('.log') or 'tfevents' in file:
            categories['日志文件'].append(file)
        elif file.endswith(('.csv', '.dat', '.h5', '.npy')):
            categories['数据文件'].append(file)
        elif file.endswith(('.md', '.rst', '.txt')):
            categories['文档'].append(file)
        elif file.endswith(('.yaml', '.yml', '.json', '.toml', '.ini', '.cfg')):
            categories['配置'].append(file)
        else:
            categories['其他'].append(file)
    
    return categories

def main():
    """主函数"""
    print("=" * 70)
    print("  .gitignore 配置验证")
    print("=" * 70)
    
    # 检查是否在 Git 仓库中
    if not os.path.exists('.git'):
        print("\n⚠️  警告: 当前目录不是 Git 仓库")
        print("   请先运行: git init")
        print("   然后运行: git add .gitignore")
        return
    
    # 检查 .gitignore 是否存在
    if not os.path.exists('.gitignore'):
        print("\n✗ .gitignore 文件不存在")
        return
    
    print("\n✓ .gitignore 文件存在")
    
    # 扫描目录
    print("\n[1/3] 扫描项目文件...")
    ignored_files, tracked_files = scan_directory('.', max_depth=3)
    
    print(f"  ✓ 找到 {len(ignored_files)} 个被忽略的文件")
    print(f"  ✓ 找到 {len(tracked_files)} 个被跟踪的文件")
    
    # 分类被忽略的文件
    print("\n[2/3] 分析被忽略的文件...")
    ignored_categories = categorize_files(ignored_files)
    
    print("\n被忽略的文件类型:")
    for category, files in ignored_categories.items():
        if files:
            print(f"  • {category}: {len(files)} 个")
            # 显示前3个示例
            for file in files[:3]:
                print(f"    - {file}")
            if len(files) > 3:
                print(f"    ... 还有 {len(files) - 3} 个")
    
    # 分类被跟踪的文件
    print("\n[3/3] 分析被跟踪的文件...")
    tracked_categories = categorize_files(tracked_files)
    
    print("\n被跟踪的文件类型:")
    for category, files in tracked_categories.items():
        if files:
            print(f"  • {category}: {len(files)} 个")
            # 显示前5个示例
            for file in files[:5]:
                print(f"    - {file}")
            if len(files) > 5:
                print(f"    ... 还有 {len(files) - 5} 个")
    
    # 检查关键文件
    print("\n" + "=" * 70)
    print("  关键文件检查")
    print("=" * 70)
    
    critical_files = {
        '应该被跟踪': [
            'main_datacenter.py',
            'main_building.py',
            'README.md',
            'requirements.txt',
            '.gitignore',
            'env/datacenter_env.py',
            'policy/diffusion_opt.py',
            'diffusion/diffusion.py',
        ],
        '应该被忽略': [
            'log/default/events.out.tfevents.123',
            'log_building/default_OfficeSmall_Hot_Dry_20251108_202639/policy_best.pth',
            '__pycache__/test.pyc',
            'data/weather_data.csv',
        ]
    }
    
    print("\n✓ 应该被跟踪的文件:")
    for file in critical_files['应该被跟踪']:
        if os.path.exists(file):
            is_ignored = check_git_ignore(file)
            status = "✗ 被忽略" if is_ignored else "✓ 被跟踪"
            print(f"  {status}: {file}")
        else:
            print(f"  - 不存在: {file}")
    
    print("\n✓ 应该被忽略的文件:")
    for file in critical_files['应该被忽略']:
        if os.path.exists(file):
            is_ignored = check_git_ignore(file)
            status = "✓ 被忽略" if is_ignored else "✗ 被跟踪"
            print(f"  {status}: {file}")
        else:
            print(f"  - 不存在: {file}")
    
    # 总结
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"\n总文件数: {len(ignored_files) + len(tracked_files)}")
    print(f"  • 被忽略: {len(ignored_files)} ({len(ignored_files)*100//(len(ignored_files)+len(tracked_files))}%)")
    print(f"  • 被跟踪: {len(tracked_files)} ({len(tracked_files)*100//(len(ignored_files)+len(tracked_files))}%)")
    
    print("\n建议:")
    print("  1. 运行 'git status' 查看当前状态")
    print("  2. 运行 'git add .' 添加所有应该跟踪的文件")
    print("  3. 运行 'git commit -m \"Initial commit\"' 提交")
    print("  4. 使用 'git check-ignore -v <file>' 检查特定文件")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

