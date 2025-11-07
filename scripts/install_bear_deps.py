#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
安装 BEAR 集成所需的依赖

该脚本会检查并安装以下依赖：
- pvlib: 太阳能数据处理
- scikit-learn: 机器学习工具
- cvxpy: 凸优化（MPC 控制器需要）
- gymnasium: 强化学习环境接口
"""

import subprocess
import sys


def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """安装包"""
    print(f"正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package_name} 安装失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("  BEAR 集成依赖安装")
    print("=" * 60)
    
    # 需要安装的包
    packages = [
        ('pvlib', 'pvlib'),
        ('scikit-learn', 'sklearn'),
        ('cvxpy', 'cvxpy'),
        ('gymnasium', 'gymnasium'),
    ]
    
    # 检查已安装的包
    print("\n检查依赖...")
    to_install = []
    for pip_name, import_name in packages:
        if check_package(import_name):
            print(f"✓ {pip_name} 已安装")
        else:
            print(f"✗ {pip_name} 未安装")
            to_install.append(pip_name)
    
    # 安装缺失的包
    if to_install:
        print(f"\n需要安装 {len(to_install)} 个包:")
        for package in to_install:
            print(f"  - {package}")
        
        response = input("\n是否继续安装? (y/n): ")
        if response.lower() != 'y':
            print("安装已取消")
            return 1
        
        print("\n开始安装...")
        failed = []
        for package in to_install:
            if not install_package(package):
                failed.append(package)
        
        if failed:
            print(f"\n✗ 以下包安装失败:")
            for package in failed:
                print(f"  - {package}")
            print("\n请手动安装:")
            print(f"  pip install {' '.join(failed)}")
            return 1
        else:
            print("\n✓ 所有依赖安装成功!")
    else:
        print("\n✓ 所有依赖已安装!")
    
    print("\n" + "=" * 60)
    print("  依赖检查完成")
    print("=" * 60)
    print("\n现在可以运行测试:")
    print("  python scripts/test_building_env_basic.py")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

