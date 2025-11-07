#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 BEAR 环境"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("正在导入模块...")
from env.building_env_wrapper import BearEnvWrapper

print("✓ 模块导入成功")

print("\n正在创建环境...")
env = BearEnvWrapper(
    building_type='OfficeSmall',
    weather_type='Hot_Dry',
    location='Tucson'
)
print("✓ 环境创建成功")
print(f"  房间数量: {env.roomnum}")
print(f"  状态维度: {env.state_dim}")
print(f"  动作维度: {env.action_dim}")

print("\n正在重置环境...")
state, info = env.reset()
print("✓ 重置成功")
print(f"  状态形状: {state.shape}")

print("\n正在执行 3 步...")
for step in range(3):
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print(f"  步数 {step+1}: 奖励={reward:.2f}")
    state = next_state

print("\n✓ 所有测试通过！")

