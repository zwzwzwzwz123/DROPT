#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试第三阶段：训练脚本功能
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 60)
print("  Phase 3: Training Script Test")
print("=" * 60)

# 测试 1: 导入模块
print("\n[1/5] Testing imports...")
try:
    from env.building_env_wrapper import make_building_env
    from diffusion.model import MLP, DoubleCritic
    from diffusion import Diffusion
    from policy import DiffusionOPT
    from tianshou.data import Collector, VectorReplayBuffer
    import torch
    print("OK - All imports successful")
except Exception as e:
    print(f"FAIL - Import error: {e}")
    sys.exit(1)

# 测试 2: 创建环境
print("\n[2/5] Testing environment creation...")
try:
    env, train_envs, test_envs = make_building_env(
        building_type='OfficeSmall',
        training_num=2,
        test_num=1
    )
    print(f"OK - Environment created")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Room num: {env.roomnum}")
except Exception as e:
    print(f"FAIL - Environment creation error: {e}")
    sys.exit(1)

# 测试 3: 创建网络
print("\n[3/5] Testing network creation...")
try:
    device = torch.device('cpu')
    
    actor = MLP(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
        t_dim=16
    ).to(device)
    
    critic = DoubleCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=256
    ).to(device)
    
    print(f"OK - Networks created")
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in critic.parameters()):,}")
except Exception as e:
    print(f"FAIL - Network creation error: {e}")
    sys.exit(1)

# 测试 4: 创建策略
print("\n[4/5] Testing policy creation...")
try:
    diffusion = Diffusion(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        model=actor,
        max_action=1.0,
        beta_schedule='vp',
        n_timesteps=5,
    )
    
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
    
    policy = DiffusionOPT(
        state_dim=env.state_dim,
        actor=diffusion,
        actor_optim=actor_optim,
        action_dim=env.action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=device,
        tau=0.005,
        gamma=0.99,
        exploration_noise=0.1,
        bc_coef=False,
        action_space=env.action_space,
    )
    
    print("OK - Policy created")
except Exception as e:
    print(f"FAIL - Policy creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 创建收集器
print("\n[5/5] Testing collector creation...")
try:
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(10000, len(train_envs)),
        exploration_noise=True
    )
    
    test_collector = Collector(policy, test_envs)
    
    print("OK - Collectors created")
except Exception as e:
    print(f"FAIL - Collector creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 所有测试通过
print("\n" + "=" * 60)
print("  ALL TESTS PASSED!")
print("=" * 60)
print("\nYou can now run training with:")
print("  python main_building.py --building-type OfficeSmall --epoch 1000")
print()

