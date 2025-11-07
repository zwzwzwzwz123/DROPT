#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 main_building.py 的基本功能

测试内容：
1. 参数解析
2. 环境创建
3. 网络创建
4. 策略创建
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from env.building_env_wrapper import make_building_env
from diffusion.model import MLP, DoubleCritic
from diffusion import Diffusion
from policy import DiffusionOPT
from tianshou.data import Batch


def test_env_creation():
    """测试环境创建"""
    print("=" * 60)
    print("  测试 1: 环境创建")
    print("=" * 60)
    
    try:
        env, train_envs, test_envs = make_building_env(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson',
            training_num=2,
            test_num=1
        )
        
        print(f"✓ 环境创建成功")
        print(f"  建筑类型: OfficeSmall")
        print(f"  房间数量: {env.roomnum}")
        print(f"  状态维度: {env.state_dim}")
        print(f"  动作维度: {env.action_dim}")
        print(f"  训练环境数: {train_envs.env_num}")
        print(f"  测试环境数: {test_envs.env_num}")
        
        return True, env
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_network_creation(env):
    """测试网络创建"""
    print("\n" + "=" * 60)
    print("  测试 2: 网络创建")
    print("=" * 60)
    
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dim = env.state_dim
        action_dim = env.action_dim
        
        # 创建 Actor
        actor = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            t_dim=16
        ).to(device)
        
        print(f"✓ Actor 创建成功")
        print(f"  参数量: {sum(p.numel() for p in actor.parameters()):,}")
        
        # 创建 Critic
        critic = DoubleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256
        ).to(device)
        
        print(f"✓ Critic 创建成功")
        print(f"  参数量: {sum(p.numel() for p in critic.parameters()):,}")
        
        return True, actor, critic, device
    except Exception as e:
        print(f"✗ 网络创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_diffusion_creation(env, actor, device):
    """测试扩散模型创建"""
    print("\n" + "=" * 60)
    print("  测试 3: 扩散模型创建")
    print("=" * 60)
    
    try:
        diffusion = Diffusion(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            model=actor,
            max_action=1.0,
            beta_schedule='vp',
            n_timesteps=5,
        )
        
        print(f"✓ 扩散模型创建成功")
        print(f"  扩散步数: 5")
        print(f"  噪声调度: vp")
        
        return True, diffusion
    except Exception as e:
        print(f"✗ 扩散模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_policy_creation(env, diffusion, actor, critic, device):
    """测试策略创建"""
    print("\n" + "=" * 60)
    print("  测试 4: 策略创建")
    print("=" * 60)

    try:
        actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

        policy = DiffusionOPT(
            state_dim=env.state_dim,
            actor=diffusion,  # 使用扩散模型作为 actor
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
        
        print(f"✓ 策略创建成功")
        print(f"  算法: DiffusionOPT")
        print(f"  折扣因子: 0.99")
        print(f"  探索噪声: 0.1")
        
        return True, policy
    except Exception as e:
        print(f"✗ 策略创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(env, policy):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("  测试 5: 前向传播")
    print("=" * 60)

    try:
        # 重置环境
        state, _ = env.reset()

        print(f"  状态形状: {state.shape}")
        print(f"  状态维度: {len(state)}")
        print(f"  环境状态维度: {env.state_dim}")

        # 创建 Batch 对象
        batch = Batch(obs=state[np.newaxis, :])

        # 前向传播
        with torch.no_grad():
            result = policy(batch)
            action = result.act.cpu().numpy()[0]

        print(f"✓ 前向传播成功")
        print(f"  输入状态形状: {state.shape}")
        print(f"  输出动作形状: {action.shape}")
        print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"✓ 环境交互成功")
        print(f"  奖励: {reward:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  main_building.py 功能测试")
    print("=" * 60)
    
    # 测试 1: 环境创建
    success, env = test_env_creation()
    if not success:
        print("\n✗ 测试失败：环境创建")
        return 1
    
    # 测试 2: 网络创建
    success, actor, critic, device = test_network_creation(env)
    if not success:
        print("\n✗ 测试失败：网络创建")
        return 1
    
    # 测试 3: 扩散模型创建
    success, diffusion = test_diffusion_creation(env, actor, device)
    if not success:
        print("\n✗ 测试失败：扩散模型创建")
        return 1
    
    # 测试 4: 策略创建
    success, policy = test_policy_creation(env, diffusion, actor, critic, device)
    if not success:
        print("\n✗ 测试失败：策略创建")
        return 1
    
    # 测试 5: 前向传播
    success = test_forward_pass(env, policy)
    if not success:
        print("\n✗ 测试失败：前向传播")
        return 1
    
    # 所有测试通过
    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60)
    print("✓ 所有测试通过！")
    print("\n可以开始训练:")
    print("  python main_building.py --building-type OfficeSmall --epoch 1000")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

