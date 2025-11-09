"""
建筑环境训练诊断脚本

用途：
1. 检查环境是否正常工作
2. 测试策略初始化
3. 验证训练循环的第一步
4. 诊断奖励函数问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from env.building_env_wrapper import make_building_env
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
from policy import DiffusionOPT

def test_environment():
    """测试环境基本功能"""
    print("=" * 60)
    print("测试 1: 环境基本功能")
    print("=" * 60)
    
    try:
        # 创建环境
        env = make_building_env(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson',
            time_resolution=3600,
            max_power=8000,
            target_temp=22.0,
            temp_tolerance=2.0,
            temp_weight=0.999,
            energy_weight=0.001,
            episode_length=None,
            add_violation_penalty=False,
            violation_penalty=100.0,
            expert_type=None,
            training_num=1,
            test_num=1
        )
        
        print(f"✓ 环境创建成功")
        print(f"  状态维度: {env.state_dim}")
        print(f"  动作维度: {env.action_dim}")
        print(f"  房间数量: {env.roomnum}")
        
        # 测试reset
        state, info = env.reset()
        print(f"\n✓ Reset成功")
        print(f"  状态形状: {state.shape}")
        print(f"  状态范围: [{state.min():.2f}, {state.max():.2f}]")
        print(f"  Info字典: {info}")
        
        # 测试step
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(f"\n✓ Step成功")
        print(f"  动作: {action}")
        print(f"  奖励: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")
        
        # 运行一个完整episode
        print(f"\n运行完整episode...")
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        rewards = []
        
        for _ in range(100):  # 最多100步
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            steps += 1
            
            if done or truncated:
                break
        
        print(f"✓ Episode完成")
        print(f"  总步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  平均奖励: {total_reward/steps:.2f}")
        print(f"  奖励范围: [{min(rewards):.2f}, {max(rewards):.2f}]")
        
        return True, env.state_dim, env.action_dim
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_policy_initialization(state_dim, action_dim):
    """测试策略初始化"""
    print("\n" + "=" * 60)
    print("测试 2: 策略初始化")
    print("=" * 60)
    
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 创建网络
        hidden_dim = 256
        max_action = 1.0
        
        actor = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            hidden_dim=hidden_dim
        ).to(device)
        
        critic = DoubleCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            hidden_dim=hidden_dim
        ).to(device)
        
        print(f"✓ 网络创建成功")
        print(f"  Actor参数量: {sum(p.numel() for p in actor.parameters()):,}")
        print(f"  Critic参数量: {sum(p.numel() for p in critic.parameters()):,}")
        
        # 创建优化器
        actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
        
        # 创建扩散模型
        diffusion = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=actor,
            max_action=max_action,
            beta_schedule='vp',
            n_timesteps=5,
        ).to(device)
        
        print(f"✓ 扩散模型创建成功")
        
        # 创建策略 - 不使用bc_coef
        from gym.spaces import Box
        action_space = Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        
        policy = DiffusionOPT(
            state_dim=state_dim,
            actor=diffusion,
            actor_optim=actor_optim,
            action_dim=action_dim,
            critic=critic,
            critic_optim=critic_optim,
            device=device,
            tau=0.005,
            gamma=0.99,
            exploration_noise=0.1,
            bc_coef=False,  # 明确设置为False
            action_space=action_space,
            estimation_step=3,
            lr_decay=False,
            lr_maxt=1000,
        )
        
        print(f"✓ 策略创建成功")
        print(f"  bc_coef: {policy._bc_coef}")
        print(f"  设备: {policy._device}")
        
        # 测试前向传播
        test_state = torch.randn(1, state_dim).to(device)
        with torch.no_grad():
            action = policy.actor(test_state)
        
        print(f"\n✓ 前向传播测试成功")
        print(f"  输入状态形状: {test_state.shape}")
        print(f"  输出动作形状: {action.shape}")
        print(f"  动作范围: [{action.min().item():.2f}, {action.max().item():.2f}]")
        
        return True, policy
        
    except Exception as e:
        print(f"✗ 策略初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def analyze_reward_function():
    """分析奖励函数"""
    print("\n" + "=" * 60)
    print("测试 3: 奖励函数分析")
    print("=" * 60)
    
    try:
        # 创建环境
        env = make_building_env(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson',
            time_resolution=3600,
            max_power=8000,
            target_temp=22.0,
            temp_tolerance=2.0,
            temp_weight=0.999,
            energy_weight=0.001,
            episode_length=None,
            add_violation_penalty=False,
            violation_penalty=100.0,
            expert_type=None,
            training_num=1,
            test_num=1
        )
        
        # 收集多个episode的奖励
        num_episodes = 5
        all_rewards = []
        episode_rewards = []
        
        print(f"运行 {num_episodes} 个episodes...")
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            ep_reward = 0
            ep_rewards = []
            
            for step in range(100):
                action = env.action_space.sample()
                next_state, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                ep_rewards.append(reward)
                all_rewards.append(reward)
                
                if done or truncated:
                    break
            
            episode_rewards.append(ep_reward)
            print(f"  Episode {ep+1}: 总奖励={ep_reward:.2f}, 步数={len(ep_rewards)}, 平均={ep_reward/len(ep_rewards):.2f}")
        
        # 统计分析
        all_rewards = np.array(all_rewards)
        episode_rewards = np.array(episode_rewards)
        
        print(f"\n奖励统计:")
        print(f"  单步奖励:")
        print(f"    平均值: {all_rewards.mean():.2f}")
        print(f"    标准差: {all_rewards.std():.2f}")
        print(f"    最小值: {all_rewards.min():.2f}")
        print(f"    最大值: {all_rewards.max():.2f}")
        print(f"    中位数: {np.median(all_rewards):.2f}")
        
        print(f"\n  Episode总奖励:")
        print(f"    平均值: {episode_rewards.mean():.2f}")
        print(f"    标准差: {episode_rewards.std():.2f}")
        print(f"    最小值: {episode_rewards.min():.2f}")
        print(f"    最大值: {episode_rewards.max():.2f}")
        
        # 诊断
        print(f"\n诊断结果:")
        if episode_rewards.mean() < -10000:
            print(f"  ⚠️  警告: Episode平均奖励过低 ({episode_rewards.mean():.2f})")
            print(f"      建议: 奖励函数可能需要重新设计")
            print(f"      - 减少惩罚权重")
            print(f"      - 添加正向奖励")
            print(f"      - 归一化奖励尺度")
        elif episode_rewards.mean() < 0:
            print(f"  ⚠️  注意: Episode平均奖励为负 ({episode_rewards.mean():.2f})")
            print(f"      建议: 考虑添加更多正向激励")
        else:
            print(f"  ✓ Episode平均奖励正常 ({episode_rewards.mean():.2f})")
        
        if all_rewards.std() > 10000:
            print(f"  ⚠️  警告: 奖励方差过大 ({all_rewards.std():.2f})")
            print(f"      建议: 考虑奖励归一化或裁剪")
        
        return True
        
    except Exception as e:
        print(f"✗ 奖励函数分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  建筑环境训练诊断")
    print("=" * 60 + "\n")
    
    # 测试1: 环境
    success, state_dim, action_dim = test_environment()
    if not success:
        print("\n环境测试失败,停止诊断")
        return
    
    # 测试2: 策略
    success, policy = test_policy_initialization(state_dim, action_dim)
    if not success:
        print("\n策略初始化失败,停止诊断")
        return
    
    # 测试3: 奖励函数
    analyze_reward_function()
    
    print("\n" + "=" * 60)
    print("  诊断完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

