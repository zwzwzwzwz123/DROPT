#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEAR 建筑环境基础功能测试脚本

测试内容：
1. 环境创建
2. 状态空间和动作空间
3. reset() 方法
4. step() 方法
5. 多步运行
6. 向量化环境
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from env.building_env_wrapper import BearEnvWrapper, make_building_env


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_env_creation():
    """测试 1: 环境创建"""
    print_separator("测试 1: 环境创建")
    
    try:
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson'
        )
        print("✓ 环境创建成功")
        print(f"  建筑类型: {env.building_type}")
        print(f"  气候类型: {env.weather_type}")
        print(f"  地理位置: {env.location}")
        print(f"  房间数量: {env.roomnum}")
        print(f"  状态维度: {env.state_dim}")
        print(f"  动作维度: {env.action_dim}")
        return True
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spaces():
    """测试 2: 状态空间和动作空间"""
    print_separator("测试 2: 状态空间和动作空间")
    
    try:
        env = BearEnvWrapper()
        
        # 测试状态空间
        obs_space = env.observation_space
        print("✓ 状态空间:")
        print(f"  类型: {type(obs_space)}")
        print(f"  形状: {obs_space.shape}")
        print(f"  最小值: {obs_space.low[:5]}... (前5个)")
        print(f"  最大值: {obs_space.high[:5]}... (前5个)")
        
        # 测试动作空间
        action_space = env.action_space
        print("\n✓ 动作空间:")
        print(f"  类型: {type(action_space)}")
        print(f"  形状: {action_space.shape}")
        print(f"  最小值: {action_space.low}")
        print(f"  最大值: {action_space.high}")
        
        # 测试采样
        sample_action = action_space.sample()
        print(f"\n✓ 随机动作采样: {sample_action}")
        
        return True
    except Exception as e:
        print(f"✗ 状态/动作空间测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset():
    """测试 3: reset() 方法"""
    print_separator("测试 3: reset() 方法")
    
    try:
        env = BearEnvWrapper()
        state, info = env.reset()
        
        print("✓ 重置成功")
        print(f"  状态形状: {state.shape}")
        print(f"  状态类型: {state.dtype}")
        print(f"  状态范围: [{state.min():.2f}, {state.max():.2f}]")
        print(f"  状态前5个值: {state[:5]}")
        print(f"\n  信息字典键: {list(info.keys())}")
        print(f"  建筑类型: {info.get('building_type')}")
        print(f"  房间数量: {info.get('roomnum')}")
        
        return True
    except Exception as e:
        print(f"✗ reset() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step():
    """测试 4: step() 方法"""
    print_separator("测试 4: step() 方法")
    
    try:
        env = BearEnvWrapper()
        state, _ = env.reset()
        
        # 执行一步
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        print("✓ step() 执行成功")
        print(f"  动作: {action}")
        print(f"  下一状态形状: {next_state.shape}")
        print(f"  奖励: {reward:.4f}")
        print(f"  done: {done}")
        print(f"  truncated: {truncated}")
        print(f"  当前步数: {info.get('current_step')}")
        print(f"  累计奖励: {info.get('total_reward'):.4f}")
        
        return True
    except Exception as e:
        print(f"✗ step() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_steps():
    """测试 5: 多步运行"""
    print_separator("测试 5: 多步运行 (24步)")
    
    try:
        env = BearEnvWrapper()
        state, _ = env.reset()
        
        total_reward = 0.0
        num_steps = 24  # 24小时
        
        print("运行中...")
        for step in range(num_steps):
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if (step + 1) % 6 == 0:  # 每6步打印一次
                print(f"  步数 {step+1:2d}: 奖励={reward:8.2f}, 累计奖励={total_reward:10.2f}")
            
            state = next_state
            
            if done:
                print(f"  环境在第 {step+1} 步结束")
                break
        
        print(f"\n✓ 多步运行成功")
        print(f"  总步数: {step+1}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  平均奖励: {total_reward/(step+1):.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 多步运行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vectorized_env():
    """测试 6: 向量化环境"""
    print_separator("测试 6: 向量化环境")
    
    try:
        env, train_envs, test_envs = make_building_env(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson',
            training_num=2,
            test_num=1
        )
        
        print("✓ 向量化环境创建成功")
        print(f"  单个环境: {type(env)}")
        print(f"  训练环境数量: {train_envs.env_num}")
        print(f"  测试环境数量: {test_envs.env_num}")
        
        # 测试训练环境
        states = train_envs.reset()
        print(f"\n✓ 训练环境重置成功")
        print(f"  状态形状: {states.shape}")
        
        # 执行一步
        actions = np.array([train_envs.action_space.sample() for _ in range(train_envs.env_num)])
        results = train_envs.step(actions)
        next_states, rewards, dones, infos = results
        
        print(f"\n✓ 训练环境 step() 成功")
        print(f"  下一状态形状: {next_states.shape}")
        print(f"  奖励: {rewards}")
        print(f"  done: {dones}")
        
        return True
    except Exception as e:
        print(f"✗ 向量化环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_buildings():
    """测试 7: 不同建筑类型"""
    print_separator("测试 7: 不同建筑类型")
    
    building_types = ['OfficeSmall', 'Hospital', 'SchoolPrimary']
    
    try:
        for building in building_types:
            env = BearEnvWrapper(
                building_type=building,
                weather_type='Hot_Dry',
                location='Tucson'
            )
            state, _ = env.reset()
            
            print(f"\n✓ {building}:")
            print(f"  房间数: {env.roomnum}")
            print(f"  状态维度: {env.state_dim}")
            print(f"  动作维度: {env.action_dim}")
        
        return True
    except Exception as e:
        print(f"✗ 不同建筑类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  BEAR 建筑环境基础功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("环境创建", test_env_creation),
        ("状态/动作空间", test_spaces),
        ("reset() 方法", test_reset),
        ("step() 方法", test_step),
        ("多步运行", test_multi_steps),
        ("向量化环境", test_vectorized_env),
        ("不同建筑类型", test_different_buildings),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print_separator("测试总结")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境基础功能正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

