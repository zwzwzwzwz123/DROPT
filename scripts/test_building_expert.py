#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEAR 建筑环境专家控制器测试脚本

测试内容：
1. MPC 控制器
2. PID 控制器
3. 规则控制器
4. Bang-Bang 控制器
5. 性能对比
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from env.building_env_wrapper import BearEnvWrapper
from env.building_expert_controller import (
    create_expert_controller,
    BearMPCWrapper,
    BearPIDController,
    BearRuleBasedController,
    BearBangBangController
)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_controller_creation():
    """测试 1: 控制器创建"""
    print_separator("测试 1: 控制器创建")
    
    try:
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson'
        )
        
        # 测试 MPC 控制器
        print("\n创建 MPC 控制器...")
        mpc = create_expert_controller('mpc', env)
        print(f"✓ MPC 控制器: {type(mpc).__name__}")
        
        # 测试 PID 控制器
        print("\n创建 PID 控制器...")
        pid = create_expert_controller('pid', env)
        print(f"✓ PID 控制器: {type(pid).__name__}")
        
        # 测试规则控制器
        print("\n创建规则控制器...")
        rule = create_expert_controller('rule', env)
        print(f"✓ 规则控制器: {type(rule).__name__}")
        
        # 测试 Bang-Bang 控制器
        print("\n创建 Bang-Bang 控制器...")
        bangbang = create_expert_controller('bangbang', env)
        print(f"✓ Bang-Bang 控制器: {type(bangbang).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ 控制器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_controller():
    """测试 2: MPC 控制器"""
    print_separator("测试 2: MPC 控制器")
    
    try:
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson'
        )
        
        mpc = BearMPCWrapper(env, planning_steps=1)
        state, _ = env.reset()
        
        print(f"初始状态: 房间温度 = {state[:env.roomnum]}")
        
        # 获取 MPC 动作
        action = mpc.get_action(state)
        print(f"MPC 动作: {action}")
        print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        print(f"奖励: {reward:.2f}")
        print(f"下一状态: 房间温度 = {next_state[:env.roomnum]}")
        
        print("\n✓ MPC 控制器测试通过")
        return True
    except Exception as e:
        print(f"✗ MPC 控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pid_controller():
    """测试 3: PID 控制器"""
    print_separator("测试 3: PID 控制器")
    
    try:
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson'
        )
        
        pid = BearPIDController(env, kp=0.5, ki=0.01, kd=0.1)
        state, _ = env.reset()
        
        print(f"初始状态: 房间温度 = {state[:env.roomnum]}")
        print(f"目标温度: {env.target_temp}°C")
        
        # 运行 5 步
        print("\n运行 5 步:")
        for step in range(5):
            action = pid.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            avg_temp = np.mean(next_state[:env.roomnum])
            print(f"  步数 {step+1}: 平均温度={avg_temp:.2f}°C, "
                  f"平均动作={np.mean(action):.3f}, 奖励={reward:.2f}")
            
            state = next_state
        
        print("\n✓ PID 控制器测试通过")
        return True
    except Exception as e:
        print(f"✗ PID 控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rule_controller():
    """测试 4: 规则控制器"""
    print_separator("测试 4: 规则控制器")
    
    try:
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson'
        )
        
        rule = BearRuleBasedController(env, cooling_power=0.8, heating_power=0.8)
        state, _ = env.reset()
        
        print(f"初始状态: 房间温度 = {state[:env.roomnum]}")
        print(f"目标温度: {env.target_temp}°C ± {env.temp_tolerance}°C")
        
        # 获取动作
        action = rule.get_action(state)
        print(f"\n规则动作: {action}")
        
        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        print(f"奖励: {reward:.2f}")
        
        print("\n✓ 规则控制器测试通过")
        return True
    except Exception as e:
        print(f"✗ 规则控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_expert():
    """测试 5: 集成专家控制器"""
    print_separator("测试 5: 集成专家控制器")
    
    try:
        # 创建带专家控制器的环境
        env = BearEnvWrapper(
            building_type='OfficeSmall',
            weather_type='Hot_Dry',
            location='Tucson',
            expert_type='mpc'  # 使用 MPC 作为专家
        )
        
        state, info = env.reset()
        print(f"环境创建成功，专家类型: {env.expert_type}")
        
        # 执行一步
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        
        # 检查是否有专家动作
        if 'expert_action' in info:
            print(f"\n✓ 专家动作已添加到 info")
            print(f"  专家动作: {info['expert_action']}")
        else:
            print(f"\n✗ 未找到专家动作")
            return False
        
        print("\n✓ 集成专家控制器测试通过")
        return True
    except Exception as e:
        print(f"✗ 集成专家控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controller_performance():
    """测试 6: 控制器性能对比"""
    print_separator("测试 6: 控制器性能对比 (24步)")
    
    controllers = {
        'MPC': 'mpc',
        'PID': 'pid',
        'Rule': 'rule',
        'BangBang': 'bangbang',
    }
    
    results = {}
    
    for name, controller_type in controllers.items():
        try:
            print(f"\n测试 {name} 控制器...")
            
            env = BearEnvWrapper(
                building_type='OfficeSmall',
                weather_type='Hot_Dry',
                location='Tucson',
                expert_type=controller_type
            )
            
            state, _ = env.reset()
            total_reward = 0.0
            temp_errors = []
            
            for step in range(24):
                # 使用专家动作
                if env.expert_controller is not None:
                    action = env.expert_controller.get_action(state)
                else:
                    action = env.action_space.sample()
                
                next_state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # 记录温度误差
                zone_temps = next_state[:env.roomnum]
                avg_temp = np.mean(zone_temps)
                temp_error = abs(avg_temp - env.target_temp)
                temp_errors.append(temp_error)
                
                state = next_state
                
                if done:
                    break
            
            # 统计结果
            results[name] = {
                'total_reward': total_reward,
                'avg_reward': total_reward / 24,
                'avg_temp_error': np.mean(temp_errors),
                'max_temp_error': np.max(temp_errors),
            }
            
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  平均奖励: {total_reward/24:.2f}")
            print(f"  平均温度误差: {np.mean(temp_errors):.2f}°C")
            print(f"  最大温度误差: {np.max(temp_errors):.2f}°C")
        
        except Exception as e:
            print(f"  ✗ {name} 控制器测试失败: {e}")
            results[name] = None
    
    # 打印对比结果
    print("\n" + "-" * 60)
    print("性能对比:")
    print(f"{'控制器':<12} {'总奖励':<12} {'平均奖励':<12} {'平均误差':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        if result is not None:
            print(f"{name:<12} {result['total_reward']:<12.2f} "
                  f"{result['avg_reward']:<12.2f} {result['avg_temp_error']:<12.2f}")
    
    print("\n✓ 性能对比测试完成")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("  BEAR 建筑环境专家控制器测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("控制器创建", test_controller_creation),
        ("MPC 控制器", test_mpc_controller),
        ("PID 控制器", test_pid_controller),
        ("规则控制器", test_rule_controller),
        ("集成专家控制器", test_integrated_expert),
        ("控制器性能对比", test_controller_performance),
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
        print("\n🎉 所有测试通过！专家控制器功能正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

