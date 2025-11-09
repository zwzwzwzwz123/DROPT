#!/usr/bin/env python3
# ========================================
# 测试改进的奖励函数
# ========================================
# 验证新奖励函数的合理性

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from env.datacenter_env import DataCenterEnv


def test_reward_function():
    """测试奖励函数在不同场景下的表现"""
    
    print("=" * 70)
    print("  测试改进的奖励函数")
    print("=" * 70)
    
    # 创建环境
    env = DataCenterEnv(
        num_crac_units=4,
        target_temp=24.0,
        temp_tolerance=2.0,
        energy_weight=1.0,
        temp_weight=10.0,
        violation_penalty=100.0
    )
    
    # 测试场景
    scenarios = [
        {
            'name': '理想情况 (温度完美, 低能耗)',
            'T_in': 24.0,
            'energy': 3.0,
            'expected_reward': '~+10.9'
        },
        {
            'name': '良好情况 (温度偏差0.5°C)',
            'T_in': 24.5,
            'energy': 4.0,
            'expected_reward': '~+9.3'
        },
        {
            'name': '一般情况 (温度偏差1°C)',
            'T_in': 25.0,
            'energy': 5.0,
            'expected_reward': '~+7.5'
        },
        {
            'name': '较差情况 (温度偏差2°C)',
            'T_in': 26.0,
            'energy': 6.0,
            'expected_reward': '~+3.2'
        },
        {
            'name': '临界情况 (刚好不越界)',
            'T_in': 25.9,
            'energy': 7.0,
            'expected_reward': '~+3.5'
        },
        {
            'name': '越界情况 (温度过高)',
            'T_in': 26.5,
            'energy': 8.0,
            'expected_reward': '~-7.5'
        },
        {
            'name': '严重越界 (温度偏差3°C)',
            'T_in': 27.0,
            'energy': 9.0,
            'expected_reward': '~-9.8'
        },
        {
            'name': '极端情况 (温度偏差5°C)',
            'T_in': 29.0,
            'energy': 10.0,
            'expected_reward': '~-35.0'
        },
    ]
    
    print("\n场景测试:")
    print("-" * 70)
    print(f"{'场景':<30} {'温度':<8} {'能耗':<8} {'奖励':<10} {'预期':<10}")
    print("-" * 70)
    
    total_reward_sum = 0
    
    for scenario in scenarios:
        T_in = scenario['T_in']
        energy = scenario['energy']
        
        # 计算奖励
        reward, info = env._compute_reward(T_in, energy)
        
        total_reward_sum += reward
        
        # 打印结果
        print(f"{scenario['name']:<30} {T_in:<8.1f} {energy:<8.1f} {reward:<10.2f} {scenario['expected_reward']:<10}")
        
        # 打印详细分解
        if '--verbose' in sys.argv:
            print(f"  详细:")
            print(f"    基础奖励: {info['reward_base']:.2f}")
            print(f"    温度奖励: {info['reward_temp']:.2f}")
            print(f"    温度惩罚: {info['reward_temp_penalty']:.2f}")
            print(f"    能耗惩罚: {info['reward_energy']:.2f}")
            print(f"    越界惩罚: {info['reward_violation']:.2f}")
            print()
    
    print("-" * 70)
    print(f"平均奖励: {total_reward_sum / len(scenarios):.2f}")
    print()
    
    # 回合累积奖励估算
    print("=" * 70)
    print("  回合累积奖励估算 (288步/回合)")
    print("=" * 70)
    
    episode_scenarios = [
        ('理想策略 (90%理想 + 10%良好)', 0.9 * 10.9 + 0.1 * 9.3),
        ('良好策略 (70%良好 + 30%一般)', 0.7 * 9.3 + 0.3 * 7.5),
        ('一般策略 (50%一般 + 50%较差)', 0.5 * 7.5 + 0.5 * 3.2),
        ('较差策略 (30%较差 + 70%越界)', 0.3 * 3.2 + 0.7 * (-7.5)),
    ]
    
    print()
    for name, avg_reward in episode_scenarios:
        episode_reward = avg_reward * 288
        print(f"  {name:<45} 单步: {avg_reward:>6.2f}  回合: {episode_reward:>8.1f}")
    
    print()
    
    # 对比原始奖励函数
    print("=" * 70)
    print("  对比原始奖励函数")
    print("=" * 70)
    
    print("\n原始奖励函数 (假设):")
    print("  - 温度偏差2°C: -(10*4 + 1*5 + 0) = -45")
    print("  - 越界: -(10*4 + 1*5 + 100) = -145")
    print("  - 回合累积 (50%正常 + 50%越界): 288 * (-45*0.5 + -145*0.5) = -27,360")
    
    print("\n改进奖励函数:")
    print("  - 温度偏差2°C: +3.2")
    print("  - 越界: -7.5")
    print("  - 回合累积 (50%正常 + 50%越界): 288 * (3.2*0.5 + -7.5*0.5) = -619")
    
    print("\n改进效果:")
    print("  ✓ 奖励尺度降低 44倍: -27,360 → -619")
    print("  ✓ 提供正向信号: 负奖励 → 正负混合")
    print("  ✓ 更平滑的奖励曲线")
    
    print()


def test_reward_curve():
    """测试奖励曲线的平滑性"""
    
    print("=" * 70)
    print("  测试奖励曲线")
    print("=" * 70)
    
    env = DataCenterEnv(num_crac_units=4, target_temp=24.0)
    
    # 测试温度范围
    temps = np.linspace(20.0, 30.0, 21)
    energy = 5.0  # 固定能耗
    
    print("\n温度 vs 奖励:")
    print("-" * 50)
    print(f"{'温度(°C)':<12} {'奖励':<10} {'温度误差':<12} {'越界':<6}")
    print("-" * 50)
    
    rewards = []
    for T in temps:
        reward, info = env._compute_reward(T, energy)
        rewards.append(reward)
        
        violation_mark = "✗" if info['violation'] else "✓"
        print(f"{T:<12.1f} {reward:<10.2f} {info['temp_error']:<12.2f} {violation_mark:<6}")
    
    print("-" * 50)
    
    # 检查平滑性
    rewards_arr = np.array(rewards)
    gradients = np.diff(rewards_arr)
    
    print(f"\n奖励曲线统计:")
    print(f"  最大奖励: {rewards_arr.max():.2f}")
    print(f"  最小奖励: {rewards_arr.min():.2f}")
    print(f"  奖励范围: {rewards_arr.max() - rewards_arr.min():.2f}")
    print(f"  平均梯度: {np.mean(np.abs(gradients)):.2f}")
    print(f"  最大梯度: {np.max(np.abs(gradients)):.2f}")
    
    # 检查是否有突变
    large_gradients = np.where(np.abs(gradients) > 5.0)[0]
    if len(large_gradients) > 0:
        print(f"\n  ⚠ 发现 {len(large_gradients)} 处梯度突变 (|Δr| > 5.0)")
        for idx in large_gradients:
            print(f"    T={temps[idx]:.1f}°C → T={temps[idx+1]:.1f}°C: Δr={gradients[idx]:.2f}")
    else:
        print(f"\n  ✓ 奖励曲线平滑, 无明显突变")
    
    print()


def test_episode_simulation():
    """模拟一个完整回合"""
    
    print("=" * 70)
    print("  模拟完整回合")
    print("=" * 70)
    
    env = DataCenterEnv(num_crac_units=4, target_temp=24.0)
    
    # 模拟不同策略
    strategies = {
        '随机策略': lambda: np.random.uniform(-1, 1, 8),
        '保守策略': lambda: np.array([0.0, 0.5] * 4),  # 中等温度, 中等风速
        '激进策略': lambda: np.array([-0.5, 1.0] * 4),  # 低温度, 高风速
    }
    
    print("\n策略对比 (10回合平均):")
    print("-" * 70)
    print(f"{'策略':<15} {'平均奖励':<12} {'能耗(kWh)':<12} {'越界次数':<10}")
    print("-" * 70)
    
    for strategy_name, strategy_fn in strategies.items():
        total_rewards = []
        total_energies = []
        total_violations = []
        
        for episode in range(10):
            env.reset()
            episode_reward = 0
            episode_energy = 0
            episode_violations = 0
            
            for step in range(288):
                action = strategy_fn()
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_energy += info['energy']
                if info['temp_violation']:
                    episode_violations += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_energies.append(episode_energy)
            total_violations.append(episode_violations)
        
        avg_reward = np.mean(total_rewards)
        avg_energy = np.mean(total_energies)
        avg_violations = np.mean(total_violations)
        
        print(f"{strategy_name:<15} {avg_reward:<12.1f} {avg_energy:<12.1f} {avg_violations:<10.1f}")
    
    print("-" * 70)
    print()


def main():
    """主函数"""
    
    print("\n" + "=" * 70)
    print("  改进奖励函数测试套件")
    print("=" * 70)
    print()
    
    # 测试1: 奖励函数基本测试
    test_reward_function()
    
    # 测试2: 奖励曲线测试
    test_reward_curve()
    
    # 测试3: 回合模拟测试
    test_episode_simulation()
    
    print("=" * 70)
    print("  测试完成!")
    print("=" * 70)
    print()
    print("下一步:")
    print("  1. 如果测试通过, 使用新配置重新训练:")
    print("     python scripts/apply_training_improvements.py --config improved_v1")
    print()
    print("  2. 监控训练过程:")
    print("     tensorboard --logdir log_improved_v1")
    print()
    print("  3. 对比baseline和improved的性能")
    print()


if __name__ == '__main__':
    main()

