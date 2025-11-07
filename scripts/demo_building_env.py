#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BEAR 建筑环境使用示例

演示如何使用 BearEnvWrapper 进行基本的环境交互
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from env.building_env_wrapper import BearEnvWrapper


def demo_basic_usage():
    """演示基本使用"""
    print("=" * 60)
    print("  演示 1: 基本使用")
    print("=" * 60)
    
    # 创建环境
    env = BearEnvWrapper(
        building_type='OfficeSmall',
        weather_type='Hot_Dry',
        location='Tucson',
        target_temp=22.0,
        temp_tolerance=2.0,
        max_power=8000,
        time_resolution=3600,  # 1小时
    )
    
    print(f"\n环境信息:")
    print(f"  建筑类型: {env.building_type}")
    print(f"  气候类型: {env.weather_type}")
    print(f"  房间数量: {env.roomnum}")
    print(f"  状态维度: {env.state_dim}")
    print(f"  动作维度: {env.action_dim}")
    
    # 重置环境
    state, info = env.reset()
    print(f"\n初始状态:")
    print(f"  房间温度: {state[:env.roomnum]}")
    print(f"  室外温度: {state[env.roomnum]:.2f}°C")
    
    # 运行几步
    print(f"\n运行 10 步:")
    for step in range(10):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行
        next_state, reward, done, truncated, info = env.step(action)
        
        # 打印信息
        zone_temps = next_state[:env.roomnum]
        avg_temp = np.mean(zone_temps)
        print(f"  步数 {step+1:2d}: 平均温度={avg_temp:5.2f}°C, 奖励={reward:8.2f}")
        
        state = next_state
        
        if done:
            break
    
    print(f"\n✓ 基本使用演示完成")


def demo_temperature_control():
    """演示温度控制"""
    print("\n" + "=" * 60)
    print("  演示 2: 简单温度控制策略")
    print("=" * 60)
    
    # 创建环境
    env = BearEnvWrapper(
        building_type='OfficeSmall',
        weather_type='Hot_Dry',
        location='Tucson',
        target_temp=22.0,
        temp_tolerance=2.0,
    )
    
    # 重置环境
    state, _ = env.reset()
    
    # 记录数据
    steps = []
    avg_temps = []
    rewards = []
    actions_taken = []
    
    print(f"\n使用简单比例控制策略运行 48 步 (2天):")
    
    for step in range(48):
        # 简单比例控制策略
        zone_temps = state[:env.roomnum]
        avg_temp = np.mean(zone_temps)
        
        # 计算温度偏差
        error = avg_temp - env.target_temp
        
        # 比例控制：温度高则制冷（负值），温度低则制热（正值）
        kp = 0.1  # 比例系数
        action = np.ones(env.action_dim) * (-kp * error)
        
        # 限制动作范围
        action = np.clip(action, -1.0, 1.0)
        
        # 执行
        next_state, reward, done, truncated, info = env.step(action)
        
        # 记录数据
        steps.append(step)
        avg_temps.append(avg_temp)
        rewards.append(reward)
        actions_taken.append(np.mean(action))
        
        # 打印信息
        if (step + 1) % 12 == 0:  # 每12步（半天）打印一次
            print(f"  步数 {step+1:2d}: 平均温度={avg_temp:5.2f}°C, "
                  f"动作={np.mean(action):6.3f}, 奖励={reward:8.2f}")
        
        state = next_state
        
        if done:
            break
    
    print(f"\n✓ 温度控制演示完成")
    print(f"  最终平均温度: {avg_temps[-1]:.2f}°C")
    print(f"  目标温度: {env.target_temp:.2f}°C")
    print(f"  温度偏差: {abs(avg_temps[-1] - env.target_temp):.2f}°C")
    print(f"  总奖励: {sum(rewards):.2f}")
    
    # 可视化（如果可用）
    try:
        plot_results(steps, avg_temps, rewards, actions_taken, env.target_temp)
    except Exception as e:
        print(f"\n注意: 无法生成图表 ({e})")


def plot_results(steps, avg_temps, rewards, actions, target_temp):
    """绘制结果"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # 温度曲线
    axes[0].plot(steps, avg_temps, 'b-', label='平均温度')
    axes[0].axhline(y=target_temp, color='r', linestyle='--', label='目标温度')
    axes[0].axhline(y=target_temp+2, color='orange', linestyle=':', alpha=0.5, label='温度上限')
    axes[0].axhline(y=target_temp-2, color='orange', linestyle=':', alpha=0.5, label='温度下限')
    axes[0].set_ylabel('温度 (°C)')
    axes[0].set_title('温度控制效果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 奖励曲线
    axes[1].plot(steps, rewards, 'g-')
    axes[1].set_ylabel('奖励')
    axes[1].set_title('即时奖励')
    axes[1].grid(True, alpha=0.3)
    
    # 动作曲线
    axes[2].plot(steps, actions, 'purple')
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('步数')
    axes[2].set_ylabel('平均动作')
    axes[2].set_title('控制动作 (负值=制冷, 正值=制热)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(project_root, 'scripts', 'building_env_demo.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ 图表已保存到: {output_path}")
    
    # 显示图表（可选）
    # plt.show()


def demo_different_buildings():
    """演示不同建筑类型"""
    print("\n" + "=" * 60)
    print("  演示 3: 不同建筑类型对比")
    print("=" * 60)
    
    building_types = [
        ('OfficeSmall', '小型办公楼'),
        ('Hospital', '医院'),
        ('SchoolPrimary', '小学'),
    ]
    
    print(f"\n建筑类型对比:")
    print(f"{'建筑类型':<20} {'房间数':<10} {'状态维度':<10} {'动作维度':<10}")
    print("-" * 60)
    
    for building_type, chinese_name in building_types:
        try:
            env = BearEnvWrapper(
                building_type=building_type,
                weather_type='Hot_Dry',
                location='Tucson'
            )
            
            print(f"{chinese_name:<18} {env.roomnum:<10} {env.state_dim:<10} {env.action_dim:<10}")
        except Exception as e:
            print(f"{chinese_name:<18} 创建失败: {e}")
    
    print(f"\n✓ 建筑类型对比完成")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  BEAR 建筑环境使用示例")
    print("=" * 60)
    
    try:
        # 演示 1: 基本使用
        demo_basic_usage()
        
        # 演示 2: 温度控制
        demo_temperature_control()
        
        # 演示 3: 不同建筑类型
        demo_different_buildings()
        
        print("\n" + "=" * 60)
        print("  所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

