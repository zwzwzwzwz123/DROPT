#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩散模型动作生成演示脚本

功能：
1. 展示扩散模型如何从噪声逐步生成最优动作
2. 可视化去噪过程
3. 对比不同扩散步数的效果
4. 与传统Actor对比

使用方法：
    python scripts/demo_diffusion_action_generation.py
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.diffusion import Diffusion
from diffusion.model import MLP


def create_sample_state():
    """创建一个示例状态（OfficeSmall建筑）"""
    state = np.array([
        # 房间温度 (6个) - 当前温度略高于目标22°C
        24.5, 23.8, 25.2, 24.1, 23.5, 24.8,
        # 室外温度 (1个) - 炎热天气
        32.0,
        # GHI - 全局水平辐照度 (6个) - 归一化值
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        # 地面温度 (1个)
        28.0,
        # 人员热负荷 (6个) - 归一化值
        0.12, 0.15, 0.10, 0.13, 0.14, 0.11
    ], dtype=np.float32)
    return state


def demo_basic_generation():
    """演示1: 基础动作生成"""
    print("=" * 60)
    print("演示1: 扩散模型基础动作生成")
    print("=" * 60)
    
    # 配置
    state_dim = 20
    action_dim = 6
    hidden_dim = 256
    diffusion_steps = 20
    
    # 创建模型
    print("\n正在创建扩散模型...")
    actor_net = MLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        t_dim=16
    )
    
    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=1.0,
        beta_schedule='vp',
        n_timesteps=diffusion_steps,
        clip_denoised=True,
        bc_coef=False
    )
    
    print(f"✓ 模型创建成功")
    print(f"  - 状态维度: {state_dim}")
    print(f"  - 动作维度: {action_dim}")
    print(f"  - 扩散步数: {diffusion_steps}")
    print(f"  - 网络参数量: {sum(p.numel() for p in actor_net.parameters()):,}")
    
    # 准备状态
    state = create_sample_state()
    state_tensor = torch.from_numpy(state).unsqueeze(0)  # [1, 20]
    
    print("\n当前环境状态:")
    print(f"  - 房间温度: {state[:6]}")
    print(f"  - 室外温度: {state[6]:.1f}°C")
    print(f"  - 目标温度: 22.0°C")
    
    # 生成动作
    print("\n正在生成动作...")
    with torch.no_grad():
        action = diffusion_actor.sample(state_tensor, verbose=False)
    
    action_np = action.numpy()[0]
    
    print("\n生成的动作:")
    room_names = ['房间1', '房间2', '房间3', '房间4', '房间5', '房间6']
    for i, (name, act) in enumerate(zip(room_names, action_np)):
        action_type = "制热" if act > 0 else "制冷"
        percentage = abs(act) * 100
        current_temp = state[i]
        print(f"  {name}: {act:+.3f} → {action_type} {percentage:.1f}% (当前温度: {current_temp:.1f}°C)")
    
    # 转换为实际功率
    max_power = 8000  # W
    actual_power = action_np * max_power
    print(f"\n实际功率 (W): {actual_power}")
    print(f"总能耗: {np.sum(np.abs(actual_power)):.0f} W")


def demo_denoising_process():
    """演示2: 可视化去噪过程"""
    print("\n" + "=" * 60)
    print("演示2: 可视化去噪过程")
    print("=" * 60)
    
    # 配置
    state_dim = 20
    action_dim = 6
    diffusion_steps = 20
    
    # 创建模型
    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, t_dim=16)
    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=1.0,
        n_timesteps=diffusion_steps,
        bc_coef=False
    )
    
    # 准备状态
    state = create_sample_state()
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    
    # 生成动作并记录中间步骤
    print("\n正在生成动作并记录去噪过程...")
    with torch.no_grad():
        action, diffusion_steps_data = diffusion_actor.sample(
            state_tensor,
            return_diffusion=True,
            verbose=False
        )
    
    # 提取数据
    steps = diffusion_steps_data[0].numpy()  # [T+1, 6]
    
    print(f"✓ 记录了 {len(steps)} 个时间步的数据")
    
    # 绘制去噪过程
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    room_names = ['房间1', '房间2', '房间3', '房间4', '房间5', '房间6']
    
    for i, (ax, name) in enumerate(zip(axes.flat, room_names)):
        # 绘制该房间动作的演化
        time_steps = np.arange(len(steps))
        ax.plot(time_steps, steps[:, i], marker='o', linewidth=2, markersize=4, label='动作值')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='零线')
        ax.axhline(y=1, color='g', linestyle=':', alpha=0.3)
        ax.axhline(y=-1, color='g', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('扩散步数 t (T→0)', fontsize=11)
        ax.set_ylabel('动作值', fontsize=11)
        ax.set_title(f'{name}的去噪过程', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # 标注初始和最终值
        initial_val = steps[0, i]
        final_val = steps[-1, i]
        ax.text(0, initial_val, f'{initial_val:.2f}', ha='right', va='bottom', fontsize=9)
        
        action_type = "制热" if final_val > 0 else "制冷"
        ax.text(len(steps)-1, final_val, 
                f'{action_type}\n{abs(final_val):.2f}',
                ha='left', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 设置y轴范围
        ax.set_ylim(-2, 2)
    
    plt.suptitle('扩散模型去噪过程：从随机噪声到最优动作', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = 'diffusion_denoising_process.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可视化图片已保存到: {save_path}")
    
    # 显示图片
    plt.show()


def demo_compare_steps():
    """演示3: 对比不同扩散步数"""
    print("\n" + "=" * 60)
    print("演示3: 对比不同扩散步数的效果")
    print("=" * 60)
    
    # 配置
    state_dim = 20
    action_dim = 6
    step_configs = [5, 10, 20, 50]
    
    # 准备状态
    state = create_sample_state()
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    
    results = {}
    
    for n_steps in step_configs:
        print(f"\n测试扩散步数: {n_steps}")
        
        # 创建模型
        actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, t_dim=16)
        diffusion = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=actor_net,
            max_action=1.0,
            n_timesteps=n_steps,
            bc_coef=False
        )
        
        # 多次采样
        actions = []
        for _ in range(10):
            with torch.no_grad():
                action = diffusion.sample(state_tensor, verbose=False)
            actions.append(action.numpy()[0])
        
        # 统计
        actions = np.array(actions)
        mean_action = np.mean(actions, axis=0)
        std_action = np.std(actions, axis=0)
        
        results[n_steps] = {
            'mean': mean_action,
            'std': std_action,
            'actions': actions
        }
        
        print(f"  平均动作: {mean_action}")
        print(f"  标准差: {std_action}")
        print(f"  平均标准差: {np.mean(std_action):.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 平均动作对比
    ax1 = axes[0]
    x = np.arange(action_dim)
    width = 0.2
    for i, n_steps in enumerate(step_configs):
        offset = (i - len(step_configs)/2 + 0.5) * width
        ax1.bar(x + offset, results[n_steps]['mean'], width, 
                label=f'{n_steps}步', alpha=0.8)
    
    ax1.set_xlabel('房间编号', fontsize=11)
    ax1.set_ylabel('平均动作值', fontsize=11)
    ax1.set_title('不同扩散步数的平均动作对比', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'房间{i+1}' for i in range(action_dim)])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 子图2: 标准差对比
    ax2 = axes[1]
    avg_stds = [np.mean(results[n]['std']) for n in step_configs]
    ax2.plot(step_configs, avg_stds, marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('扩散步数', fontsize=11)
    ax2.set_ylabel('平均标准差', fontsize=11)
    ax2.set_title('扩散步数 vs 动作稳定性', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 标注数值
    for n, std in zip(step_configs, avg_stds):
        ax2.text(n, std, f'{std:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = 'diffusion_steps_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图片已保存到: {save_path}")
    
    plt.show()


def demo_action_distribution():
    """演示4: 动作分布可视化"""
    print("\n" + "=" * 60)
    print("演示4: 动作分布可视化")
    print("=" * 60)
    
    # 配置
    state_dim = 20
    action_dim = 6
    diffusion_steps = 20
    n_samples = 100
    
    # 创建模型
    actor_net = MLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, t_dim=16)
    diffusion_actor = Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        model=actor_net,
        max_action=1.0,
        n_timesteps=diffusion_steps,
        bc_coef=False
    )
    
    # 准备状态
    state = create_sample_state()
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    
    # 生成多个动作样本
    print(f"\n正在生成 {n_samples} 个动作样本...")
    actions = []
    for i in range(n_samples):
        with torch.no_grad():
            action = diffusion_actor.sample(state_tensor, verbose=False)
        actions.append(action.numpy()[0])
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{n_samples}")
    
    actions = np.array(actions)  # [n_samples, 6]
    
    # 可视化分布
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    room_names = ['房间1', '房间2', '房间3', '房间4', '房间5', '房间6']
    
    for i, (ax, name) in enumerate(zip(axes.flat, room_names)):
        # 绘制直方图
        ax.hist(actions[:, i], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零线')
        
        # 统计信息
        mean_val = np.mean(actions[:, i])
        std_val = np.std(actions[:, i])
        ax.axvline(x=mean_val, color='g', linestyle='-', linewidth=2, label=f'均值: {mean_val:.3f}')
        
        ax.set_xlabel('动作值', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{name}的动作分布 (σ={std_val:.3f})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'扩散模型动作分布 ({n_samples}个样本)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    save_path = 'diffusion_action_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 分布图片已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("扩散模型动作生成演示")
    print("=" * 60)
    print("\n本脚本将演示扩散模型如何生成建筑控制动作")
    print("包括4个演示：")
    print("  1. 基础动作生成")
    print("  2. 可视化去噪过程")
    print("  3. 对比不同扩散步数")
    print("  4. 动作分布可视化")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行演示
    demo_basic_generation()
    demo_denoising_process()
    demo_compare_steps()
    demo_action_distribution()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

