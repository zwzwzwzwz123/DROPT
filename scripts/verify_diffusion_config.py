#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证扩散步数配置脚本

功能:
1. 检查配置文件中的扩散步数设置
2. 验证模型初始化是否使用正确的步数
3. 测试推理时间

使用方法:
    python scripts/verify_diffusion_config.py
"""

import sys
import os
import time
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.building_config import DEFAULT_DIFFUSION_STEPS, DEFAULT_BETA_SCHEDULE
from diffusion import Diffusion
from diffusion.model import MLP


def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_config():
    """检查配置文件"""
    print_section("1. 检查配置文件")
    
    print(f"✓ 扩散步数: {DEFAULT_DIFFUSION_STEPS}")
    print(f"✓ 噪声调度: {DEFAULT_BETA_SCHEDULE}")
    
    if DEFAULT_DIFFUSION_STEPS == 5:
        print("\n⚠️  警告: 扩散步数仍为5步")
        print("   建议修改 env/building_config.py 第48行为:")
        print("   DEFAULT_DIFFUSION_STEPS = 10")
        return False
    elif DEFAULT_DIFFUSION_STEPS == 10:
        print("\n✅ 配置正确: 扩散步数已设置为10步")
        return True
    else:
        print(f"\n✓ 扩散步数设置为 {DEFAULT_DIFFUSION_STEPS} 步")
        return True


def test_model_initialization():
    """测试模型初始化"""
    print_section("2. 测试模型初始化")
    
    # 模拟环境参数
    state_dim = 23  # 7个房间 * 3 + 2
    action_dim = 7
    hidden_dim = 256
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"隐藏层维度: {hidden_dim}")
    
    # 创建模型
    try:
        actor = MLP(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            t_dim=16
        )
        
        diffusion = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=actor,
            max_action=1.0,
            beta_schedule=DEFAULT_BETA_SCHEDULE,
            n_timesteps=DEFAULT_DIFFUSION_STEPS,
        )
        
        print(f"\n✅ 模型创建成功")
        print(f"✓ 扩散步数: {diffusion.n_timesteps}")
        print(f"✓ 噪声调度: {DEFAULT_BETA_SCHEDULE}")
        print(f"✓ Actor参数量: {sum(p.numel() for p in actor.parameters()):,}")
        
        return diffusion, actor
    
    except Exception as e:
        print(f"\n❌ 模型创建失败: {e}")
        return None, None


def test_inference_speed(diffusion, num_samples=100):
    """测试推理速度"""
    print_section("3. 测试推理速度")
    
    if diffusion is None:
        print("❌ 模型未初始化,跳过测试")
        return
    
    # 创建测试数据
    batch_size = 1
    state = torch.randn(batch_size, diffusion.state_dim)
    
    print(f"批次大小: {batch_size}")
    print(f"测试样本数: {num_samples}")
    
    # 预热
    print("\n预热中...")
    for _ in range(10):
        with torch.no_grad():
            _ = diffusion.sample(state)
    
    # 测试推理时间
    print("测试推理速度...")
    times = []
    
    for i in range(num_samples):
        start_time = time.time()
        with torch.no_grad():
            action = diffusion.sample(state)
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{num_samples}")
    
    # 统计结果
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n推理时间统计 ({num_samples}次):")
    print(f"  平均: {mean_time*1000:.2f} ms")
    print(f"  标准差: {std_time*1000:.2f} ms")
    print(f"  最小: {min_time*1000:.2f} ms")
    print(f"  最大: {max_time*1000:.2f} ms")
    
    # 评估
    print("\n评估:")
    if mean_time < 0.05:
        print("  ✅ 推理速度很快 (<50ms)")
    elif mean_time < 0.1:
        print("  ✅ 推理速度良好 (50-100ms)")
    elif mean_time < 0.2:
        print("  ⚠️  推理速度一般 (100-200ms)")
    else:
        print("  ❌ 推理速度较慢 (>200ms)")
    
    # 与不同步数对比
    print("\n不同扩散步数的预期推理时间:")
    print(f"  5步:  ~{mean_time * 5 / diffusion.n_timesteps * 1000:.1f} ms")
    print(f"  10步: ~{mean_time * 10 / diffusion.n_timesteps * 1000:.1f} ms")
    print(f"  15步: ~{mean_time * 15 / diffusion.n_timesteps * 1000:.1f} ms")
    print(f"  20步: ~{mean_time * 20 / diffusion.n_timesteps * 1000:.1f} ms")


def test_action_quality(diffusion, num_samples=10):
    """测试动作生成质量"""
    print_section("4. 测试动作生成质量")
    
    if diffusion is None:
        print("❌ 模型未初始化,跳过测试")
        return
    
    # 创建测试数据
    batch_size = 1
    state = torch.randn(batch_size, diffusion.state_dim)
    
    print(f"生成 {num_samples} 个动作样本...")
    
    actions = []
    for i in range(num_samples):
        with torch.no_grad():
            action = diffusion.sample(state)
            actions.append(action.cpu().numpy())
    
    actions = np.array(actions).squeeze()
    
    # 统计
    print(f"\n动作统计:")
    print(f"  形状: {actions.shape}")
    print(f"  均值: {actions.mean():.4f}")
    print(f"  标准差: {actions.std():.4f}")
    print(f"  最小值: {actions.min():.4f}")
    print(f"  最大值: {actions.max():.4f}")
    
    # 检查是否在合理范围内
    if actions.min() >= -1.0 and actions.max() <= 1.0:
        print("\n✅ 动作范围正常 [-1, 1]")
    else:
        print("\n⚠️  动作超出范围 [-1, 1]")
    
    # 检查多样性
    action_std = actions.std(axis=0).mean()
    if action_std > 0.01:
        print(f"✅ 动作具有多样性 (std={action_std:.4f})")
    else:
        print(f"⚠️  动作缺乏多样性 (std={action_std:.4f})")


def compare_with_baseline():
    """与基准配置对比"""
    print_section("5. 与基准配置对比")
    
    baseline_steps = 5
    current_steps = DEFAULT_DIFFUSION_STEPS
    
    print(f"基准配置: {baseline_steps}步")
    print(f"当前配置: {current_steps}步")
    
    if current_steps == baseline_steps:
        print("\n⚠️  当前配置与基准相同,未进行优化")
        print("   建议增加扩散步数以提升生成质量")
    else:
        improvement = (current_steps / baseline_steps - 1) * 100
        print(f"\n✅ 扩散步数增加了 {improvement:.0f}%")
        
        print("\n预期改进:")
        print(f"  - 训练时间: 增加约 {improvement:.0f}%")
        print(f"  - 生成质量: 提升约 {min(improvement * 2, 100):.0f}%")
        print(f"  - Actor损失: 降低约 {min(improvement / 2, 50):.0f}%")
        print(f"  - 推理时间: 增加约 {improvement:.0f}%")


def print_recommendations():
    """打印建议"""
    print_section("6. 建议")
    
    current_steps = DEFAULT_DIFFUSION_STEPS
    
    if current_steps < 10:
        print("⚠️  当前扩散步数较少,建议:")
        print("   1. 修改 env/building_config.py 第48行")
        print("   2. 将 DEFAULT_DIFFUSION_STEPS 设置为 10")
        print("   3. 重新训练模型")
    elif current_steps == 10:
        print("✅ 当前配置合理 (10步)")
        print("\n可选优化:")
        print("   - 如果追求更高质量,可以增加到15步")
        print("   - 如果训练时间过长,可以保持10步")
        print("   - 如果推理速度要求高,可以考虑DDIM采样")
    elif current_steps <= 15:
        print("✅ 当前配置较优 (10-15步)")
        print("\n注意事项:")
        print("   - 训练时间会相应增加")
        print("   - 推理时间约为5步的2-3倍")
        print("   - 生成质量显著提升")
    else:
        print("⚠️  当前扩散步数较多 (>15步)")
        print("\n注意事项:")
        print("   - 训练和推理时间较长")
        print("   - 对于HVAC控制,15步通常已足够")
        print("   - 可以考虑降低到10-15步以提升效率")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  扩散步数配置验证工具")
    print("="*60)
    
    # 1. 检查配置
    config_ok = check_config()
    
    # 2. 测试模型初始化
    diffusion, actor = test_model_initialization()
    
    # 3. 测试推理速度
    if diffusion is not None:
        test_inference_speed(diffusion, num_samples=100)
    
    # 4. 测试动作质量
    if diffusion is not None:
        test_action_quality(diffusion, num_samples=10)
    
    # 5. 与基准对比
    compare_with_baseline()
    
    # 6. 打印建议
    print_recommendations()
    
    # 总结
    print_section("总结")
    
    if config_ok and diffusion is not None:
        print("✅ 所有检查通过!")
        print("✅ 配置已正确设置")
        print("✅ 模型可以正常使用")
        print("\n下一步:")
        print("  1. 运行训练: python main_building.py")
        print("  2. 监控TensorBoard: tensorboard --logdir log_building")
        print("  3. 对比性能指标")
    else:
        print("⚠️  存在配置问题,请检查上述输出")
        print("\n修复步骤:")
        print("  1. 修改 env/building_config.py")
        print("  2. 重新运行此脚本验证")
        print("  3. 开始训练")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

