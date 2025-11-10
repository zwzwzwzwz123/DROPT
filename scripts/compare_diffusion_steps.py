#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩散步数对比实验脚本

功能:
1. 自动训练不同扩散步数的模型
2. 对比训练效率和性能
3. 生成对比报告

使用方法:
    python scripts/compare_diffusion_steps.py --steps 5 10 15 --epochs 5000
"""

import argparse
import subprocess
import os
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def run_training(diffusion_steps, epochs, building_type, weather_type, log_prefix):
    """
    运行单次训练实验
    
    参数:
    - diffusion_steps: 扩散步数
    - epochs: 训练轮次
    - building_type: 建筑类型
    - weather_type: 气候类型
    - log_prefix: 日志前缀
    
    返回:
    - log_path: 日志路径
    - training_time: 训练时间(秒)
    """
    print(f"\n{'='*60}")
    print(f"开始训练: {diffusion_steps}步扩散模型")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        "python", "main_building.py",
        "--diffusion-steps", str(diffusion_steps),
        "--epoch", str(epochs),
        "--building-type", building_type,
        "--weather-type", weather_type,
        "--log-prefix", f"{log_prefix}_{diffusion_steps}steps",
        "--save-interval", "1000",
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None, None
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ 训练完成!")
    print(f"⏱️  训练时间: {training_time/60:.2f} 分钟")
    
    # 查找日志路径
    log_dir = "log_building"
    log_folders = [f for f in os.listdir(log_dir) if f.startswith(f"{log_prefix}_{diffusion_steps}steps")]
    if log_folders:
        log_path = os.path.join(log_dir, sorted(log_folders)[-1])
        print(f"📁 日志路径: {log_path}")
        return log_path, training_time
    else:
        print(f"⚠️  未找到日志文件夹")
        return None, training_time


def parse_tensorboard_logs(log_path):
    """
    解析TensorBoard日志
    
    参数:
    - log_path: 日志路径
    
    返回:
    - metrics: 指标字典
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        
        metrics = {}
        
        # 提取关键指标
        for tag in ea.Tags()['scalars']:
            try:
                events = ea.Scalars(tag)
                if events:
                    # 取最后100个值的平均
                    values = [e.value for e in events[-100:]]
                    metrics[tag] = sum(values) / len(values)
            except:
                pass
        
        return metrics
    except Exception as e:
        print(f"⚠️  解析日志失败: {e}")
        return {}


def generate_comparison_report(results, output_dir="reports"):
    """
    生成对比报告
    
    参数:
    - results: 实验结果列表
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f"diffusion_steps_comparison_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📊 CSV报告已保存: {csv_path}")
    
    # 生成图表
    if len(results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('扩散步数对比分析', fontsize=16)
        
        # 1. 训练时间对比
        axes[0, 0].bar(df['diffusion_steps'], df['training_time_minutes'])
        axes[0, 0].set_xlabel('扩散步数')
        axes[0, 0].set_ylabel('训练时间 (分钟)')
        axes[0, 0].set_title('训练时间对比')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actor损失对比
        if 'actor_loss' in df.columns:
            axes[0, 1].bar(df['diffusion_steps'], df['actor_loss'])
            axes[0, 1].set_xlabel('扩散步数')
            axes[0, 1].set_ylabel('Actor损失')
            axes[0, 1].set_title('Actor损失对比')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Critic损失对比
        if 'critic_loss' in df.columns:
            axes[1, 0].bar(df['diffusion_steps'], df['critic_loss'])
            axes[1, 0].set_xlabel('扩散步数')
            axes[1, 0].set_ylabel('Critic损失')
            axes[1, 0].set_title('Critic损失对比')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 测试奖励对比
        if 'test_reward' in df.columns:
            axes[1, 1].bar(df['diffusion_steps'], df['test_reward'])
            axes[1, 1].set_xlabel('扩散步数')
            axes[1, 1].set_ylabel('测试奖励')
            axes[1, 1].set_title('测试奖励对比')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"diffusion_steps_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 图表已保存: {plot_path}")
        
        plt.close()
    
    # 生成Markdown报告
    md_path = os.path.join(output_dir, f"diffusion_steps_comparison_{timestamp}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 扩散步数对比实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验配置\n\n")
        f.write(f"- 训练轮次: {results[0]['epochs']}\n")
        f.write(f"- 建筑类型: {results[0]['building_type']}\n")
        f.write(f"- 气候类型: {results[0]['weather_type']}\n\n")
        
        f.write("## 实验结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 结论\n\n")
        
        # 找出最佳配置
        if 'actor_loss' in df.columns:
            best_idx = df['actor_loss'].idxmin()
            best_steps = df.loc[best_idx, 'diffusion_steps']
            f.write(f"- **最低Actor损失**: {best_steps}步 ({df.loc[best_idx, 'actor_loss']:.2f})\n")
        
        if 'test_reward' in df.columns:
            best_idx = df['test_reward'].idxmax()
            best_steps = df.loc[best_idx, 'diffusion_steps']
            f.write(f"- **最高测试奖励**: {best_steps}步 ({df.loc[best_idx, 'test_reward']:.2f})\n")
        
        # 训练效率
        baseline_time = df.loc[0, 'training_time_minutes']
        for idx, row in df.iterrows():
            if idx > 0:
                time_increase = (row['training_time_minutes'] / baseline_time - 1) * 100
                f.write(f"- **{row['diffusion_steps']}步训练时间**: 比{df.loc[0, 'diffusion_steps']}步增加{time_increase:.1f}%\n")
    
    print(f"📝 Markdown报告已保存: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='扩散步数对比实验')
    parser.add_argument('--steps', type=int, nargs='+', default=[5, 10, 15],
                        help='要测试的扩散步数列表 (默认: 5 10 15)')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='每个实验的训练轮次 (默认: 5000)')
    parser.add_argument('--building-type', type=str, default='OfficeSmall',
                        help='建筑类型 (默认: OfficeSmall)')
    parser.add_argument('--weather-type', type=str, default='Hot_Dry',
                        help='气候类型 (默认: Hot_Dry)')
    parser.add_argument('--log-prefix', type=str, default='compare',
                        help='日志前缀 (默认: compare)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("扩散步数对比实验")
    print("="*60)
    print(f"\n测试步数: {args.steps}")
    print(f"训练轮次: {args.epochs}")
    print(f"建筑类型: {args.building_type}")
    print(f"气候类型: {args.weather_type}")
    
    # 运行实验
    results = []
    
    for steps in args.steps:
        log_path, training_time = run_training(
            diffusion_steps=steps,
            epochs=args.epochs,
            building_type=args.building_type,
            weather_type=args.weather_type,
            log_prefix=args.log_prefix
        )
        
        if log_path:
            # 解析日志
            metrics = parse_tensorboard_logs(log_path)
            
            # 记录结果
            result = {
                'diffusion_steps': steps,
                'epochs': args.epochs,
                'building_type': args.building_type,
                'weather_type': args.weather_type,
                'training_time_minutes': training_time / 60,
                'log_path': log_path,
            }
            
            # 添加指标
            if 'loss/actor' in metrics:
                result['actor_loss'] = metrics['loss/actor']
            if 'loss/critic' in metrics:
                result['critic_loss'] = metrics['loss/critic']
            if 'test/reward' in metrics:
                result['test_reward'] = metrics['test/reward']
            
            results.append(result)
    
    # 生成报告
    if results:
        print("\n" + "="*60)
        print("生成对比报告")
        print("="*60)
        generate_comparison_report(results)
    else:
        print("\n❌ 没有成功的实验结果")
    
    print("\n✅ 对比实验完成!")


if __name__ == '__main__':
    main()

