# ========================================
# 参数敏感性分析脚本
# ========================================
# 分析关键参数对训练性能的影响

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.datacenter_env import DataCenterEnv
from env.expert_controller import PIDController
import argparse


def run_single_episode(env, controller, episode_length=288):
    """
    运行单个回合并收集统计数据
    
    参数：
    - env: 环境实例
    - controller: 控制器实例
    - episode_length: 回合长度
    
    返回：
    - metrics: 性能指标字典
    """
    state = env.reset()
    
    # 统计变量
    total_energy = 0.0
    total_temp_deviation = 0.0
    violation_count = 0
    temp_history = []
    energy_history = []
    
    for step in range(episode_length):
        # 提取状态
        T_in = state[0]
        T_out = state[1]
        H_in = state[2]
        IT_load = state[3]
        
        # 专家控制器生成动作
        action = controller.get_action(T_in, T_out, H_in, IT_load)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 收集统计
        total_energy += info['energy']
        total_temp_deviation += abs(T_in - env.target_temp)
        temp_history.append(T_in)
        energy_history.append(info['energy'])
        
        if info['temp_violation']:
            violation_count += 1
        
        state = next_state
        
        if done:
            break
    
    # 计算指标
    metrics = {
        'total_energy': total_energy,
        'avg_temp_deviation': total_temp_deviation / episode_length,
        'violation_rate': violation_count / episode_length * 100,
        'temp_std': np.std(temp_history),
        'energy_std': np.std(energy_history),
        'max_temp': np.max(temp_history),
        'min_temp': np.min(temp_history),
    }
    
    return metrics


def sensitivity_analysis(
    param_name: str,
    param_range: list,
    base_config: dict,
    num_episodes: int = 10,
    output_dir: str = 'results/sensitivity'
):
    """
    对单个参数进行敏感性分析
    
    参数：
    - param_name: 参数名称
    - param_range: 参数取值范围（相对于基准值的倍数）
    - base_config: 基准配置
    - num_episodes: 每个参数值运行的回合数
    - output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"敏感性分析: {param_name}")
    print(f"{'='*60}")
    
    results = []
    
    for scale in param_range:
        print(f"\n测试 {param_name} = {scale:.2f}x 基准值...")
        
        # 创建修改后的配置
        config = base_config.copy()
        
        # 根据参数名称修改相应的配置
        if param_name == 'thermal_mass':
            # 通过修改room_volume来改变thermal_mass
            config['room_volume'] = base_config.get('room_volume', 1000.0) * scale
        elif param_name == 'wall_ua':
            config['wall_ua'] = base_config.get('wall_ua', 50.0) * scale
        elif param_name == 'cop_nominal':
            config['cop_nominal'] = base_config.get('cop_nominal', 3.0) * scale
        elif param_name == 'crac_capacity':
            config['crac_capacity'] = base_config.get('crac_capacity', 100.0) * scale
        else:
            print(f"警告: 未知参数 {param_name}")
            continue
        
        # 运行多个回合
        episode_metrics = []
        for ep in range(num_episodes):
            # 创建环境
            env = DataCenterEnv(
                num_crac_units=base_config.get('num_crac', 4),
                target_temp=base_config.get('target_temp', 24.0),
                temp_tolerance=base_config.get('temp_tolerance', 2.0),
                energy_weight=base_config.get('energy_weight', 1.0),
                temp_weight=base_config.get('temp_weight', 10.0),
                violation_penalty=base_config.get('violation_penalty', 100.0),
            )
            
            # 创建控制器
            controller = PIDController(
                num_crac=base_config.get('num_crac', 4),
                target_temp=base_config.get('target_temp', 24.0),
            )
            
            # 运行回合
            metrics = run_single_episode(env, controller)
            episode_metrics.append(metrics)
            
            print(f"  回合 {ep+1}/{num_episodes}: "
                  f"能耗={metrics['total_energy']:.1f}kWh, "
                  f"温度偏差={metrics['avg_temp_deviation']:.2f}°C, "
                  f"越界率={metrics['violation_rate']:.1f}%")
        
        # 计算平均指标
        avg_metrics = {
            'param_value': scale,
            'total_energy_mean': np.mean([m['total_energy'] for m in episode_metrics]),
            'total_energy_std': np.std([m['total_energy'] for m in episode_metrics]),
            'avg_temp_deviation_mean': np.mean([m['avg_temp_deviation'] for m in episode_metrics]),
            'avg_temp_deviation_std': np.std([m['avg_temp_deviation'] for m in episode_metrics]),
            'violation_rate_mean': np.mean([m['violation_rate'] for m in episode_metrics]),
            'violation_rate_std': np.std([m['violation_rate'] for m in episode_metrics]),
            'temp_std_mean': np.mean([m['temp_std'] for m in episode_metrics]),
        }
        
        results.append(avg_metrics)
        
        print(f"\n  平均结果:")
        print(f"    能耗: {avg_metrics['total_energy_mean']:.1f} ± {avg_metrics['total_energy_std']:.1f} kWh")
        print(f"    温度偏差: {avg_metrics['avg_temp_deviation_mean']:.2f} ± {avg_metrics['avg_temp_deviation_std']:.2f} °C")
        print(f"    越界率: {avg_metrics['violation_rate_mean']:.1f} ± {avg_metrics['violation_rate_std']:.1f} %")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f'sensitivity_{param_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存到: {csv_path}")
    
    # 绘图
    plot_sensitivity_results(df, param_name, output_dir)
    
    return df


def plot_sensitivity_results(df, param_name, output_dir):
    """
    绘制敏感性分析结果
    
    参数：
    - df: 结果DataFrame
    - param_name: 参数名称
    - output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'敏感性分析: {param_name}', fontsize=16)
    
    # 1. 能耗
    ax = axes[0, 0]
    ax.errorbar(df['param_value'], df['total_energy_mean'], 
                yerr=df['total_energy_std'], marker='o', capsize=5)
    ax.set_xlabel(f'{param_name} (相对基准值)')
    ax.set_ylabel('总能耗 (kWh)')
    ax.set_title('能耗 vs 参数值')
    ax.grid(True, alpha=0.3)
    
    # 2. 温度偏差
    ax = axes[0, 1]
    ax.errorbar(df['param_value'], df['avg_temp_deviation_mean'], 
                yerr=df['avg_temp_deviation_std'], marker='o', capsize=5, color='orange')
    ax.set_xlabel(f'{param_name} (相对基准值)')
    ax.set_ylabel('平均温度偏差 (°C)')
    ax.set_title('温度偏差 vs 参数值')
    ax.grid(True, alpha=0.3)
    
    # 3. 越界率
    ax = axes[1, 0]
    ax.errorbar(df['param_value'], df['violation_rate_mean'], 
                yerr=df['violation_rate_std'], marker='o', capsize=5, color='red')
    ax.set_xlabel(f'{param_name} (相对基准值)')
    ax.set_ylabel('温度越界率 (%)')
    ax.set_title('越界率 vs 参数值')
    ax.grid(True, alpha=0.3)
    
    # 4. 温度标准差
    ax = axes[1, 1]
    ax.plot(df['param_value'], df['temp_std_mean'], marker='o', color='green')
    ax.set_xlabel(f'{param_name} (相对基准值)')
    ax.set_ylabel('温度标准差 (°C)')
    ax.set_title('温度稳定性 vs 参数值')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, f'sensitivity_{param_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存到: {plot_path}")
    plt.close()


def run_full_sensitivity_analysis(output_dir='results/sensitivity'):
    """
    运行完整的敏感性分析
    """
    print("\n" + "="*60)
    print("完整敏感性分析")
    print("="*60)
    
    # 基准配置
    base_config = {
        'num_crac': 4,
        'target_temp': 24.0,
        'temp_tolerance': 2.0,
        'room_volume': 1000.0,
        'wall_ua': 50.0,
        'cop_nominal': 3.0,
        'crac_capacity': 100.0,
        'energy_weight': 1.0,
        'temp_weight': 10.0,
        'violation_penalty': 100.0,
    }
    
    # 参数列表和范围
    params_to_test = {
        'thermal_mass': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        'wall_ua': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        'cop_nominal': [0.7, 0.85, 1.0, 1.15, 1.3],
        'crac_capacity': [0.7, 0.85, 1.0, 1.15, 1.3],
    }
    
    # 运行分析
    all_results = {}
    for param_name, param_range in params_to_test.items():
        df = sensitivity_analysis(
            param_name=param_name,
            param_range=param_range,
            base_config=base_config,
            num_episodes=5,  # 每个参数值运行5个回合
            output_dir=output_dir
        )
        all_results[param_name] = df
    
    # 生成总结报告
    generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*60)
    print("敏感性分析完成！")
    print(f"结果保存在: {output_dir}")
    print("="*60)


def generate_summary_report(all_results, output_dir):
    """
    生成总结报告
    
    参数：
    - all_results: 所有参数的结果字典
    - output_dir: 输出目录
    """
    report_path = os.path.join(output_dir, 'sensitivity_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("敏感性分析总结报告\n")
        f.write("="*60 + "\n\n")
        
        for param_name, df in all_results.items():
            f.write(f"\n参数: {param_name}\n")
            f.write("-"*40 + "\n")
            
            # 计算敏感度（变化率）
            baseline_idx = df['param_value'].sub(1.0).abs().idxmin()
            baseline_energy = df.loc[baseline_idx, 'total_energy_mean']
            baseline_temp = df.loc[baseline_idx, 'avg_temp_deviation_mean']
            baseline_violation = df.loc[baseline_idx, 'violation_rate_mean']
            
            max_energy_change = (df['total_energy_mean'].max() - baseline_energy) / baseline_energy * 100
            max_temp_change = (df['avg_temp_deviation_mean'].max() - baseline_temp) / baseline_temp * 100
            max_violation_change = (df['violation_rate_mean'].max() - baseline_violation) / (baseline_violation + 1e-6) * 100
            
            f.write(f"基准值 (1.0x):\n")
            f.write(f"  能耗: {baseline_energy:.1f} kWh\n")
            f.write(f"  温度偏差: {baseline_temp:.2f} °C\n")
            f.write(f"  越界率: {baseline_violation:.1f} %\n\n")
            
            f.write(f"最大变化:\n")
            f.write(f"  能耗: {max_energy_change:+.1f}%\n")
            f.write(f"  温度偏差: {max_temp_change:+.1f}%\n")
            f.write(f"  越界率: {max_violation_change:+.1f}%\n\n")
            
            # 敏感度评级
            if abs(max_energy_change) > 30:
                sensitivity = "高"
            elif abs(max_energy_change) > 15:
                sensitivity = "中"
            else:
                sensitivity = "低"
            
            f.write(f"敏感度评级: {sensitivity}\n")
            f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("建议:\n")
        f.write("="*60 + "\n")
        f.write("1. 对高敏感度参数进行精确测量和校准\n")
        f.write("2. 对中敏感度参数使用文献典型值\n")
        f.write("3. 对低敏感度参数可使用默认值\n")
    
    print(f"\n✓ 总结报告已保存到: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='参数敏感性分析')
    parser.add_argument('--param', type=str, default='all',
                        choices=['all', 'thermal_mass', 'wall_ua', 'cop_nominal', 'crac_capacity'],
                        help='要分析的参数（默认: all）')
    parser.add_argument('--output-dir', type=str, default='results/sensitivity',
                        help='输出目录（默认: results/sensitivity）')
    parser.add_argument('--num-episodes', type=int, default=5,
                        help='每个参数值运行的回合数（默认: 5）')
    
    args = parser.parse_args()
    
    if args.param == 'all':
        run_full_sensitivity_analysis(output_dir=args.output_dir)
    else:
        base_config = {
            'num_crac': 4,
            'target_temp': 24.0,
            'temp_tolerance': 2.0,
            'room_volume': 1000.0,
            'wall_ua': 50.0,
            'cop_nominal': 3.0,
            'crac_capacity': 100.0,
            'energy_weight': 1.0,
            'temp_weight': 10.0,
            'violation_penalty': 100.0,
        }
        
        param_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        sensitivity_analysis(
            param_name=args.param,
            param_range=param_range,
            base_config=base_config,
            num_episodes=args.num_episodes,
            output_dir=args.output_dir
        )

