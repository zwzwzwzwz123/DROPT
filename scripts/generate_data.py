# ========================================
# 生成模拟数据
# ========================================
# 生成气象数据和负载轨迹用于训练

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_weather_data(
    days: int = 365,
    time_step_minutes: int = 5,
    output_file: str = 'data/weather_data.csv'
):
    """
    生成模拟气象数据
    
    参数：
    - days: 生成天数
    - time_step_minutes: 时间步长（分钟）
    - output_file: 输出文件路径
    """
    print(f"生成 {days} 天的气象数据...")
    
    # 计算总步数
    steps_per_day = 24 * 60 // time_step_minutes
    total_steps = days * steps_per_day
    
    # 生成时间序列
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*time_step_minutes) 
                  for i in range(total_steps)]
    
    # 生成温度数据
    temperatures = []
    for i in range(total_steps):
        # 天数（用于季节变化）
        day = i // steps_per_day
        # 小时（用于日变化）
        hour = (i % steps_per_day) * time_step_minutes / 60.0
        
        # 季节变化（年周期）
        seasonal_temp = 25.0 + 10.0 * np.sin(2 * np.pi * day / 365 - np.pi/2)
        
        # 日变化（日周期）
        daily_variation = 5.0 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
        
        # 随机噪声
        noise = np.random.normal(0, 1.0)
        
        # 总温度
        temp = seasonal_temp + daily_variation + noise
        temp = np.clip(temp, 5.0, 45.0)
        temperatures.append(temp)
    
    # 生成湿度数据
    humidities = []
    for i in range(total_steps):
        # 湿度与温度负相关
        base_humidity = 60.0 - 0.5 * (temperatures[i] - 25.0)
        noise = np.random.normal(0, 5.0)
        humidity = base_humidity + noise
        humidity = np.clip(humidity, 20.0, 90.0)
        humidities.append(humidity)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidities,
    })
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✓ 气象数据已保存到: {output_file}")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 温度范围: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
    print(f"  - 湿度范围: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    
    return df


def generate_workload_data(
    days: int = 365,
    time_step_minutes: int = 5,
    peak_load: float = 400.0,
    base_load: float = 150.0,
    output_file: str = 'data/workload_trace.csv'
):
    """
    生成模拟IT负载轨迹
    
    参数：
    - days: 生成天数
    - time_step_minutes: 时间步长（分钟）
    - peak_load: 峰值负载 (kW)
    - base_load: 基础负载 (kW)
    - output_file: 输出文件路径
    """
    print(f"\n生成 {days} 天的负载轨迹...")
    
    # 计算总步数
    steps_per_day = 24 * 60 // time_step_minutes
    total_steps = days * steps_per_day
    
    # 生成时间序列
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*time_step_minutes) 
                  for i in range(total_steps)]
    
    # 生成负载数据
    loads = []
    for i in range(total_steps):
        # 天数
        day = i // steps_per_day
        # 小时
        hour = (i % steps_per_day) * time_step_minutes / 60.0
        # 星期几（0=周一，6=周日）
        weekday = (day % 7)
        
        # 工作日 vs 周末
        if weekday < 5:  # 工作日
            # 工作时间（8:00-18:00）负载高
            if 8 <= hour < 18:
                daily_pattern = peak_load
            elif 6 <= hour < 8 or 18 <= hour < 22:
                # 过渡时段
                daily_pattern = (peak_load + base_load) / 2
            else:
                # 夜间
                daily_pattern = base_load
        else:  # 周末
            # 周末负载较低
            daily_pattern = base_load + 0.3 * (peak_load - base_load)
        
        # 添加平滑过渡（正弦波）
        smooth_factor = 0.2 * (peak_load - base_load) * np.sin(2 * np.pi * hour / 24)
        
        # 随机波动
        noise = np.random.normal(0, 0.05 * (peak_load - base_load))
        
        # 总负载
        load = daily_pattern + smooth_factor + noise
        load = np.clip(load, base_load * 0.8, peak_load * 1.1)
        loads.append(load)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'load': loads,
    })
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✓ 负载轨迹已保存到: {output_file}")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 负载范围: {df['load'].min():.1f}kW - {df['load'].max():.1f}kW")
    print(f"  - 平均负载: {df['load'].mean():.1f}kW")
    
    return df


def plot_sample_data(weather_df, workload_df, days_to_plot: int = 7):
    """
    绘制样本数据（可选）
    
    参数：
    - weather_df: 气象数据DataFrame
    - workload_df: 负载数据DataFrame
    - days_to_plot: 绘制天数
    """
    try:
        import matplotlib.pyplot as plt
        
        # 选择前N天的数据
        steps_per_day = 288  # 假设5分钟一步
        n_steps = days_to_plot * steps_per_day
        
        weather_sample = weather_df.iloc[:n_steps]
        workload_sample = workload_df.iloc[:n_steps]
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 温度
        axes[0].plot(weather_sample['temperature'], label='Temperature', color='red')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].set_title(f'Sample Data - First {days_to_plot} Days')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 湿度
        axes[1].plot(weather_sample['humidity'], label='Humidity', color='blue')
        axes[1].set_ylabel('Humidity (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 负载
        axes[2].plot(workload_sample['load'], label='IT Load', color='green')
        axes[2].set_ylabel('Load (kW)')
        axes[2].set_xlabel('Time Steps (5 min each)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'data/sample_data_plot.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n✓ 样本数据图表已保存到: {output_path}")
        
        plt.close()
        
    except ImportError:
        print("\n⚠ matplotlib未安装，跳过绘图")


def main():
    """
    主函数：生成所有数据
    """
    print("=" * 60)
    print("数据中心模拟数据生成器")
    print("=" * 60)
    
    # 生成气象数据
    weather_df = generate_weather_data(
        days=365,
        time_step_minutes=5,
        output_file='data/weather_data.csv'
    )
    
    # 生成负载轨迹
    workload_df = generate_workload_data(
        days=365,
        time_step_minutes=5,
        peak_load=400.0,
        base_load=150.0,
        output_file='data/workload_trace.csv'
    )
    
    # 绘制样本数据
    plot_sample_data(weather_df, workload_df, days_to_plot=7)
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print("\n使用方法:")
    print("  在训练时添加参数:")
    print("  --use-real-weather --weather-file data/weather_data.csv")
    print("  --workload-file data/workload_trace.csv")


if __name__ == '__main__':
    main()

