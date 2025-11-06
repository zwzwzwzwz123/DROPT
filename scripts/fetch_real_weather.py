# ========================================
# 获取真实气象数据
# ========================================
# 从公开API获取真实气象数据用于训练

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import time


def fetch_openweathermap_history(
    api_key: str,
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    output_file: str = 'data/real_weather.csv'
):
    """
    从OpenWeatherMap获取历史气象数据
    
    参数：
    - api_key: OpenWeatherMap API密钥
    - lat: 纬度
    - lon: 经度
    - start_date: 开始日期
    - end_date: 结束日期
    - output_file: 输出文件路径
    
    注意：需要订阅OpenWeatherMap的历史数据API
    """
    print(f"从OpenWeatherMap获取气象数据...")
    print(f"位置: ({lat}, {lon})")
    print(f"时间范围: {start_date} 到 {end_date}")
    
    base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Unix时间戳
        timestamp = int(current_date.timestamp())
        
        # API请求
        params = {
            'lat': lat,
            'lon': lon,
            'dt': timestamp,
            'appid': api_key,
            'units': 'metric'  # 摄氏度
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 提取数据
            if 'hourly' in data:
                for hour_data in data['hourly']:
                    all_data.append({
                        'timestamp': datetime.fromtimestamp(hour_data['dt']),
                        'temperature': hour_data['temp'],
                        'humidity': hour_data['humidity'],
                        'pressure': hour_data.get('pressure', 1013),
                        'wind_speed': hour_data.get('wind_speed', 0),
                    })
            
            print(f"✓ 已获取 {current_date.date()} 的数据")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ 获取 {current_date.date()} 数据失败: {e}")
        
        # 下一天
        current_date += timedelta(days=1)
        
        # 避免API限流
        time.sleep(1)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ 真实气象数据已保存到: {output_file}")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 温度范围: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
    print(f"  - 湿度范围: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    
    return df


def fetch_noaa_data(
    station_id: str,
    start_date: str,
    end_date: str,
    output_file: str = 'data/real_weather.csv'
):
    """
    从NOAA (美国国家海洋和大气管理局) 获取气象数据
    
    参数：
    - station_id: 气象站ID (例如: 'GHCND:USW00094728' 纽约中央公园)
    - start_date: 开始日期 (格式: 'YYYY-MM-DD')
    - end_date: 结束日期 (格式: 'YYYY-MM-DD')
    - output_file: 输出文件路径
    
    注意：需要NOAA API token (免费)
    """
    print(f"从NOAA获取气象数据...")
    print(f"气象站: {station_id}")
    print(f"时间范围: {start_date} 到 {end_date}")
    
    # NOAA API endpoint
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    # 需要在 https://www.ncdc.noaa.gov/cdo-web/token 申请token
    token = os.environ.get('NOAA_TOKEN', 'YOUR_NOAA_TOKEN')
    
    headers = {'token': token}
    params = {
        'datasetid': 'GHCND',
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'datatypeid': 'TMAX,TMIN,TAVG',  # 最高温、最低温、平均温
        'units': 'metric',
        'limit': 1000
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 解析数据
        records = []
        for result in data.get('results', []):
            records.append({
                'date': result['date'],
                'datatype': result['datatype'],
                'value': result['value']
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(records)
        df = df.pivot(index='date', columns='datatype', values='value').reset_index()
        
        # 保存
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 真实气象数据已保存到: {output_file}")
        print(f"  - 总记录数: {len(df)}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"✗ 获取数据失败: {e}")
        print("提示: 请确保已设置NOAA_TOKEN环境变量")
        return None


def use_public_dataset(
    dataset_name: str = 'google_datacenter',
    output_file: str = 'data/real_weather.csv'
):
    """
    使用公开的数据中心数据集
    
    参数：
    - dataset_name: 数据集名称
    - output_file: 输出文件路径
    """
    print(f"使用公开数据集: {dataset_name}")
    
    if dataset_name == 'google_datacenter':
        print("\nGoogle数据中心数据集:")
        print("  - 来源: https://www.google.com/about/datacenters/efficiency/")
        print("  - 包含: PUE, 温度, 负载等")
        print("  - 说明: 需要手动下载并放置在data/目录")
        print("\n步骤:")
        print("  1. 访问上述网址")
        print("  2. 下载数据集")
        print("  3. 解压到 data/google_datacenter/")
        print("  4. 运行预处理脚本")
        
    elif dataset_name == 'alibaba_cluster':
        print("\n阿里巴巴集群追踪数据:")
        print("  - 来源: https://github.com/alibaba/clusterdata")
        print("  - 包含: 服务器负载, 功率")
        print("  - 时间跨度: 8天")
        print("\n步骤:")
        print("  1. git clone https://github.com/alibaba/clusterdata")
        print("  2. 提取相关数据")
        print("  3. 运行预处理脚本")
        
    elif dataset_name == 'ashrae':
        print("\nASHRAE数据中心数据:")
        print("  - 来源: ASHRAE RP-1193")
        print("  - 包含: 多个数据中心实测数据")
        print("  - 说明: 需要购买")
        print("\n步骤:")
        print("  1. 访问 https://www.ashrae.org/")
        print("  2. 购买RP-1193数据集")
        print("  3. 运行预处理脚本")
    
    else:
        print(f"未知数据集: {dataset_name}")


def generate_realistic_synthetic_data(
    days: int = 365,
    location: str = 'beijing',
    output_file: str = 'data/realistic_weather.csv'
):
    """
    生成更真实的合成气象数据
    基于真实气象统计特征
    
    参数：
    - days: 生成天数
    - location: 位置（用于选择气候参数）
    - output_file: 输出文件路径
    """
    print(f"生成真实感合成气象数据...")
    print(f"位置: {location}")
    print(f"天数: {days}")
    
    # 不同位置的气候参数（基于真实统计数据）
    climate_params = {
        'beijing': {
            'annual_mean_temp': 12.5,
            'annual_temp_range': 15.0,
            'daily_temp_range': 8.0,
            'mean_humidity': 55.0,
            'humidity_range': 20.0,
        },
        'shanghai': {
            'annual_mean_temp': 17.0,
            'annual_temp_range': 12.0,
            'daily_temp_range': 6.0,
            'mean_humidity': 70.0,
            'humidity_range': 15.0,
        },
        'guangzhou': {
            'annual_mean_temp': 22.0,
            'annual_temp_range': 8.0,
            'daily_temp_range': 5.0,
            'mean_humidity': 75.0,
            'humidity_range': 12.0,
        },
        'newyork': {
            'annual_mean_temp': 12.0,
            'annual_temp_range': 14.0,
            'daily_temp_range': 7.0,
            'mean_humidity': 60.0,
            'humidity_range': 18.0,
        },
    }
    
    params = climate_params.get(location, climate_params['beijing'])
    
    # 生成时间序列
    time_step_minutes = 5
    steps_per_day = 24 * 60 // time_step_minutes
    total_steps = days * steps_per_day
    
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*time_step_minutes) 
                  for i in range(total_steps)]
    
    # 生成温度（使用更复杂的模型）
    temperatures = []
    for i in range(total_steps):
        day = i // steps_per_day
        hour = (i % steps_per_day) * time_step_minutes / 60.0
        
        # 年周期（季节变化）
        seasonal = params['annual_mean_temp'] + \
                   params['annual_temp_range'] * np.sin(2*np.pi*day/365 - np.pi/2)
        
        # 日周期（昼夜变化）
        daily = params['daily_temp_range'] * np.sin(2*np.pi*hour/24 - np.pi/2)
        
        # 天气系统（3-7天周期）
        weather_cycle = 2.0 * np.sin(2*np.pi*day/5 + np.random.uniform(0, 2*np.pi))
        
        # 随机波动（马尔可夫过程）
        if i == 0:
            noise = np.random.normal(0, 1.0)
        else:
            # 自相关噪声
            noise = 0.7 * noise + 0.3 * np.random.normal(0, 1.0)
        
        temp = seasonal + daily + weather_cycle + noise
        temperatures.append(temp)
    
    # 生成湿度（与温度负相关）
    humidities = []
    for i, temp in enumerate(temperatures):
        # 基准湿度（与温度负相关）
        base_humidity = params['mean_humidity'] - \
                        0.5 * (temp - params['annual_mean_temp'])
        
        # 随机波动
        noise = np.random.normal(0, params['humidity_range'] * 0.3)
        
        humidity = base_humidity + noise
        humidity = np.clip(humidity, 20.0, 95.0)
        humidities.append(humidity)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'humidity': humidities,
    })
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ 真实感合成数据已保存到: {output_file}")
    print(f"  - 总记录数: {len(df)}")
    print(f"  - 温度范围: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
    print(f"  - 平均温度: {df['temperature'].mean():.1f}°C")
    print(f"  - 湿度范围: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='获取真实气象数据')
    parser.add_argument('--method', type=str, default='synthetic',
                        choices=['openweathermap', 'noaa', 'public', 'synthetic'],
                        help='数据获取方法')
    parser.add_argument('--location', type=str, default='beijing',
                        help='位置（用于合成数据）')
    parser.add_argument('--days', type=int, default=365,
                        help='生成天数（用于合成数据）')
    parser.add_argument('--output', type=str, default='data/real_weather.csv',
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("真实气象数据获取工具")
    print("="*60)
    
    if args.method == 'openweathermap':
        print("\n注意: 需要OpenWeatherMap API密钥")
        print("获取方式: https://openweathermap.org/api")
        api_key = input("请输入API密钥: ")
        
        # 示例：北京
        lat, lon = 39.9042, 116.4074
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        fetch_openweathermap_history(api_key, lat, lon, start_date, end_date, args.output)
        
    elif args.method == 'noaa':
        print("\n注意: 需要NOAA API token")
        print("获取方式: https://www.ncdc.noaa.gov/cdo-web/token")
        
        # 示例：纽约中央公园
        station_id = 'GHCND:USW00094728'
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        
        fetch_noaa_data(station_id, start_date, end_date, args.output)
        
    elif args.method == 'public':
        print("\n公开数据集信息:")
        use_public_dataset('google_datacenter', args.output)
        
    elif args.method == 'synthetic':
        generate_realistic_synthetic_data(args.days, args.location, args.output)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)

