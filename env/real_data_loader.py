# ========================================
# 真实数据加载器
# ========================================
# 从真实数据中采样训练episode

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import random


class RealDataLoader:
    """
    真实数据加载器
    
    功能：
    1. 加载和验证真实数据
    2. 采样训练episode
    3. 提供统计信息
    4. 数据增强
    """
    
    def __init__(
        self,
        data_file: str,
        episode_length: int = 288,  # 24小时，每5分钟一步
        num_crac: int = 4,
        augmentation: bool = True,
    ):
        """
        初始化数据加载器
        
        参数：
        - data_file: 数据文件路径
        - episode_length: episode长度（步数）
        - num_crac: CRAC数量
        - augmentation: 是否启用数据增强
        """
        self.data_file = data_file
        self.episode_length = episode_length
        self.num_crac = num_crac
        self.augmentation = augmentation
        
        # 加载数据
        self.df = self._load_data()
        
        # 计算可用episode数量
        self.num_episodes = max(1, (len(self.df) - episode_length) // (episode_length // 2))
        
        print(f"✓ 真实数据加载器初始化完成")
        print(f"  - 数据点数: {len(self.df)}")
        print(f"  - 可用episode数: {self.num_episodes}")
        print(f"  - Episode长度: {episode_length} 步")
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        print(f"加载真实数据: {self.data_file}")
        
        df = pd.read_csv(self.data_file, parse_dates=['timestamp'])
        
        # 验证必需字段
        required_fields = ['timestamp', 'T_indoor', 'T_outdoor', 'H_indoor', 'IT_load']
        missing_fields = [f for f in required_fields if f not in df.columns]
        
        if missing_fields:
            raise ValueError(f"数据缺少必需字段: {missing_fields}")
        
        # 检查CRAC相关字段
        self.has_crac_data = all(f'T_supply_{i+1}' in df.columns for i in range(self.num_crac))
        self.has_action_data = all(f'fan_speed_{i+1}' in df.columns for i in range(self.num_crac))
        self.has_energy_data = 'CRAC_power' in df.columns
        
        if not self.has_crac_data:
            print(f"  ⚠ 缺少CRAC供风温度数据，将使用默认值")
        if not self.has_action_data:
            print(f"  ⚠ 缺少CRAC动作数据，将使用默认值")
        if not self.has_energy_data:
            print(f"  ⚠ 缺少能耗数据，将使用估算值")
        
        return df
    
    def get_episode(self, episode_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        获取一个episode的数据
        
        参数：
        - episode_idx: episode索引（None表示随机采样）
        
        返回：
        - episode_data: 包含所有数据的字典
        """
        # 随机或指定起始位置
        if episode_idx is None:
            max_start = len(self.df) - self.episode_length
            start_idx = random.randint(0, max_start)
        else:
            start_idx = episode_idx * (self.episode_length // 2)
            start_idx = min(start_idx, len(self.df) - self.episode_length)
        
        end_idx = start_idx + self.episode_length
        
        # 提取数据
        episode_df = self.df.iloc[start_idx:end_idx].copy()
        
        # 构建episode数据
        episode_data = {
            'timestamps': episode_df['timestamp'].values,
            'T_indoor': episode_df['T_indoor'].values,
            'T_outdoor': episode_df['T_outdoor'].values,
            'H_indoor': episode_df['H_indoor'].values,
            'IT_load': episode_df['IT_load'].values,
        }
        
        # CRAC供风温度
        if self.has_crac_data:
            T_supply = np.zeros((self.episode_length, self.num_crac))
            for i in range(self.num_crac):
                T_supply[:, i] = episode_df[f'T_supply_{i+1}'].values
            episode_data['T_supply'] = T_supply
        else:
            # 默认值：室内温度 - 6°C
            T_supply = np.tile(episode_data['T_indoor'][:, None] - 6.0, (1, self.num_crac))
            episode_data['T_supply'] = T_supply
        
        # CRAC动作（归一化到[-1, 1]）
        if self.has_action_data:
            actions = np.zeros((self.episode_length, self.num_crac * 2))
            for i in range(self.num_crac):
                # 温度设定（假设在18-28°C范围）
                if f'T_setpoint_{i+1}' in episode_df.columns:
                    T_set = episode_df[f'T_setpoint_{i+1}'].values
                    actions[:, i*2] = (T_set - 23.0) / 5.0  # 归一化
                else:
                    actions[:, i*2] = 0.0  # 默认中间值
                
                # 风速（假设在0.3-1.0范围）
                fan_speed = episode_df[f'fan_speed_{i+1}'].values / 100.0  # 百分比转小数
                actions[:, i*2+1] = (fan_speed - 0.65) / 0.35  # 归一化
            
            episode_data['actions'] = actions
        else:
            # 默认动作
            episode_data['actions'] = np.zeros((self.episode_length, self.num_crac * 2))
        
        # 能耗
        if self.has_energy_data:
            # 转换为每步能耗（kWh）
            time_step_hours = 5.0 / 60.0  # 5分钟
            episode_data['energy'] = episode_df['CRAC_power'].values * time_step_hours
        else:
            # 估算能耗（基于简单COP模型）
            cooling_load = episode_data['IT_load'] * 0.8  # 假设80%转化为热量
            cop = 3.0  # 假设COP
            time_step_hours = 5.0 / 60.0
            episode_data['energy'] = cooling_load / cop * time_step_hours
        
        # 数据增强
        if self.augmentation:
            episode_data = self._augment_data(episode_data)
        
        return episode_data
    
    def _augment_data(self, episode_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        数据增强
        
        方法：
        1. 添加小的随机噪声
        2. 温度偏移
        3. 负载缩放
        """
        # 1. 温度噪声（±0.2°C）
        episode_data['T_indoor'] += np.random.normal(0, 0.2, size=episode_data['T_indoor'].shape)
        episode_data['T_outdoor'] += np.random.normal(0, 0.3, size=episode_data['T_outdoor'].shape)
        
        # 2. 湿度噪声（±1%）
        episode_data['H_indoor'] += np.random.normal(0, 1.0, size=episode_data['H_indoor'].shape)
        episode_data['H_indoor'] = np.clip(episode_data['H_indoor'], 20, 90)
        
        # 3. 负载缩放（±5%）
        load_scale = np.random.uniform(0.95, 1.05)
        episode_data['IT_load'] *= load_scale
        episode_data['energy'] *= load_scale
        
        # 4. 温度偏移（±0.5°C，模拟不同季节）
        temp_offset = np.random.uniform(-0.5, 0.5)
        episode_data['T_indoor'] += temp_offset
        episode_data['T_outdoor'] += temp_offset * 2  # 室外温度变化更大
        
        return episode_data
    
    def get_statistics(self) -> Dict[str, float]:
        """获取数据统计信息"""
        stats = {
            'T_indoor_mean': self.df['T_indoor'].mean(),
            'T_indoor_std': self.df['T_indoor'].std(),
            'T_indoor_min': self.df['T_indoor'].min(),
            'T_indoor_max': self.df['T_indoor'].max(),
            
            'T_outdoor_mean': self.df['T_outdoor'].mean(),
            'T_outdoor_std': self.df['T_outdoor'].std(),
            'T_outdoor_min': self.df['T_outdoor'].min(),
            'T_outdoor_max': self.df['T_outdoor'].max(),
            
            'H_indoor_mean': self.df['H_indoor'].mean(),
            'H_indoor_std': self.df['H_indoor'].std(),
            
            'IT_load_mean': self.df['IT_load'].mean(),
            'IT_load_std': self.df['IT_load'].std(),
            'IT_load_min': self.df['IT_load'].min(),
            'IT_load_max': self.df['IT_load'].max(),
        }
        
        if self.has_energy_data:
            stats['CRAC_power_mean'] = self.df['CRAC_power'].mean()
            stats['CRAC_power_std'] = self.df['CRAC_power'].std()
            
            # PUE
            pue = (self.df['IT_load'] + self.df['CRAC_power']) / self.df['IT_load']
            stats['PUE_mean'] = pue.mean()
            stats['PUE_std'] = pue.std()
        
        return stats
    
    def split_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple['RealDataLoader', 'RealDataLoader', 'RealDataLoader']:
        """
        划分训练集、验证集、测试集
        
        参数：
        - train_ratio: 训练集比例
        - val_ratio: 验证集比例
        - test_ratio: 测试集比例
        
        返回：
        - train_loader, val_loader, test_loader
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        total_len = len(self.df)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        # 创建临时文件
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        
        # 训练集
        train_file = os.path.join(temp_dir, 'train.csv')
        self.df.iloc[:train_end].to_csv(train_file, index=False)
        train_loader = RealDataLoader(train_file, self.episode_length, self.num_crac, self.augmentation)
        
        # 验证集
        val_file = os.path.join(temp_dir, 'val.csv')
        self.df.iloc[train_end:val_end].to_csv(val_file, index=False)
        val_loader = RealDataLoader(val_file, self.episode_length, self.num_crac, False)  # 验证集不增强
        
        # 测试集
        test_file = os.path.join(temp_dir, 'test.csv')
        self.df.iloc[val_end:].to_csv(test_file, index=False)
        test_loader = RealDataLoader(test_file, self.episode_length, self.num_crac, False)  # 测试集不增强
        
        print(f"✓ 数据集划分完成:")
        print(f"  - 训练集: {len(train_loader.df)} 条 ({train_ratio*100:.0f}%)")
        print(f"  - 验证集: {len(val_loader.df)} 条 ({val_ratio*100:.0f}%)")
        print(f"  - 测试集: {len(test_loader.df)} 条 ({test_ratio*100:.0f}%)")
        
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """测试数据加载器"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python real_data_loader.py <data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    print("="*60)
    print("测试真实数据加载器")
    print("="*60)
    
    # 创建加载器
    loader = RealDataLoader(data_file, episode_length=288, num_crac=4)
    
    # 获取统计信息
    print("\n数据统计:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 采样episode
    print("\n采样episode:")
    episode = loader.get_episode()
    print(f"  - 时间范围: {episode['timestamps'][0]} 到 {episode['timestamps'][-1]}")
    print(f"  - 温度范围: {episode['T_indoor'].min():.1f} - {episode['T_indoor'].max():.1f}°C")
    print(f"  - 负载范围: {episode['IT_load'].min():.1f} - {episode['IT_load'].max():.1f}kW")
    print(f"  - 能耗总计: {episode['energy'].sum():.1f}kWh")
    
    # 划分数据集
    print("\n划分数据集:")
    train_loader, val_loader, test_loader = loader.split_train_val_test()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

