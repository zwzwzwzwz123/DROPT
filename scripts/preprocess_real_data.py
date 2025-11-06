# ========================================
# 真实数据预处理脚本
# ========================================
# 清洗、验证、转换真实数据中心数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from scipy.signal import savgol_filter


class RealDataPreprocessor:
    """真实数据预处理器"""
    
    def __init__(self, input_file: str, output_file: str = None):
        """
        初始化预处理器
        
        参数：
        - input_file: 输入CSV文件路径
        - output_file: 输出CSV文件路径
        """
        self.input_file = input_file
        self.output_file = output_file or input_file.replace('.csv', '_processed.csv')
        self.df = None
        self.quality_report = {}
    
    def load_data(self):
        """加载数据"""
        print(f"加载数据: {self.input_file}")
        
        try:
            # 尝试自动解析时间戳
            self.df = pd.read_csv(self.input_file, parse_dates=['timestamp'])
        except:
            # 手动解析
            self.df = pd.read_csv(self.input_file)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"✓ 加载完成: {len(self.df)} 条记录")
        print(f"  时间范围: {self.df['timestamp'].min()} 到 {self.df['timestamp'].max()}")
        print(f"  字段: {list(self.df.columns)}")
        
        return self.df
    
    def validate_data(self):
        """验证数据质量"""
        print("\n" + "="*60)
        print("数据质量验证")
        print("="*60)
        
        # 必需字段检查
        required_fields = ['timestamp', 'T_indoor', 'T_outdoor', 'H_indoor', 'IT_load']
        missing_fields = [f for f in required_fields if f not in self.df.columns]
        
        if missing_fields:
            print(f"❌ 缺少必需字段: {missing_fields}")
            print("   请确保数据包含以下字段:")
            print("   - timestamp: 时间戳")
            print("   - T_indoor: 室内温度 (°C)")
            print("   - T_outdoor: 室外温度 (°C)")
            print("   - H_indoor: 室内湿度 (%)")
            print("   - IT_load: IT负载 (kW)")
            return False
        
        # 缺失值检查
        print("\n1. 缺失值检查:")
        for col in required_fields:
            missing_count = self.df[col].isna().sum()
            missing_rate = missing_count / len(self.df) * 100
            status = "✓" if missing_rate < 5 else "⚠" if missing_rate < 10 else "❌"
            print(f"  {status} {col}: {missing_count} ({missing_rate:.1f}%)")
            self.quality_report[f'{col}_missing_rate'] = missing_rate
        
        # 时间戳检查
        print("\n2. 时间戳检查:")
        time_diffs = self.df['timestamp'].diff().dt.total_seconds()
        median_interval = time_diffs.median()
        print(f"  ✓ 中位采样间隔: {median_interval:.0f} 秒 ({median_interval/60:.1f} 分钟)")
        
        # 检查时间跳跃
        large_gaps = time_diffs[time_diffs > median_interval * 3]
        if len(large_gaps) > 0:
            print(f"  ⚠ 发现 {len(large_gaps)} 处时间跳跃 (>{median_interval*3/60:.0f}分钟)")
        
        self.quality_report['sampling_interval'] = median_interval
        self.quality_report['time_gaps'] = len(large_gaps)
        
        # 物理约束检查
        print("\n3. 物理约束检查:")
        checks = [
            ('T_indoor', 15, 35, '°C'),
            ('T_outdoor', -20, 50, '°C'),
            ('H_indoor', 20, 90, '%'),
            ('IT_load', 0, 1000, 'kW'),
        ]
        
        for col, min_val, max_val, unit in checks:
            if col in self.df.columns:
                violations = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                violation_rate = violations / len(self.df) * 100
                status = "✓" if violation_rate < 1 else "⚠" if violation_rate < 5 else "❌"
                print(f"  {status} {col}: {violations} 违反 [{min_val}, {max_val}]{unit} ({violation_rate:.1f}%)")
                self.quality_report[f'{col}_violations'] = violation_rate
        
        # 能效检查（如果有CRAC_power）
        if 'CRAC_power' in self.df.columns:
            print("\n4. 能效检查:")
            pue = (self.df['IT_load'] + self.df['CRAC_power']) / self.df['IT_load']
            pue_valid = pue[(pue >= 1.0) & (pue <= 3.0)]
            print(f"  ✓ PUE范围: {pue_valid.min():.2f} - {pue_valid.max():.2f}")
            print(f"  ✓ PUE平均: {pue_valid.mean():.2f}")
            
            pue_violations = ((pue < 1.0) | (pue > 3.0)).sum()
            if pue_violations > 0:
                print(f"  ⚠ PUE异常: {pue_violations} 条记录")
            
            self.quality_report['pue_mean'] = pue_valid.mean()
            self.quality_report['pue_violations'] = pue_violations
        
        print("\n" + "="*60)
        overall_quality = "良好" if all(v < 5 for k, v in self.quality_report.items() if 'rate' in k or 'violations' in k) else "需要改进"
        print(f"总体质量: {overall_quality}")
        print("="*60)
        
        return True
    
    def clean_data(self):
        """清洗数据"""
        print("\n" + "="*60)
        print("数据清洗")
        print("="*60)
        
        original_len = len(self.df)
        
        # 1. 删除重复时间戳
        duplicates = self.df.duplicated(subset=['timestamp'], keep='first')
        if duplicates.sum() > 0:
            print(f"1. 删除重复时间戳: {duplicates.sum()} 条")
            self.df = self.df[~duplicates]
        
        # 2. 排序
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        print(f"2. 按时间排序: ✓")
        
        # 3. 处理异常值
        print(f"3. 处理异常值:")
        
        # 温度异常值（3-sigma规则）
        for col in ['T_indoor', 'T_outdoor']:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                outliers = (self.df[col] - mean).abs() > 3 * std
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # 用中位数替换
                    median = self.df[col].median()
                    self.df.loc[outliers, col] = median
                    print(f"   - {col}: {outlier_count} 个异常值 → 替换为中位数 {median:.1f}")
        
        # 物理约束修正
        if 'T_indoor' in self.df.columns:
            self.df['T_indoor'] = self.df['T_indoor'].clip(15, 35)
        if 'T_outdoor' in self.df.columns:
            self.df['T_outdoor'] = self.df['T_outdoor'].clip(-20, 50)
        if 'H_indoor' in self.df.columns:
            self.df['H_indoor'] = self.df['H_indoor'].clip(20, 90)
        if 'IT_load' in self.df.columns:
            self.df['IT_load'] = self.df['IT_load'].clip(0, 1000)
        
        # 4. 处理缺失值
        print(f"4. 处理缺失值:")
        
        for col in ['T_indoor', 'T_outdoor', 'H_indoor', 'IT_load']:
            if col in self.df.columns:
                missing_before = self.df[col].isna().sum()
                
                if missing_before > 0:
                    # 线性插值
                    self.df[col] = self.df[col].interpolate(method='linear', limit=6)
                    
                    # 前向填充剩余
                    self.df[col] = self.df[col].fillna(method='ffill', limit=3)
                    
                    # 后向填充剩余
                    self.df[col] = self.df[col].fillna(method='bfill', limit=3)
                    
                    missing_after = self.df[col].isna().sum()
                    print(f"   - {col}: {missing_before} → {missing_after}")
        
        # 5. 删除仍有缺失的行
        missing_rows = self.df[['T_indoor', 'T_outdoor', 'H_indoor', 'IT_load']].isna().any(axis=1)
        if missing_rows.sum() > 0:
            print(f"5. 删除无法修复的行: {missing_rows.sum()} 条")
            self.df = self.df[~missing_rows]
        
        print(f"\n清洗完成: {original_len} → {len(self.df)} 条记录 (保留 {len(self.df)/original_len*100:.1f}%)")
        print("="*60)
        
        return self.df
    
    def resample_data(self, target_interval: str = '5T'):
        """重采样到统一时间间隔"""
        print(f"\n重采样到 {target_interval} 间隔...")
        
        original_len = len(self.df)
        
        # 设置时间索引
        self.df = self.df.set_index('timestamp')
        
        # 重采样（平均值）
        self.df = self.df.resample(target_interval).mean()
        
        # 插值填充
        self.df = self.df.interpolate(method='linear')
        
        # 重置索引
        self.df = self.df.reset_index()
        
        print(f"✓ 重采样完成: {original_len} → {len(self.df)} 条记录")
        
        return self.df
    
    def smooth_data(self, window: int = 3):
        """平滑数据（去除高频噪声）"""
        print(f"\n平滑数据 (窗口={window})...")
        
        for col in ['T_indoor', 'T_outdoor', 'H_indoor']:
            if col in self.df.columns:
                # 保存原始数据
                self.df[f'{col}_raw'] = self.df[col]
                
                # Savitzky-Golay滤波
                if len(self.df) > window:
                    self.df[col] = savgol_filter(self.df[col], 
                                                  window_length=min(window, len(self.df)//2*2-1), 
                                                  polyorder=2)
        
        print(f"✓ 平滑完成")
        
        return self.df
    
    def save_data(self):
        """保存处理后的数据"""
        print(f"\n保存数据到: {self.output_file}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # 保存CSV
        self.df.to_csv(self.output_file, index=False)
        
        print(f"✓ 保存完成: {len(self.df)} 条记录")
        
        # 保存质量报告
        report_file = self.output_file.replace('.csv', '_quality_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("数据质量报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"输入文件: {self.input_file}\n")
            f.write(f"输出文件: {self.output_file}\n")
            f.write(f"处理时间: {datetime.now()}\n\n")
            
            f.write("质量指标:\n")
            for key, value in self.quality_report.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"✓ 质量报告: {report_file}")
        
        return self.output_file
    
    def plot_data(self, output_dir: str = None):
        """可视化数据"""
        if output_dir is None:
            output_dir = os.path.dirname(self.output_file)
        
        print(f"\n生成可视化图表...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('真实数据可视化', fontsize=16)
        
        # 1. 室内温度
        ax = axes[0, 0]
        ax.plot(self.df['timestamp'], self.df['T_indoor'], linewidth=0.5)
        ax.set_ylabel('室内温度 (°C)')
        ax.set_title('室内温度时间序列')
        ax.grid(True, alpha=0.3)
        
        # 2. 室外温度
        ax = axes[0, 1]
        ax.plot(self.df['timestamp'], self.df['T_outdoor'], linewidth=0.5, color='orange')
        ax.set_ylabel('室外温度 (°C)')
        ax.set_title('室外温度时间序列')
        ax.grid(True, alpha=0.3)
        
        # 3. 湿度
        ax = axes[1, 0]
        ax.plot(self.df['timestamp'], self.df['H_indoor'], linewidth=0.5, color='green')
        ax.set_ylabel('湿度 (%)')
        ax.set_title('室内湿度时间序列')
        ax.grid(True, alpha=0.3)
        
        # 4. IT负载
        ax = axes[1, 1]
        ax.plot(self.df['timestamp'], self.df['IT_load'], linewidth=0.5, color='red')
        ax.set_ylabel('IT负载 (kW)')
        ax.set_title('IT负载时间序列')
        ax.grid(True, alpha=0.3)
        
        # 5. 温度分布
        ax = axes[2, 0]
        ax.hist(self.df['T_indoor'], bins=50, alpha=0.7, label='室内')
        ax.hist(self.df['T_outdoor'], bins=50, alpha=0.7, label='室外')
        ax.set_xlabel('温度 (°C)')
        ax.set_ylabel('频数')
        ax.set_title('温度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 相关性
        ax = axes[2, 1]
        if 'CRAC_power' in self.df.columns:
            ax.scatter(self.df['IT_load'], self.df['CRAC_power'], alpha=0.3, s=1)
            ax.set_xlabel('IT负载 (kW)')
            ax.set_ylabel('空调功率 (kW)')
            ax.set_title('负载 vs 空调功率')
        else:
            ax.scatter(self.df['T_outdoor'], self.df['T_indoor'], alpha=0.3, s=1)
            ax.set_xlabel('室外温度 (°C)')
            ax.set_ylabel('室内温度 (°C)')
            ax.set_title('室外 vs 室内温度')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, 'data_visualization.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化图表: {plot_file}")
        plt.close()
        
        return plot_file


def main():
    parser = argparse.ArgumentParser(description='真实数据预处理')
    parser.add_argument('--input', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径（默认: 输入文件名_processed.csv）')
    parser.add_argument('--resample', type=str, default='5T',
                        help='重采样间隔（默认: 5T = 5分钟）')
    parser.add_argument('--smooth', type=int, default=3,
                        help='平滑窗口大小（默认: 3）')
    parser.add_argument('--validate', action='store_true',
                        help='是否进行数据验证')
    parser.add_argument('--plot', action='store_true',
                        help='是否生成可视化图表')
    
    args = parser.parse_args()
    
    print("="*60)
    print("真实数据预处理工具")
    print("="*60)
    
    # 创建预处理器
    preprocessor = RealDataPreprocessor(args.input, args.output)
    
    # 加载数据
    preprocessor.load_data()
    
    # 验证数据
    if args.validate:
        preprocessor.validate_data()
    
    # 清洗数据
    preprocessor.clean_data()
    
    # 重采样
    preprocessor.resample_data(args.resample)
    
    # 平滑
    if args.smooth > 0:
        preprocessor.smooth_data(args.smooth)
    
    # 保存数据
    output_file = preprocessor.save_data()
    
    # 可视化
    if args.plot:
        preprocessor.plot_data()
    
    print("\n" + "="*60)
    print("预处理完成！")
    print(f"输出文件: {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()

