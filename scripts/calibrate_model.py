# ========================================
# 模型参数校准脚本
# ========================================
# 使用真实数据校准热力学模型参数

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.thermal_model import ThermalModel
from scipy.optimize import minimize, differential_evolution
from bayes_opt import BayesianOptimization


class ModelCalibrator:
    """模型参数校准器"""
    
    def __init__(self, real_data_file: str, num_crac: int = 4):
        """
        初始化校准器
        
        参数：
        - real_data_file: 真实数据文件
        - num_crac: CRAC数量
        """
        self.real_data_file = real_data_file
        self.num_crac = num_crac
        
        # 加载真实数据
        self.df = pd.read_csv(real_data_file, parse_dates=['timestamp'])
        print(f"✓ 加载真实数据: {len(self.df)} 条记录")
        
        # 参数范围
        self.param_bounds = {
            'thermal_mass': (500, 2000),      # kJ/K
            'wall_ua': (20, 100),             # kW/K
            'cop_nominal': (2.0, 4.5),        # -
            'crac_capacity': (50, 200),       # kW
        }
        
        # 最佳参数
        self.best_params = None
        self.best_score = -np.inf
    
    def simulate_episode(
        self,
        thermal_mass: float,
        wall_ua: float,
        cop_nominal: float,
        crac_capacity: float,
        episode_length: int = 288
    ) -> dict:
        """
        使用给定参数运行仿真
        
        参数：
        - thermal_mass: 热质量
        - wall_ua: 墙体传热系数
        - cop_nominal: 标称COP
        - crac_capacity: CRAC制冷容量
        - episode_length: episode长度
        
        返回：
        - 仿真结果字典
        """
        # 随机选择起始位置
        max_start = len(self.df) - episode_length
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + episode_length
        
        episode_df = self.df.iloc[start_idx:end_idx]
        
        # 创建热力学模型
        model = ThermalModel(
            room_volume=500.0,  # 固定
            thermal_mass=thermal_mass,
            wall_ua=wall_ua,
            cop_nominal=cop_nominal,
            crac_capacity=crac_capacity,
            num_crac_units=self.num_crac,
            time_step=5.0
        )
        
        # 初始化状态
        T_indoor = episode_df['T_indoor'].iloc[0]
        H_indoor = episode_df['H_indoor'].iloc[0]
        
        # 仿真
        sim_temps = []
        sim_energies = []
        
        for i in range(episode_length):
            # 外部条件
            T_outdoor = episode_df['T_outdoor'].iloc[i]
            IT_load = episode_df['IT_load'].iloc[i]
            
            # 控制动作（如果有的话）
            if all(f'T_setpoint_{j+1}' in episode_df.columns for j in range(self.num_crac)):
                T_setpoints = np.array([episode_df[f'T_setpoint_{j+1}'].iloc[i] 
                                       for j in range(self.num_crac)])
            else:
                T_setpoints = np.ones(self.num_crac) * 20.0  # 默认20°C
            
            if all(f'fan_speed_{j+1}' in episode_df.columns for j in range(self.num_crac)):
                fan_speeds = np.array([episode_df[f'fan_speed_{j+1}'].iloc[i] / 100.0 
                                      for j in range(self.num_crac)])
            else:
                fan_speeds = np.ones(self.num_crac) * 0.7  # 默认70%
            
            # 更新状态
            T_indoor, H_indoor, cooling_power, energy = model.step(
                T_indoor, H_indoor, T_outdoor, IT_load,
                T_setpoints, fan_speeds
            )
            
            sim_temps.append(T_indoor)
            sim_energies.append(energy)
        
        return {
            'temperatures': np.array(sim_temps),
            'energies': np.array(sim_energies),
            'real_temperatures': episode_df['T_indoor'].values,
            'real_energies': episode_df['CRAC_power'].values * (5.0/60.0) if 'CRAC_power' in episode_df.columns else None
        }
    
    def compute_fitness(
        self,
        thermal_mass: float,
        wall_ua: float,
        cop_nominal: float,
        crac_capacity: float,
        num_episodes: int = 10
    ) -> float:
        """
        计算参数的适应度（越大越好）
        
        参数：
        - thermal_mass, wall_ua, cop_nominal, crac_capacity: 模型参数
        - num_episodes: 评估episode数量
        
        返回：
        - 适应度分数（R²）
        """
        temp_errors = []
        energy_errors = []
        
        for _ in range(num_episodes):
            try:
                result = self.simulate_episode(
                    thermal_mass, wall_ua, cop_nominal, crac_capacity
                )
                
                # 温度误差（RMSE）
                temp_rmse = np.sqrt(np.mean((result['temperatures'] - result['real_temperatures'])**2))
                temp_errors.append(temp_rmse)
                
                # 能耗误差（MAPE）
                if result['real_energies'] is not None:
                    energy_mape = np.mean(np.abs((result['energies'] - result['real_energies']) / 
                                                 (result['real_energies'] + 1e-6))) * 100
                    energy_errors.append(energy_mape)
                
            except Exception as e:
                print(f"仿真失败: {e}")
                return -1e6
        
        # 综合评分（温度为主，能耗为辅）
        avg_temp_rmse = np.mean(temp_errors)
        avg_energy_mape = np.mean(energy_errors) if energy_errors else 0
        
        # 适应度：温度RMSE越小越好，能耗MAPE越小越好
        fitness = -avg_temp_rmse - 0.01 * avg_energy_mape
        
        return fitness
    
    def calibrate_least_squares(self) -> dict:
        """使用最小二乘法校准"""
        print("\n使用最小二乘法校准...")
        
        def objective(params):
            thermal_mass, wall_ua, cop_nominal, crac_capacity = params
            fitness = self.compute_fitness(thermal_mass, wall_ua, cop_nominal, crac_capacity, num_episodes=5)
            return -fitness  # 最小化负适应度
        
        # 初始猜测
        x0 = [1200, 50, 3.0, 100]
        
        # 边界
        bounds = [
            self.param_bounds['thermal_mass'],
            self.param_bounds['wall_ua'],
            self.param_bounds['cop_nominal'],
            self.param_bounds['crac_capacity']
        ]
        
        # 优化
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'disp': True})
        
        params = {
            'thermal_mass': result.x[0],
            'wall_ua': result.x[1],
            'cop_nominal': result.x[2],
            'crac_capacity': result.x[3]
        }
        
        return params
    
    def calibrate_genetic(self) -> dict:
        """使用遗传算法校准"""
        print("\n使用遗传算法校准...")
        
        def objective(params):
            thermal_mass, wall_ua, cop_nominal, crac_capacity = params
            fitness = self.compute_fitness(thermal_mass, wall_ua, cop_nominal, crac_capacity, num_episodes=5)
            return -fitness  # 最小化负适应度
        
        # 边界
        bounds = [
            self.param_bounds['thermal_mass'],
            self.param_bounds['wall_ua'],
            self.param_bounds['cop_nominal'],
            self.param_bounds['crac_capacity']
        ]
        
        # 优化
        result = differential_evolution(objective, bounds, maxiter=30, disp=True, workers=1)
        
        params = {
            'thermal_mass': result.x[0],
            'wall_ua': result.x[1],
            'cop_nominal': result.x[2],
            'crac_capacity': result.x[3]
        }
        
        return params
    
    def calibrate_bayesian(self) -> dict:
        """使用贝叶斯优化校准"""
        print("\n使用贝叶斯优化校准...")
        
        def objective(thermal_mass, wall_ua, cop_nominal, crac_capacity):
            fitness = self.compute_fitness(thermal_mass, wall_ua, cop_nominal, crac_capacity, num_episodes=5)
            return fitness
        
        # 创建优化器
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                'thermal_mass': self.param_bounds['thermal_mass'],
                'wall_ua': self.param_bounds['wall_ua'],
                'cop_nominal': self.param_bounds['cop_nominal'],
                'crac_capacity': self.param_bounds['crac_capacity']
            },
            random_state=42,
            verbose=2
        )
        
        # 优化
        optimizer.maximize(init_points=10, n_iter=40)
        
        params = optimizer.max['params']
        
        return params
    
    def validate_params(self, params: dict, num_episodes: int = 20) -> dict:
        """
        验证校准后的参数
        
        参数：
        - params: 参数字典
        - num_episodes: 验证episode数量
        
        返回：
        - 验证指标字典
        """
        print(f"\n验证参数 (使用 {num_episodes} 个episodes)...")
        
        temp_rmses = []
        temp_maes = []
        temp_r2s = []
        energy_mapes = []
        
        for i in range(num_episodes):
            result = self.simulate_episode(
                params['thermal_mass'],
                params['wall_ua'],
                params['cop_nominal'],
                params['crac_capacity']
            )
            
            # 温度指标
            sim_temp = result['temperatures']
            real_temp = result['real_temperatures']
            
            rmse = np.sqrt(np.mean((sim_temp - real_temp)**2))
            mae = np.mean(np.abs(sim_temp - real_temp))
            r2 = 1 - np.sum((sim_temp - real_temp)**2) / np.sum((real_temp - real_temp.mean())**2)
            
            temp_rmses.append(rmse)
            temp_maes.append(mae)
            temp_r2s.append(r2)
            
            # 能耗指标
            if result['real_energies'] is not None:
                sim_energy = result['energies']
                real_energy = result['real_energies']
                mape = np.mean(np.abs((sim_energy - real_energy) / (real_energy + 1e-6))) * 100
                energy_mapes.append(mape)
        
        metrics = {
            'temp_rmse': np.mean(temp_rmses),
            'temp_rmse_std': np.std(temp_rmses),
            'temp_mae': np.mean(temp_maes),
            'temp_r2': np.mean(temp_r2s),
            'energy_mape': np.mean(energy_mapes) if energy_mapes else None,
            'energy_mape_std': np.std(energy_mapes) if energy_mapes else None,
        }
        
        print("\n验证结果:")
        print(f"  温度RMSE: {metrics['temp_rmse']:.3f} ± {metrics['temp_rmse_std']:.3f} °C")
        print(f"  温度MAE:  {metrics['temp_mae']:.3f} °C")
        print(f"  温度R²:   {metrics['temp_r2']:.4f}")
        if metrics['energy_mape'] is not None:
            print(f"  能耗MAPE: {metrics['energy_mape']:.1f} ± {metrics['energy_mape_std']:.1f} %")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='模型参数校准')
    parser.add_argument('--real-data', type=str, required=True,
                        help='真实数据文件路径')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['least_squares', 'genetic', 'bayesian'],
                        help='校准方法')
    parser.add_argument('--num-crac', type=int, default=4,
                        help='CRAC数量')
    parser.add_argument('--output', type=str, default='results/calibrated_params.json',
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("模型参数校准工具")
    print("="*60)
    
    # 创建校准器
    calibrator = ModelCalibrator(args.real_data, args.num_crac)
    
    # 校准
    if args.method == 'least_squares':
        params = calibrator.calibrate_least_squares()
    elif args.method == 'genetic':
        params = calibrator.calibrate_genetic()
    elif args.method == 'bayesian':
        params = calibrator.calibrate_bayesian()
    
    print("\n" + "="*60)
    print("校准完成！")
    print("="*60)
    print("\n最优参数:")
    for key, value in params.items():
        print(f"  {key}: {value:.2f}")
    
    # 验证
    metrics = calibrator.validate_params(params, num_episodes=20)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    result = {
        'parameters': params,
        'validation_metrics': metrics,
        'method': args.method,
        'data_file': args.real_data
    }
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {args.output}")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == '__main__':
    main()

