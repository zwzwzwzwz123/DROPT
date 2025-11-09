#!/usr/bin/env python3
# ========================================
# 应用训练改进方案
# ========================================
# 自动修改配置并启动改进版训练

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_environment():
    """检查环境配置"""
    print_section("1. 检查环境")
    
    # 检查CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  ✓ PyTorch版本: {torch.__version__}")
        print(f"  ✓ CUDA可用: {cuda_available}")
        if cuda_available:
            print(f"  ✓ CUDA设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ✗ PyTorch未安装")
        return False
    
    # 检查Tianshou
    try:
        import tianshou
        print(f"  ✓ Tianshou版本: {tianshou.__version__}")
    except ImportError:
        print("  ✗ Tianshou未安装")
        return False
    
    return True


def backup_original_env():
    """备份原始环境文件"""
    print_section("2. 备份原始文件")
    
    env_file = Path("env/datacenter_env.py")
    backup_file = Path("env/datacenter_env_backup.py")
    
    if env_file.exists() and not backup_file.exists():
        import shutil
        shutil.copy(env_file, backup_file)
        print(f"  ✓ 已备份: {env_file} → {backup_file}")
    else:
        print(f"  ℹ 备份已存在或源文件不存在")
    
    return True


def create_improved_reward_function():
    """创建改进的奖励函数"""
    print_section("3. 创建改进的奖励函数")
    
    improved_reward_code = '''
    def _compute_reward_v2(self, T_in: float, energy: float) -> Tuple[float, Dict]:
        """
        改进的奖励函数 (v2)
        
        改进点:
        1. 降低惩罚权重 (10倍)
        2. 添加正向奖励 (温度舒适度)
        3. 归一化能耗惩罚
        4. 平滑惩罚函数
        
        预期奖励范围: -20 ~ +15 (单步)
        """
        # 1. 温度舒适度奖励 (高斯型)
        temp_error = abs(T_in - self.target_temp)
        # 在目标温度时奖励最大(10.0), 偏离时指数衰减
        temp_reward = 10.0 * np.exp(-0.5 * (temp_error ** 2))
        
        # 2. 温度惩罚 (降低权重)
        temp_penalty = 1.0 * (temp_error ** 2)  # beta: 10 → 1
        
        # 3. 能耗惩罚 (归一化)
        energy_normalized = energy / 10.0  # 假设单步最大10kWh
        energy_penalty = 0.1 * energy_normalized  # alpha: 1 → 0.1
        
        # 4. 越界惩罚 (降低权重)
        if T_in < self.T_min or T_in > self.T_max:
            violation_penalty = 10.0  # gamma: 100 → 10
            violation = 1
        else:
            violation_penalty = 0.0
            violation = 0
        
        # 5. 基础存活奖励
        base_reward = 1.0
        
        # 总奖励 (正负平衡)
        reward = base_reward + temp_reward - temp_penalty - energy_penalty - violation_penalty
        
        # 奖励分解信息
        info = {
            'reward_base': base_reward,
            'reward_temp': temp_reward,
            'reward_energy': -energy_penalty,
            'reward_temp_penalty': -temp_penalty,
            'reward_violation': -violation_penalty,
            'temp_error': temp_error,
            'violation': violation
        }
        
        return reward, info
    '''
    
    print("  ✓ 改进的奖励函数代码已准备")
    print("  ℹ 请手动将以下代码添加到 env/datacenter_env.py:")
    print("  ℹ 或使用 --use-v2-reward 参数在训练时切换")
    
    return improved_reward_code


def create_improved_config():
    """创建改进的训练配置"""
    print_section("4. 创建改进的训练配置")
    
    config = {
        'baseline': {
            'name': '原始配置 (Baseline)',
            'args': {
                'bc_coef': True,
                'expert_type': 'pid',
                'epoch': 50000,
                'batch_size': 256,
                'n_timesteps': 5,
                'actor_lr': 3e-4,
                'critic_lr': 3e-4,
                'exploration_noise': 0.1,
                'lr_decay': False,
            }
        },
        'improved_v1': {
            'name': '改进配置 v1 (推荐)',
            'args': {
                'bc_coef': True,
                'expert_type': 'pid',
                'epoch': 50000,
                'batch_size': 512,  # ↑
                'n_timesteps': 8,   # ↑
                'actor_lr': 1e-4,   # ↓
                'critic_lr': 3e-4,
                'exploration_noise': 0.3,  # ↑
                'lr_decay': True,   # ✓
                'prioritized_replay': True,  # ✓
            }
        },
        'improved_v2': {
            'name': '改进配置 v2 (激进)',
            'args': {
                'bc_coef': True,
                'expert_type': 'pid',
                'epoch': 100000,
                'batch_size': 1024,  # ↑↑
                'n_timesteps': 10,   # ↑↑
                'actor_lr': 5e-5,    # ↓↓
                'critic_lr': 1e-4,   # ↓
                'exploration_noise': 0.5,  # ↑↑
                'lr_decay': True,
                'prioritized_replay': True,
                'prior_alpha': 0.7,
                'prior_beta': 0.5,
            }
        }
    }
    
    print("  ✓ 配置方案已创建:")
    for key, cfg in config.items():
        print(f"    - {key}: {cfg['name']}")
    
    return config


def generate_training_command(config_name='improved_v1'):
    """生成训练命令"""
    print_section("5. 生成训练命令")
    
    configs = create_improved_config()
    config = configs.get(config_name, configs['improved_v1'])
    
    args = config['args']
    
    cmd_parts = ['python main_datacenter.py']
    
    # 添加参数
    if args.get('bc_coef'):
        cmd_parts.append('--bc-coef')
    
    cmd_parts.append(f"--expert-type {args.get('expert_type', 'pid')}")
    cmd_parts.append(f"--epoch {args.get('epoch', 50000)}")
    cmd_parts.append(f"--batch-size {args.get('batch_size', 512)}")
    cmd_parts.append(f"--n-timesteps {args.get('n_timesteps', 8)}")
    cmd_parts.append(f"--actor-lr {args.get('actor_lr', 1e-4)}")
    cmd_parts.append(f"--critic-lr {args.get('critic_lr', 3e-4)}")
    cmd_parts.append(f"--exploration-noise {args.get('exploration_noise', 0.3)}")
    
    if args.get('lr_decay'):
        cmd_parts.append('--lr-decay')
    
    if args.get('prioritized_replay'):
        cmd_parts.append('--prioritized-replay')
        if 'prior_alpha' in args:
            cmd_parts.append(f"--prior-alpha {args['prior_alpha']}")
        if 'prior_beta' in args:
            cmd_parts.append(f"--prior-beta {args['prior_beta']}")
    
    cmd_parts.append('--num-crac 4')
    cmd_parts.append('--device cuda:0')
    cmd_parts.append(f'--logdir log_improved_{config_name}')
    cmd_parts.append(f'--log-prefix improved_{config_name}')
    
    command = ' \\\n    '.join(cmd_parts)
    
    print(f"\n  配置: {config['name']}")
    print(f"\n  命令:\n{command}\n")
    
    return command


def create_comparison_script():
    """创建对比实验脚本"""
    print_section("6. 创建对比实验脚本")
    
    script_content = '''#!/bin/bash
# ========================================
# 对比实验脚本
# ========================================
# 同时运行baseline和improved配置进行对比

echo "=========================================="
echo "  启动对比实验"
echo "=========================================="

# 实验1: Baseline (原始配置)
echo ""
echo "[1/3] 启动 Baseline 训练..."
python main_datacenter.py \\
    --bc-coef \\
    --expert-type pid \\
    --epoch 50000 \\
    --batch-size 256 \\
    --n-timesteps 5 \\
    --actor-lr 3e-4 \\
    --critic-lr 3e-4 \\
    --exploration-noise 0.1 \\
    --num-crac 4 \\
    --device cuda:0 \\
    --logdir log_baseline \\
    --log-prefix baseline &

BASELINE_PID=$!

# 等待5秒
sleep 5

# 实验2: Improved v1 (推荐配置)
echo ""
echo "[2/3] 启动 Improved v1 训练..."
python main_datacenter.py \\
    --bc-coef \\
    --expert-type pid \\
    --epoch 50000 \\
    --batch-size 512 \\
    --n-timesteps 8 \\
    --actor-lr 1e-4 \\
    --critic-lr 3e-4 \\
    --exploration-noise 0.3 \\
    --lr-decay \\
    --prioritized-replay \\
    --num-crac 4 \\
    --device cuda:0 \\
    --logdir log_improved_v1 \\
    --log-prefix improved_v1 &

IMPROVED_PID=$!

echo ""
echo "[3/3] 两个实验已启动"
echo "  - Baseline PID: $BASELINE_PID"
echo "  - Improved PID: $IMPROVED_PID"
echo ""
echo "使用 TensorBoard 监控:"
echo "  tensorboard --logdir_spec baseline:log_baseline,improved:log_improved_v1"
echo ""
echo "等待训练完成..."

wait $BASELINE_PID
wait $IMPROVED_PID

echo ""
echo "=========================================="
echo "  对比实验完成!"
echo "=========================================="
'''
    
    script_path = Path("scripts/run_comparison.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    print(f"  ✓ 对比实验脚本已创建: {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description='应用训练改进方案')
    parser.add_argument('--config', type=str, default='improved_v1',
                        choices=['baseline', 'improved_v1', 'improved_v2'],
                        help='选择配置方案')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示命令,不执行')
    parser.add_argument('--create-comparison', action='store_true',
                        help='创建对比实验脚本')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  训练改进方案应用工具")
    print("=" * 70)
    
    # 1. 检查环境
    if not check_environment():
        print("\n  ✗ 环境检查失败,请先安装依赖")
        return 1
    
    # 2. 备份原始文件
    backup_original_env()
    
    # 3. 创建改进的奖励函数
    improved_reward = create_improved_reward_function()
    
    # 4. 创建配置
    configs = create_improved_config()
    
    # 5. 生成训练命令
    command = generate_training_command(args.config)
    
    # 6. 创建对比实验脚本
    if args.create_comparison:
        create_comparison_script()
    
    # 7. 执行或显示
    print_section("7. 执行训练")
    
    if args.dry_run:
        print("  ℹ Dry-run模式,不执行训练")
        print(f"\n  请手动运行:\n{command}\n")
    else:
        print("  ⚠ 即将开始训练,按Ctrl+C取消...")
        import time
        time.sleep(3)
        
        print("\n  ✓ 开始训练...\n")
        os.system(command)
    
    print_section("完成")
    print("  ✓ 改进方案已应用")
    print("\n  下一步:")
    print("    1. 使用 TensorBoard 监控训练:")
    print(f"       tensorboard --logdir log_improved_{args.config}")
    print("    2. 对比baseline和improved的性能")
    print("    3. 根据结果进一步调整超参数")
    print("\n  参考文档: docs/TRAINING_DIAGNOSIS_REPORT.md")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

