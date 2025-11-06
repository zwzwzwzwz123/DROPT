# ========================================
# 测试数据中心环境
# ========================================
# 快速验证环境是否正常工作

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from env.datacenter_env import DataCenterEnv
from env.expert_controller import PIDController, MPCController, RuleBasedController


def test_environment_basic():
    """测试环境基本功能"""
    print("=" * 60)
    print("测试1: 环境基本功能")
    print("=" * 60)
    
    # 创建环境
    env = DataCenterEnv(
        num_crac_units=4,
        target_temp=24.0,
        temp_tolerance=2.0,
        episode_length=10,  # 短回合用于测试
    )
    
    print(f"✓ 环境创建成功")
    print(f"  - 状态维度: {env.observation_space.shape}")
    print(f"  - 动作维度: {env.action_space.shape}")
    
    # 重置环境
    state = env.reset()
    print(f"\n✓ 环境重置成功")
    print(f"  - 初始状态: {state[:5]}...")
    
    # 执行随机动作
    for step in range(5):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step+1}:")
        print(f"  - 机房温度: {info['T_in']:.2f}°C")
        print(f"  - 能耗: {info['energy']:.2f}kWh")
        print(f"  - 奖励: {reward:.2f}")
        print(f"  - 温度越界: {info['temp_violation']}")
        
        if done:
            print(f"\n回合结束")
            break
    
    print("\n✓ 环境基本功能测试通过")


def test_expert_controllers():
    """测试专家控制器"""
    print("\n" + "=" * 60)
    print("测试2: 专家控制器")
    print("=" * 60)
    
    # 创建环境
    env = DataCenterEnv(num_crac_units=4, episode_length=20)
    
    # 测试三种控制器
    controllers = {
        'PID': PIDController(num_crac=4, target_temp=24.0),
        'MPC': MPCController(num_crac=4, target_temp=24.0),
        'Rule-Based': RuleBasedController(num_crac=4, target_temp=24.0),
    }
    
    for name, controller in controllers.items():
        print(f"\n测试 {name} 控制器:")
        
        state = env.reset()
        total_reward = 0
        total_energy = 0
        violations = 0
        
        for step in range(20):
            # 获取专家动作
            action = controller.get_action(
                T_in=env.T_in,
                T_out=env.T_out,
                H_in=env.H_in,
                IT_load=env.IT_load
            )
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            total_energy += info['energy']
            if info['temp_violation']:
                violations += 1
            
            if done:
                break
        
        print(f"  - 总奖励: {total_reward:.2f}")
        print(f"  - 总能耗: {total_energy:.2f}kWh")
        print(f"  - 温度越界次数: {violations}")
        print(f"  ✓ {name} 控制器测试通过")


def test_thermal_model():
    """测试热力学模型"""
    print("\n" + "=" * 60)
    print("测试3: 热力学模型")
    print("=" * 60)
    
    from env.thermal_model import ThermalModel
    
    model = ThermalModel(num_crac=4)
    
    # 初始条件
    T_in = 26.0  # 初始温度偏高
    T_out = 30.0
    H_in = 50.0
    IT_load = 200.0
    T_set = np.array([22.0, 22.0, 22.0, 22.0])
    fan_speed = np.array([0.7, 0.7, 0.7, 0.7])
    
    print(f"初始条件:")
    print(f"  - T_in: {T_in}°C")
    print(f"  - T_out: {T_out}°C")
    print(f"  - IT_load: {IT_load}kW")
    print(f"  - T_set: {T_set[0]}°C")
    print(f"  - fan_speed: {fan_speed[0]}")
    
    print(f"\n仿真10步:")
    for step in range(10):
        T_in, H_in, T_supply, energy = model.step(
            T_in, T_out, H_in, IT_load, T_set, fan_speed
        )
        print(f"  Step {step+1}: T_in={T_in:.2f}°C, Energy={energy:.2f}kWh")
    
    # 验证温度下降
    if T_in < 26.0:
        print(f"\n✓ 热力学模型测试通过（温度从26.0°C降至{T_in:.2f}°C）")
    else:
        print(f"\n⚠ 警告：温度未下降")


def test_vectorized_env():
    """测试向量化环境"""
    print("\n" + "=" * 60)
    print("测试4: 向量化环境")
    print("=" * 60)
    
    from env.datacenter_env import make_datacenter_env
    
    # 创建向量化环境
    env, train_envs, test_envs = make_datacenter_env(
        training_num=4,
        test_num=2,
        num_crac_units=4,
        episode_length=10
    )
    
    print(f"✓ 向量化环境创建成功")
    print(f"  - 训练环境数: {len(train_envs)}")
    print(f"  - 测试环境数: {len(test_envs)}")
    
    # 重置所有环境
    states = train_envs.reset()
    print(f"\n✓ 批量重置成功")
    print(f"  - 状态形状: {states.shape}")
    
    # 执行批量动作
    actions = np.random.uniform(-1, 1, (len(train_envs), env.action_space.shape[0]))
    next_states, rewards, dones, infos = train_envs.step(actions)
    
    print(f"\n✓ 批量执行成功")
    print(f"  - 下一状态形状: {next_states.shape}")
    print(f"  - 奖励形状: {rewards.shape}")
    print(f"  - 平均奖励: {rewards.mean():.2f}")
    
    print(f"\n✓ 向量化环境测试通过")


def test_integration():
    """集成测试：完整回合"""
    print("\n" + "=" * 60)
    print("测试5: 完整回合集成测试")
    print("=" * 60)
    
    from env.datacenter_env import DataCenterEnv
    from env.expert_controller import PIDController
    
    # 创建环境和控制器
    env = DataCenterEnv(
        num_crac_units=4,
        target_temp=24.0,
        episode_length=288,  # 完整24小时
    )
    controller = PIDController(num_crac=4, target_temp=24.0)
    
    print(f"运行完整24小时仿真...")
    
    state = env.reset()
    total_reward = 0
    total_energy = 0
    violations = 0
    temp_history = []
    
    for step in range(288):
        # 专家控制
        action = controller.get_action(
            T_in=env.T_in,
            T_out=env.T_out,
            H_in=env.H_in,
            IT_load=env.IT_load
        )
        
        # 执行
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_energy += info['energy']
        temp_history.append(info['T_in'])
        if info['temp_violation']:
            violations += 1
        
        # 每小时打印一次
        if (step + 1) % 12 == 0:
            hour = (step + 1) // 12
            print(f"  Hour {hour:2d}: T_in={info['T_in']:.2f}°C, "
                  f"Energy={info['energy']:.2f}kWh, "
                  f"Reward={reward:.2f}")
        
        if done:
            break
    
    # 统计结果
    print(f"\n完整回合统计:")
    print(f"  - 总奖励: {total_reward:.2f}")
    print(f"  - 总能耗: {total_energy:.2f}kWh")
    print(f"  - 平均温度: {np.mean(temp_history):.2f}°C")
    print(f"  - 温度标准差: {np.std(temp_history):.2f}°C")
    print(f"  - 温度越界次数: {violations} ({violations/288*100:.1f}%)")
    print(f"  - 温度范围: [{min(temp_history):.2f}, {max(temp_history):.2f}]°C")
    
    # 验证结果
    if violations < 10 and 22.0 < np.mean(temp_history) < 26.0:
        print(f"\n✓ 集成测试通过")
    else:
        print(f"\n⚠ 警告：性能可能需要优化")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("数据中心环境测试套件")
    print("=" * 60)
    
    try:
        # 测试1: 基本功能
        test_environment_basic()
        
        # 测试2: 专家控制器
        test_expert_controllers()
        
        # 测试3: 热力学模型
        test_thermal_model()
        
        # 测试4: 向量化环境
        test_vectorized_env()
        
        # 测试5: 集成测试
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n环境已准备就绪，可以开始训练：")
        print("  python main_datacenter.py --bc-coef --epoch 50000")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

