# ========================================
# 效用函数和水注入算法实现
# ========================================
# 包含：
# 1. 水注入算法（最优功率分配）
# 2. 奖励计算函数
# 3. 信道增益生成

import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def rayleigh_channel_gain(ex, sta):
    """
    生成瑞利分布的信道增益
    
    瑞利衰落常用于模拟无线信道
    
    参数：
    - ex: 均值
    - sta: 标准差
    
    返回：
    - gain: 信道功率增益
    """
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # 取绝对值的平方得到瑞利分布
    gain = np.abs(gain) ** 2
    return gain


def water(s, total_power):
    """
    水注入算法（Waterfilling Algorithm）
    
    计算多信道功率分配的最优解
    这是通信理论中的经典算法，可以最大化系统总容量
    
    算法原理：
    - 像往容器注水一样分配功率
    - 好的信道（增益高）分配更多功率
    - 差的信道（增益低）可能不分配功率
    - 所有信道的"水位"（功率/增益 + 噪声/增益）相同
    
    参数：
    - s: 信道增益向量 g_n
    - total_power: 总功率预算
    
    返回：
    - expert: 归一化的最优功率分配
    - sumdata_rate: 最大数据速率
    - subexpert: 带噪声的次优解（用于模仿学习）
    
    数学公式：
    最大化 Σ log₂(1 + g_n * p_n)
    约束：Σ p_n ≤ total_power, p_n ≥ 0
    
    最优解：p_n = max(α - N₀/g_n, 0)
    其中α是水位（通过二分搜索确定）
    """
    a = total_power  # 总功率预算
    g_n = s          # 信道增益
    N_0 = 1          # 噪声功率（归一化为1）

    # ========== 二分搜索确定水位α ==========
    # 初始化搜索边界
    L = 0  # 下界
    U = a + N_0 * np.sum(1 / (g_n + 1e-6))  # 上界（理论最大水位）

    # 搜索精度
    precision = 1e-6

    # 二分搜索循环
    while U - L > precision:
        alpha_bar = (L + U) / 2  # 当前水位（中点）
        
        # 根据当前水位计算功率分配
        # p_n = max(α - N₀/g_n, 0)
        p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)
        P = np.sum(p_n)  # 总功率

        # 调整搜索边界
        if P > a:  # 功率超支，降低水位
            U = alpha_bar
        else:      # 功率不足，提高水位
            L = alpha_bar

    # ========== 计算最终功率分配 ==========
    p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)

    # ========== 计算数据速率 ==========
    # Shannon公式：C = log₂(1 + SNR)
    SNR = g_n * p_n_final / N_0      # 信噪比
    data_rate = np.log2(1 + SNR)     # 各信道速率
    sumdata_rate = np.sum(data_rate)  # 总速率

    # ========== 返回结果 ==========
    # 归一化功率分配（作为专家动作）
    expert = p_n_final / total_power
    # 添加噪声的次优解（用于数据增强）
    subexpert = p_n_final / total_power + np.random.normal(0, 0.1, len(p_n_final))
    
    return expert, sumdata_rate, subexpert

def CompUtility(State, Aution):
    """
    计算效用（奖励）函数
    
    评估策略输出的功率分配方案的性能
    
    参数：
    - State: 信道增益状态
    - Aution: 策略输出的原始动作
    
    返回：
    - reward: 奖励值（实际速率 - 最优速率）
    - expert_action: 专家动作（水注入算法的最优解）
    - subopt_expert_action: 次优专家动作（带噪声）
    - Aution: 处理后的实际动作
    
    奖励设计：
    reward = Σ log₂(1 + g_n * p_n) - 最优速率
    
    目标：
    - reward → 0 表示接近最优解
    - reward < 0 表示低于最优（需要改进）
    """
    # ========== 处理动作 ==========
    actions = torch.from_numpy(np.array(Aution)).float()
    # 取绝对值确保功率非负
    actions = torch.abs(actions)
    Aution = actions.numpy()
    
    # ========== 归一化功率分配 ==========
    total_power = 3  # 总功率预算
    # 将动作归一化为权重，然后乘以总功率
    normalized_weights = Aution / np.sum(Aution)
    a = normalized_weights * total_power  # 实际功率分配

    # ========== 计算实际数据速率 ==========
    g_n = State  # 信道增益
    SNR = g_n * a  # 信噪比
    data_rate = np.log2(1 + SNR)  # Shannon公式

    # ========== 计算最优解（专家动作） ==========
    expert_action, sumdata_rate, subopt_expert_action = water(g_n, total_power)

    # ========== 计算奖励 ==========
    # 奖励 = 实际速率 - 最优速率
    # 目标是让这个差值尽可能接近0
    reward = np.sum(data_rate) - sumdata_rate

    return reward, expert_action, subopt_expert_action, Aution