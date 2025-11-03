# ========================================
# GUI版本的主训练程序
# ========================================
# 带图形界面的扩散模型训练程序
# 可通过GUI配置参数、启动训练、查看实时输出

# ---- force local package resolution ----
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # ← 用 insert(0) 把项目根放最前面，优先于 site-packages

# 如果你之前运行过，可能已经把“错误的 env 模块”缓存进 sys.modules 了，保险起见清掉：
if "env" in sys.modules and not os.path.isfile(os.path.join(ROOT, "env", "__init__.py")):
    # 只有在确实不是你项目里的 env 时才删除缓存（这里是双保险，防误删）
    del sys.modules["env"]
# ---- end patch ----

import argparse
from parameter_gui import create_gui
import json
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings
import sys
from types import SimpleNamespace
import traceback

# 忽略警告信息
warnings.filterwarnings('ignore')


class GUILogger:
    """
    GUI日志记录器
    将标准输出重定向到GUI的文本框
    """
    def __init__(self, update_func):
        self.update_func = update_func

    def write(self, message):
        """写入消息到GUI"""
        self.update_func(message)

    def flush(self):
        """刷新缓冲（空操作）"""
        pass


def get_device(requested_device):
    """
    检查并返回可用的计算设备
    
    如果请求CUDA但不可用，则回退到CPU
    """
    if torch.cuda.is_available() and 'cuda' in requested_device:
        return requested_device
    print(f"CUDA不可用。使用CPU代替{requested_device}")
    return 'cpu'

def main(args, update_output, stop_training):
    """
    GUI版本的主训练函数
    
    参数：
    - args: 从GUI获取的参数字典
    - update_output: GUI输出更新回调函数
    - stop_training: 检查是否停止训练的回调函数
    
    与命令行版本的区别：
    - 输出重定向到GUI文本框
    - 支持用户手动停止训练
    - 实时更新训练信息到GUI
    """
    # ========== 重定向输出到GUI ==========
    sys.stdout = GUILogger(update_output)
    sys.stderr = GUILogger(update_output)

    print("开始训练，参数配置：")
    print(json.dumps(args, indent=2))

    # 转换参数字典为对象
    args_obj = SimpleNamespace(**args)

    # ========== 创建环境 ==========
    env, train_envs, test_envs = make_aigc_env(args_obj.training_num, args_obj.test_num)
    args_obj.state_shape = env.observation_space.shape[0]
    args_obj.action_shape = env.action_space.n
    args_obj.max_action = 1.
    args_obj.exploration_noise = args_obj.exploration_noise * args_obj.max_action

    # ========== 创建Actor网络（扩散模型） ==========
    print("初始化Actor网络（扩散模型）...")
    actor_net = MLP(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape
    )
    actor = Diffusion(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape,
        model=actor_net,
        max_action=args_obj.max_action,
        beta_schedule=args_obj.beta_schedule,
        n_timesteps=args_obj.n_timesteps,
        bc_coef=args_obj.bc_coef
    ).to(args_obj.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args_obj.actor_lr,
        weight_decay=args_obj.wd
    )

    # ========== 创建Critic网络（双Q网络） ==========
    print("初始化Critic网络（双Q网络）...")
    critic = DoubleCritic(
        state_dim=args_obj.state_shape,
        action_dim=args_obj.action_shape
    ).to(args_obj.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args_obj.critic_lr,
        weight_decay=args_obj.wd
    )

    # ========== 设置日志系统 ==========
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args_obj.logdir, args_obj.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args_obj))
    logger = TensorboardLogger(writer)
    print(f"日志保存路径: {log_path}")

    # ========== 定义策略 ==========
    print("初始化DiffusionOPT策略...")
    policy = DiffusionOPT(
        args_obj.state_shape,
        actor,
        actor_optim,
        args_obj.action_shape,
        critic,
        critic_optim,
        args_obj.device,
        tau=args_obj.tau,
        gamma=args_obj.gamma,
        estimation_step=args_obj.n_step,
        lr_decay=args_obj.lr_decay,
        lr_maxt=args_obj.epoch,
        bc_coef=args_obj.bc_coef,
        action_space=env.action_space,
        exploration_noise=args_obj.exploration_noise,
    )

    # ========== 加载预训练模型（如果提供） ==========
    if args_obj.resume_path:
        ckpt = torch.load(args_obj.resume_path, map_location=args_obj.device)
        policy.load_state_dict(ckpt)
        print("已加载模型：", args_obj.resume_path)

    # ========== 设置经验回放缓冲区 ==========
    if args_obj.prioritized_replay:
        print("使用优先经验回放...")
        buffer = PrioritizedVectorReplayBuffer(
            args_obj.buffer_size,
            buffer_num=len(train_envs),
            alpha=args_obj.prior_alpha,
            beta=args_obj.prior_beta,
        )
    else:
        print("使用普通经验回放...")
        buffer = VectorReplayBuffer(
            args_obj.buffer_size,
            buffer_num=len(train_envs)
        )

    # ========== 设置数据收集器 ==========
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        """保存最佳模型"""
        save_path = os.path.join(log_path, 'policy.pth')
        torch.save(policy.state_dict(), save_path)
        print(f"保存最佳模型到: {save_path}")

    def train_callback(epoch: int, env_step: int, **kwargs):
        """训练回调：显示进度"""
        print(f"轮次: {epoch}, 环境步数: {env_step}")

    def stop_fn(reward, **kwargs):
        """停止函数：检查用户是否点击停止按钮"""
        if stop_training():
            print(f"用户停止训练。最佳奖励: {reward}")
            return True
        return False

    # ========== 开始训练 ==========
    if not args_obj.watch:
        print("=" * 60)
        print("开始训练...")
        print(f"设备: {args_obj.device}")
        print(f"扩散步数: {args_obj.n_timesteps}")
        print(f"训练模式: {'行为克隆（有专家数据）' if args_obj.bc_coef else '策略梯度（无专家数据）'}")
        print(f"总轮次: {args_obj.epoch}")
        print("=" * 60)
        
        try:
            result = offpolicy_trainer(
                policy,
                train_collector,
                test_collector,
                args_obj.epoch,
                args_obj.step_per_epoch,
                args_obj.step_per_collect,
                args_obj.test_num,
                args_obj.batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
                test_in_train=False,
                stop_fn=stop_fn,
                train_fn=train_callback
            )
            print("\n训练结果：")
            pprint.pprint(result)
            print("训练成功完成！")
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            print(traceback.format_exc())
    
    print("训练流程结束。")

    # ========== 推理/观察模式 ==========
    if args_obj.watch:
        print("进入观察模式...")
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"最终奖励: {rews.mean():.4f}, 回合长度: {lens.mean():.0f}")





if __name__ == '__main__':
    """
    GUI程序入口
    
    启动GUI界面，用户可以：
    1. 在GUI中配置所有训练参数
    2. 点击"提交并开始训练"按钮启动训练
    3. 在右侧文本框实时查看训练输出
    4. 点击"停止训练"按钮随时中止训练
    5. 点击"启动TensorBoard"查看训练曲线
    """
    # 创建GUI
    root, start_training, update_output, stop_training = create_gui()
    # 进入主循环（阻塞直到窗口关闭）
    root.mainloop()
