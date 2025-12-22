# ========================================
# 训练日志格式化工具
# ========================================
# 提供美化的终端日志输出，使训练过程更清晰易读

import time
import math
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta


class TrainingLogger:
    """训练日志格式化器"""

    def __init__(
        self,
        total_epochs: int,
        reward_scale: float = 1.0,
        diffusion_steps: int = None,
        metrics_getter: Optional[Callable[[str], Optional[Dict[str, float]]]] = None,
        context_info: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化日志格式化器

        参数:
        - total_epochs: 总训练轮次
        - reward_scale: 奖励缩放系数
        - diffusion_steps: 扩散模型步数（可选）
        """
        self.total_epochs = total_epochs  # 总轮次
        self.reward_scale = reward_scale  # 奖励缩放系数
        self.diffusion_steps = diffusion_steps  # 扩散步数
        self.metrics_getter = metrics_getter
        self.context_info = context_info or {}
        self.start_time = time.time()  # 训练开始时间
        self.last_epoch_time = time.time()  # 上一轮次时间
        self.epoch_times = []  # 每轮耗时记录

        # 用于检测异常值的阈值
        self.thresholds = {
            'actor_loss_high': 20.0,  # Actor损失过高阈值
            'critic_loss_high': 300.0,  # Critic损失过高阈值
            'grad_norm_high': 1000.0,  # 梯度范数过高阈值
        }
    
    def format_time(self, seconds: float) -> str:
        """
        格式化时间显示
        
        参数:
        - seconds: 秒数
        
        返回:
        - 格式化的时间字符串
        """
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分"
    
    def get_indicator(self, value: float, threshold: float, lower_is_better: bool = True) -> str:
        """
        获取指标状态指示符
        
        参数:
        - value: 当前值
        - threshold: 阈值
        - lower_is_better: 是否越低越好
        
        返回:
        - 状态指示符（✓ 正常，⚠ 警告）
        """
        if lower_is_better:
            return "✓" if value < threshold else "⚠"
        else:
            return "✓" if value > threshold else "⚠"
    
    def log_epoch(
        self,
        epoch: int,
        train_result: Dict[str, Any],
        test_result: Optional[Dict[str, Any]] = None
    ):
        """
        记录并格式化输出一个epoch的训练信息
        
        参数:
        - epoch: 当前轮次
        - train_result: 训练结果字典
        - test_result: 测试结果字典（可选）
        """
        # 计算时间统计
        current_time = time.time()
        epoch_time = current_time - self.last_epoch_time
        self.last_epoch_time = current_time
        self.epoch_times.append(epoch_time)
        
        # 保留最近100个epoch的时间用于估算
        if len(self.epoch_times) > 100:
            self.epoch_times.pop(0)
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        elapsed_time = current_time - self.start_time
        remaining_epochs = self.total_epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # 提取关键指标
        actor_loss = train_result.get('loss/actor', 0.0)
        critic_loss = train_result.get('loss/critic', 0.0)
        bc_loss = train_result.get('loss/bc', None)
        actor_grad = train_result.get('grad_norm/actor', 0.0)
        critic_grad = train_result.get('grad_norm/critic', 0.0)

        # 奖励值（尝试多个可能的键名）
        train_reward = train_result.get('train/reward',
                                       train_result.get('rew',
                                       train_result.get('rews', 0.0)))
        test_reward = 0.0
        if test_result:
            test_reward = test_result.get('test/reward',
                                         test_result.get('rew',
                                         test_result.get('rews', 0.0)))

        q_mean = train_result.get('q_value/q_mean', 0.0)
        td_error = train_result.get('q_value/td_error', 0.0)

        # 探索率
        exploration_noise = train_result.get('exploration/noise', 0.0)

        # 打印格式化的日志
        print("\n" + "=" * 80)
        epoch_info = f"  Epoch {epoch}/{self.total_epochs}  [{epoch/self.total_epochs*100:.1f}%]"
        if self.diffusion_steps:
            epoch_info += f"  | 扩散步数: {self.diffusion_steps}"
        print(epoch_info)
        print("=" * 80)

        # 上下文提示信息（静态）
        if self.context_info:
            sup_steps = self.context_info.get('support_steps')
            state_dim = self.context_info.get('state_dim')
            action_dim = self.context_info.get('action_dim')
            context_dim = self.context_info.get('context_dim')
            token_dim = self.context_info.get('token_dim')
            summary_dim = self.context_info.get('summary_dim')
            print("\n上下文配置:")
            print(f"  支持步数: {sup_steps}")
            print(f"  状态维度: {state_dim} | 动作维度: {action_dim}")
            print(f"  token维度: {token_dim} | 摘要维度: {summary_dim} | 上下文维度: {context_dim}")
        
        # 损失指标
        print("\n📊 损失指标:")
        actor_indicator = self.get_indicator(actor_loss, self.thresholds['actor_loss_high'])
        critic_indicator = self.get_indicator(critic_loss, self.thresholds['critic_loss_high'])
        print(f"  {actor_indicator} Actor损失:     {actor_loss:>10.3f}")
        print(f"  {critic_indicator} Critic损失:    {critic_loss:>10.3f}")
        if bc_loss is not None:
            print(f"BC损失:       {bc_loss:>10.3f}")
        
        # 梯度信息
        print("\n📈 梯度范数:")
        actor_grad_indicator = self.get_indicator(actor_grad, self.thresholds['grad_norm_high'])
        critic_grad_indicator = self.get_indicator(critic_grad, self.thresholds['grad_norm_high'])
        print(f"  {actor_grad_indicator} Actor梯度:     {actor_grad:>10.3f}")
        print(f"  {critic_grad_indicator} Critic梯度:    {critic_grad:>10.3f}")
        
        # 性能指标
        print("\n🎯 性能指标:")
        print(f"  训练奖励:       {train_reward:>10.2f}  (缩放后)")
        if test_result:
            print(f"  测试奖励:       {test_reward:>10.2f}  (缩放后)")
        print(f"  真实训练奖励:   {train_reward/self.reward_scale:>10.2f}")
        if test_result:
            print(f"  真实测试奖励:   {test_reward/self.reward_scale:>10.2f}")

        if self.metrics_getter:
            train_metrics = self.metrics_getter('train')
            test_metrics = self.metrics_getter('test')
            print("\n\U0001f321\ufe0f \u73af\u5883\u6307\u6807:")
            if train_metrics:
                self._print_env_metrics("\u8bad\u7ec3", train_metrics)
            else:
                print("  \u8bad\u7ec3: \u6682\u65e0\u6307\u6807\uff0c\u7b49\u5f85\u63d0\u4ea4 episode \u6570\u636e")
            if test_metrics:
                self._print_env_metrics("\u6d4b\u8bd5", test_metrics)
            elif test_result is not None:
                print("  \u6d4b\u8bd5: \u6682\u65e0\u6307\u6807\uff0c\u7b49\u5f85\u63d0\u4ea4 episode \u6570\u636e")
        # Q值统计
        print("\n💎 Q值统计:")
        print(f"  Q均值:          {q_mean:>10.3f}")
        print(f"  TD误差:         {td_error:>10.3f}")
        
        # 探索信息
        if exploration_noise > 0:
            print("\n🔍 探索信息:")
            print(f"  探索噪声:       {exploration_noise:>10.3f}")
        
        # 时间统计
        print("\n⏱️  时间统计:")
        print(f"  本轮耗时:       {self.format_time(epoch_time)}")
        print(f"  已用时间:       {self.format_time(elapsed_time)}")
        print(f"  预计剩余:       {self.format_time(estimated_remaining)}")
        print(f"  平均每轮:       {self.format_time(avg_epoch_time)}")
        
        # 异常警告
        warnings = []
        if actor_loss > self.thresholds['actor_loss_high']:
            warnings.append(f"Actor损失过高 ({actor_loss:.2f} > {self.thresholds['actor_loss_high']})")
        if critic_loss > self.thresholds['critic_loss_high']:
            warnings.append(f"Critic损失过高 ({critic_loss:.2f} > {self.thresholds['critic_loss_high']})")
        if critic_grad > self.thresholds['grad_norm_high']:
            warnings.append(f"Critic梯度过大 ({critic_grad:.2f} > {self.thresholds['grad_norm_high']})")
        
        if warnings:
            print("\n⚠️  警告:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("\n" + "=" * 80)

    def _print_env_metrics(self, label: str, metrics: Dict[str, Optional[float]]) -> None:
        energy = metrics.get('avg_energy')
        comfort = metrics.get('avg_comfort_mean')
        violations = metrics.get('avg_violations')
        parts = []
        if energy is not None:
            parts.append(f"平均能耗: {energy:.2f} kWh")
        if comfort is not None:
            parts.append(f"平均温差: {comfort:.2f} °C")
        if violations is not None:
            parts.append(f"平均越界: {violations:.2f}")
        if parts:
            print(f"  {label}: " + " | ".join(parts))
    
    def log_compact(
        self,
        epoch: int,
        train_result: Dict[str, Any],
        test_result: Optional[Dict[str, Any]] = None
    ):
        """
        紧凑格式的日志输出（适合频繁输出）
        
        参数:
        - epoch: 当前轮次
        - train_result: 训练结果字典
        - test_result: 测试结果字典（可选）
        """
        actor_loss = train_result.get('loss/actor', 0.0)
        critic_loss = train_result.get('loss/critic', 0.0)

        # 奖励值（尝试多个可能的键名）
        train_reward = train_result.get('train/reward',
                                       train_result.get('rew',
                                       train_result.get('rews', 0.0)))
        test_reward = 0.0
        if test_result:
            test_reward = test_result.get('test/reward',
                                         test_result.get('rew',
                                         test_result.get('rews', 0.0)))
        
        # 计算进度
        progress = epoch / self.total_epochs * 100
        elapsed = time.time() - self.start_time

        # 单行输出
        compact_line = (f"Epoch {epoch:>5}/{self.total_epochs} [{progress:>5.1f}%] | "
                       f"Actor: {actor_loss:>7.2f} | Critic: {critic_loss:>7.2f} | "
                       f"Train: {train_reward:>8.2f} | Test: {test_reward:>8.2f} | "
                       f"Time: {self.format_time(elapsed)}")
        if self.diffusion_steps:
            compact_line += f" | Diff: {self.diffusion_steps}步"
        print(compact_line)
    
    def log_summary(self, final_result: Dict[str, Any]):
        """
        训练结束后的总结日志
        
        参数:
        - final_result: 最终训练结果
        """
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("  🎉 训练完成总结")
        print("=" * 80)
        
        print(f"\n总训练时间: {self.format_time(total_time)}")
        print(f"总轮次: {self.total_epochs}")
        print(f"平均每轮: {self.format_time(total_time / self.total_epochs)}")
        
        if 'best_reward' in final_result:
            best_reward = final_result['best_reward']
            print(f"\n最佳测试奖励: {best_reward:.2f} (缩放后)")
            print(f"真实最佳奖励: {best_reward / self.reward_scale:.2f}")
        
        print("\n" + "=" * 80)


def create_epoch_logger(total_epochs: int, reward_scale: float = 1.0, verbose: bool = True):
    """
    创建epoch日志记录器（用于Tianshou trainer的回调）

    参数:
    - total_epochs: 总训练轮次
    - reward_scale: 奖励缩放系数
    - verbose: 是否详细输出（True=详细格式，False=紧凑格式）

    返回:
    - 日志回调函数
    """
    logger = TrainingLogger(total_epochs, reward_scale)

    def log_fn(epoch: int, env_step: int, gradient_step: int,
               train_result: Dict[str, Any], test_result: Optional[Dict[str, Any]] = None):
        """
        Tianshou trainer的日志回调函数

        参数:
        - epoch: 当前轮次
        - env_step: 环境步数
        - gradient_step: 梯度更新步数
        - train_result: 训练结果
        - test_result: 测试结果
        """
        if verbose:
            logger.log_epoch(epoch, train_result, test_result)
        else:
            logger.log_compact(epoch, train_result, test_result)

    return log_fn, logger


class EnhancedTensorboardLogger:
    """
    增强的TensorBoard日志记录器

    在原有TensorBoard记录的基础上，添加美化的终端输出
    继承并扩展TensorboardLogger的所有功能
    """

    def __init__(self, writer, total_epochs: int, reward_scale: float = 1.0,
                 log_interval: int = 1, verbose: bool = True, diffusion_steps: int = None,
                 update_log_interval: int = 1, step_per_epoch: int = 1,
                 metrics_getter: Optional[Callable[[str], Optional[Dict[str, float]]]] = None,
                 context_info: Optional[Dict[str, Any]] = None,
                 train_eval_collector=None,
                 train_eval_episodes: int = 1):
        """
        初始化增强日志记录器

        参数:
        - writer: TensorBoard SummaryWriter
        - total_epochs: 总训练轮次
        - reward_scale: 奖励缩放系数
        - log_interval: 日志输出间隔（每N个epoch输出一次）
        - verbose: 是否详细输出
        - diffusion_steps: 扩散模型步数（可选）
        """
        from tianshou.utils import TensorboardLogger

        self.tb_logger = TensorboardLogger(writer)  # 原始TensorBoard logger
        self.training_logger = TrainingLogger(
            total_epochs,
            reward_scale,
            diffusion_steps,
            metrics_getter=metrics_getter,
            context_info=context_info,
        )  # 终端日志格式化器
        self.log_interval = log_interval  # epoch日志输出间隔
        self.verbose = verbose  # 是否详细输出
        self.writer = writer  # TensorBoard writer
        self.update_log_interval = max(1, update_log_interval)  # 梯度日志抽样间隔
        self.step_per_epoch = max(1, step_per_epoch)

        # 初始化结果缓存
        self._last_train_result = {}
        self._last_test_result = {}
        self._last_update_result = {}
        self._last_train_eval_result = None
        self._current_epoch = 0
        self._last_output_epoch = -1  # 记录上次输出的epoch，避免重复输出
        self._has_update_data = False  # 标记是否有更新数据
        self.train_eval_collector = train_eval_collector
        self.train_eval_episodes = max(1, train_eval_episodes)

    def write(self, step_type: str, step: int, data: Dict[str, Any]):
        """
        写入日志（兼容Tianshou的Logger接口）

        参数:
        - step_type: 步骤类型（'train', 'test', 'update'等）
        - step: 步骤编号
        - data: 数据字典
        """
        # 写入TensorBoard
        self.tb_logger.write(step_type, step, data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn=None,
    ):
        """保存数据（兼容Tianshou的Logger接口）"""
        self.tb_logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self):
        """恢复数据（兼容Tianshou的Logger接口）"""
        return self.tb_logger.restore_data()

    def log_info_data(self, log_data: Dict[str, Any], step: int):
        """记录训练器整体信息（兼容 tianshou>=1.2 的 BaseLogger 接口）。"""
        if hasattr(self.tb_logger, "log_info_data"):
            self.tb_logger.log_info_data(log_data, step)
        else:
            self.tb_logger.write("info", step, log_data)

    def log_test_data(self, collect_result: Dict[str, Any], step: int):
        """
        记录测试数据（兼容Tianshou的Logger接口）

        参数:
        - collect_result: 收集结果字典
        - step: 当前步数
        """
        # 调用原始TensorBoard logger
        self.tb_logger.log_test_data(collect_result, step)

        # 保存测试结果
        self._last_test_result = collect_result
        # 根据 env_step 推算当前 epoch
        self._current_epoch = max(1, math.ceil(step / self.step_per_epoch))

        # 额外评估训练任务，获得与测试相同标准的训练奖励
        if self.train_eval_collector is not None:
            try:
                self._last_train_eval_result = self.train_eval_collector.collect(
                    n_episode=self.train_eval_episodes
                )
            except Exception as exc:
                print(f"警告: 训练集评估失败: {exc}")
                self._last_train_eval_result = None
        else:
            self._last_train_eval_result = None

        # 输出到终端（测试后输出）
        self._output_to_terminal()

    def log_train_data(self, collect_result: Dict[str, Any], step: int):
        """
        记录训练数据（兼容Tianshou的Logger接口）

        参数:
        - collect_result: 收集结果字典
        - step: 当前步数
        """
        # 调用原始TensorBoard logger
        self.tb_logger.log_train_data(collect_result, step)

        # 保存训练结果
        self._last_train_result = collect_result

        # 根据 env_step 推算当前 epoch
        self._current_epoch = max(1, math.ceil(step / self.step_per_epoch))

    def log_update_data(self, update_result: Dict[str, Any], step: int):
        """
        记录更新数据（兼容Tianshou的Logger接口）

        参数:
        - update_result: 更新结果字典
        - step: 当前步数（注意：这是gradient_step，不是epoch！）
        """
        # 抽样写入TensorBoard，减少频繁IO
        if step % self.update_log_interval == 0:
            self.tb_logger.log_update_data(update_result, step)

        # 保存更新结果（合并到训练结果中）
        self._last_train_result.update(update_result)

        # 注意：step是gradient_step，不是epoch
        # 我们需要从train_data或test_data中获取真正的epoch
        # 这里暂时保存step，但不更新_current_epoch
        # self._current_epoch = step  # ← 这是错误的！

        # 标记有更新数据（用于判断是否应该输出）
        self._has_update_data = True

    def _output_to_terminal(self):
        """输出到终端（内部方法）"""
        # 检查是否达到输出间隔
        if self._current_epoch % self.log_interval != 0:
            return

        # 检查是否有训练数据（避免在初始测试时输出）
        if not self._last_train_result:
            return

        # 检查是否有更新数据（只有在有更新数据时才输出）
        if not self._has_update_data:
            return

        # 检查是否已经输出过这个epoch（避免重复输出）
        if self._current_epoch == self._last_output_epoch:
            return

        # 记录本次输出的epoch
        self._last_output_epoch = self._current_epoch

        # 重置更新数据标记
        self._has_update_data = False

        # 合并训练和更新结果
        train_result = self._last_train_result.copy()
        test_result = self._last_test_result.copy() if self._last_test_result else None

        # 如果存在额外的训练集评估结果，则使用其奖励替换原训练奖励，保证可比性
        if self._last_train_eval_result:
            eval_reward = self._last_train_eval_result.get(
                'train/reward',
                self._last_train_eval_result.get('rew')
            )
            if eval_reward is not None:
                train_result['train/reward'] = eval_reward
            eval_len = self._last_train_eval_result.get('len')
            if eval_len is not None:
                train_result['train/len'] = eval_len

        # 输出到终端
        if self.verbose:
            self.training_logger.log_epoch(self._current_epoch, train_result, test_result)
        else:
            self.training_logger.log_compact(self._current_epoch, train_result, test_result)
