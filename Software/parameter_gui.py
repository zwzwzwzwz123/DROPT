import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import subprocess
import threading
import queue
import webbrowser

class GUI:
    """
    图形用户界面类
    提供参数配置、训练控制和实时输出显示功能
    """
    def __init__(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("基于生成扩散模型的网络优化")  # 设置窗口标题
        self.root.geometry("1200x700")  # 设置窗口大小

        # 创建主笔记本（标签页容器）
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)

        # 创建参数配置标签页
        self.create_parameter_tab()
        # 创建输出显示标签页
        self.create_output_tab()

        # 创建输出队列，用于线程安全的GUI更新
        self.output_queue = queue.Queue()
        # 每100ms检查一次输出队列
        self.root.after(100, self.check_output_queue)

        # 初始化训练线程和TensorBoard进程
        self.training_thread = None  # 训练线程对象
        self.tensorboard_process = None  # TensorBoard子进程对象
        self.stop_training_flag = threading.Event()  # 用于控制训练停止的标志

    def create_parameter_tab(self):
        """
        创建参数配置和训练控制标签页
        包含左侧参数面板和右侧输出显示面板
        """
        main_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(main_frame, text='参数配置与训练')

        # 左侧：参数配置面板
        param_frame = ttk.Frame(main_frame)
        param_frame.grid(row=0, column=0, sticky="nsew")
        
        # 右侧：输出显示面板
        self.output_frame = ttk.Frame(main_frame)
        self.output_frame.grid(row=0, column=1, sticky="nsew")
        
        # 设置列权重，使两个面板均分空间
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 常用参数配置字典
        self.common_variables = {
            'exploration_noise': tk.DoubleVar(value=0.1),      # 探索噪声标准差
            'step_per_epoch': tk.IntVar(value=10),             # 每个训练轮次的步数
            'step_per_collect': tk.IntVar(value=10),           # 每次收集的步数
            'seed': tk.IntVar(value=1),                        # 随机种子
            'buffer_size': tk.DoubleVar(value=1e6),            # 经验回放缓冲区大小
            'epoch': tk.IntVar(value=1000000),                 # 总训练轮次
            'batch_size': tk.IntVar(value=512),                # 批次大小
            'actor_lr': tk.DoubleVar(value=1e-4),              # Actor网络学习率
            'critic_lr': tk.DoubleVar(value=1e-4),             # Critic网络学习率
            'n_timesteps': tk.IntVar(value=6),                 # 扩散模型时间步数（重要参数）
            'beta_schedule': tk.StringVar(value='vp'),         # 噪声调度策略（vp/linear/cosine）
            'device': tk.StringVar(value='cpu'),               # 计算设备（cpu/cuda:0）
            'bc_coef': tk.BooleanVar(value=False),             # 是否使用行为克隆（False=无专家数据模式）
            'prior_alpha': tk.DoubleVar(value=0.4),            # 优先经验回放alpha参数
            'prior_beta': tk.DoubleVar(value=0.4),             # 优先经验回放beta参数
        }

        # 高级参数配置字典
        self.advanced_variables = {
            'algorithm': tk.StringVar(value="diffusion_opt"),  # 算法名称
            'tau': tk.DoubleVar(value=0.005),                  # 目标网络软更新系数
            'wd': tk.DoubleVar(value=1e-4),                    # 权重衰减（L2正则化）
            'gamma': tk.DoubleVar(value=1),                    # 折扣因子
            'n_step': tk.IntVar(value=3),                      # N步TD学习
            'logdir': tk.StringVar(value='log'),               # 日志保存目录
            'training_num': tk.IntVar(value=1),                # 并行训练环境数量
            'test_num': tk.IntVar(value=1),                    # 并行测试环境数量
            'log_prefix': tk.StringVar(value='default'),       # 日志前缀
            'render': tk.DoubleVar(value=0.1),                 # 渲染参数
            'rew_norm': tk.IntVar(value=0),                    # 是否标准化奖励
            'resume_path': tk.StringVar(value=''),             # 恢复训练的模型路径
            'watch': tk.BooleanVar(value=False),               # 是否仅观察模式（不训练）
            'prioritized_replay': tk.BooleanVar(value=False),  # 是否使用优先经验回放
            'lr_decay': tk.BooleanVar(value=False),            # 是否使用学习率衰减
            'note': tk.StringVar(value=''),                    # 备注信息
        }

        # 创建参数配置笔记本（分为常用和高级两个标签页）
        param_notebook = ttk.Notebook(param_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True)

        # 常用参数标签页
        common_frame = ttk.Frame(param_notebook)
        param_notebook.add(common_frame, text='常用参数')

        # 高级参数标签页
        advanced_frame = ttk.Frame(param_notebook)
        param_notebook.add(advanced_frame, text='高级参数')

        # 在各标签页中创建参数输入控件
        self.create_param_widgets(common_frame, self.common_variables)
        self.create_param_widgets(advanced_frame, self.advanced_variables)

        # 创建按钮区域
        button_frame = ttk.Frame(param_frame)
        button_frame.pack(pady=10)

        # 添加控制按钮
        ttk.Button(button_frame, text="提交并开始训练", command=self.on_submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="停止训练", command=self.stop_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="启动TensorBoard", command=self.start_tensorboard).pack(side=tk.LEFT, padx=5)

    def create_param_widgets(self, parent, variables):
        """
        根据参数类型创建相应的输入控件
        - 布尔型：复选框
        - beta_schedule：下拉菜单
        - prior_alpha/beta：滑动条
        - 其他：文本输入框
        """
        for i, (key, var) in enumerate(variables.items()):
            # 创建参数标签（将下划线替换为空格，首字母大写）
            ttk.Label(parent, text=key.replace('_', ' ').title()).grid(column=0, row=i, sticky=tk.W)
            
            # 根据变量类型创建不同的输入控件
            if isinstance(var, tk.BooleanVar):
                # 布尔型参数：使用复选框
                ttk.Checkbutton(parent, variable=var).grid(column=1, row=i, sticky=tk.W)
            elif key == 'beta_schedule':
                # 噪声调度策略：使用下拉菜单
                ttk.Combobox(parent, textvariable=var, values=["vp", "linear", "cosine"]).grid(column=1, row=i, sticky=(tk.W, tk.E))
            elif key in ['prior_alpha', 'prior_beta']:
                # 优先级参数：使用滑动条（范围0-1）
                ttk.Scale(parent, from_=0, to=1, orient=tk.HORIZONTAL, variable=var).grid(column=1, row=i, sticky=(tk.W, tk.E))
                # 显示当前值
                ttk.Label(parent, textvariable=var).grid(column=2, row=i, sticky=tk.W)
            else:
                # 其他参数：使用文本输入框
                ttk.Entry(parent, textvariable=var).grid(column=1, row=i, sticky=(tk.W, tk.E))

    def create_output_tab(self):
        """
        创建输出显示标签页
        显示训练过程的实时输出信息
        """
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def on_submit(self):
        """
        提交按钮回调函数
        收集所有参数并启动训练
        """
        # 合并常用参数和高级参数
        params = {**self.common_variables, **self.advanced_variables}
        # 获取所有参数的当前值
        args = {key: var.get() for key, var in params.items()}
        # 启动训练
        self.start_training(args)
        return args

    def start_training(self, args):
        """
        在新线程中启动训练过程
        避免阻塞GUI主线程
        """
        # 检查是否已有训练进程在运行
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("训练", "训练已经在运行中。")
            return

        # 清除停止标志
        self.stop_training_flag.clear()

        def run_training():
            """训练线程执行的函数"""
            from main import main
            # 调用main函数，传入参数、输出更新函数和停止检查函数
            main(args, self.update_output, self.should_stop_training)

        # 创建并启动训练线程
        self.training_thread = threading.Thread(target=run_training)
        self.training_thread.start()

    def should_stop_training(self):
        """
        检查是否应该停止训练
        返回停止标志的状态
        """
        return self.stop_training_flag.is_set()

    def stop_training(self):
        """
        停止训练按钮的回调函数
        设置停止标志以终止训练循环
        """
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training_flag.set()
            self.update_output("训练已停止。")
        else:
            self.update_output("没有正在运行的训练进程。")

    def update_output(self, message):
        """
        更新输出显示区域
        将消息放入队列，确保线程安全
        """
        self.output_queue.put(message)
        self.root.update_idletasks()  # 强制更新GUI

    def check_output_queue(self):
        """
        定期检查输出队列并更新显示
        从队列中取出消息并添加到文本框
        """
        while not self.output_queue.empty():
            message = self.output_queue.get()
            # 在文本框末尾插入消息
            self.output_text.insert(tk.END, message + '\n')
            # 自动滚动到最新消息
            self.output_text.see(tk.END)
        # 100ms后再次检查队列
        self.root.after(100, self.check_output_queue)

    def start_tensorboard(self):
        """
        启动TensorBoard可视化服务
        在浏览器中查看训练过程和结果
        """
        # 从高级参数中获取日志目录
        logdir = self.advanced_variables['logdir'].get()
        
        # 检查TensorBoard进程是否已经在运行
        if not self.tensorboard_process or self.tensorboard_process.poll() is not None:
            def run_tensorboard():
                """在后台线程中启动TensorBoard"""
                try:
                    # 启动TensorBoard子进程，监听6006端口
                    self.tensorboard_process = subprocess.Popen(
                        ['tensorboard', '--logdir', logdir, '--port', '6006'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
                    # 在默认浏览器中打开TensorBoard页面
                    webbrowser.open('http://localhost:6006')
                except Exception as e:
                    self.update_output(f"启动TensorBoard失败: {str(e)}")

            # 创建守护线程启动TensorBoard
            threading.Thread(target=run_tensorboard, daemon=True).start()
        else:
            # 如果已经在运行，直接打开浏览器
            webbrowser.open('http://localhost:6006')
            messagebox.showinfo("TensorBoard", "TensorBoard已经在运行中。")

def create_gui():
    """
    创建并返回GUI对象及其关键方法
    供主程序调用
    """
    gui = GUI()
    return gui.root, gui.start_training, gui.update_output, gui.stop_training

if __name__ == '__main__':
    # 独立运行时的测试代码
    root, on_submit, update_output, stop_training = create_gui()
    root.mainloop()
    args = on_submit()
    print(json.dumps(args, indent=2))
