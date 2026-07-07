"""
部署期引导（deployment-time guidance）工具包。

该包完全独立于训练脚本，只在部署/评估阶段使用：
- surrogate.py: 从观测转移数据学到的温度预测小模型 + 舒适/能耗引导函数
- policy_io.py: 复现网络结构并加载已训练好的策略 checkpoint

设计原则（重要）：
- 温度预测模型只使用真实控制器可观测的量：state、action、下一步实测区温。
- 绝不访问 BEAR 仿真器内部（A_d/B_d、真实转移、隐藏态），避免"上帝视角"。
"""
