# ========================================
# 通用专家控制器基线评估脚本
# ========================================
# 可在 BEAR 建筑环境或 SustainDC/数据中心环境中运行 PID/MPC/规则控制器
# 通过命令行或下方常量选择要评估的环境与控制策略

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np

# BEAR 默认配置
from env.building_config import (
    DEFAULT_BUILDING_TYPE,
    DEFAULT_WEATHER_TYPE,
    DEFAULT_LOCATION,
    DEFAULT_TARGET_TEMP as DEFAULT_BUILDING_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE as DEFAULT_BUILDING_TEMP_TOLERANCE,
    DEFAULT_ENERGY_WEIGHT as DEFAULT_BUILDING_ENERGY_WEIGHT,
    DEFAULT_TEMP_WEIGHT as DEFAULT_BUILDING_TEMP_WEIGHT,
    DEFAULT_MAX_POWER as DEFAULT_BUILDING_MAX_POWER,
    DEFAULT_TIME_RESOLUTION as DEFAULT_BUILDING_TIME_RESOLUTION,
    DEFAULT_REWARD_SCALE as DEFAULT_BUILDING_REWARD_SCALE,
    DEFAULT_VIOLATION_PENALTY as DEFAULT_BUILDING_VIOLATION_PENALTY,
    DEFAULT_MPC_PLANNING_STEPS as DEFAULT_BUILDING_MPC_STEPS,
    DEFAULT_PID_KP,
    DEFAULT_PID_KI,
    DEFAULT_PID_KD,
    DEFAULT_PID_INTEGRAL_LIMIT,
)

# SustainDC 默认配置
from env.sustaindc_config import (
    DEFAULT_WRAPPER_CONFIG as SDC_DEFAULT_WRAPPER,
    DEFAULT_REWARD_AGGREGATION as SDC_DEFAULT_REWARD_AGG,
    DEFAULT_LOCATION as SDC_DEFAULT_LOCATION,
    DEFAULT_MONTH_INDEX as SDC_DEFAULT_MONTH_INDEX,
    DEFAULT_DAYS_PER_EPISODE as SDC_DEFAULT_DAYS,
    DEFAULT_TIMEZONE_SHIFT as SDC_DEFAULT_TZ_SHIFT,
    DEFAULT_DATACENTER_CAPACITY_MW as SDC_DEFAULT_CAPACITY,
    DEFAULT_MAX_BAT_CAP_MW as SDC_DEFAULT_BAT_CAP,
    DEFAULT_FLEXIBLE_LOAD_RATIO as SDC_DEFAULT_FLEX_RATIO,
    DEFAULT_INDIVIDUAL_REWARD_WEIGHT as SDC_DEFAULT_INDIVIDUAL_WEIGHT,
)

# ========== 顶部环境开关 ==========
# 在此填入 'bear' 或 'sustaindc'，作为脚本默认运行环境
DEFAULT_ENV_SELECTION = "sustaindc"


# -----------------------------------------------------------------------------#
# SustainDC / 数据中心环境逻辑
# -----------------------------------------------------------------------------#
def _extract_ctrl_inputs(info: Dict[str, float], state: np.ndarray) -> Dict[str, float]:
    """从 info/state 中提取 PID/MPC 所需的温度和负载输入"""
    if info:
        return {
            "T_in": float(info.get("T_in", state[0])),
            "T_out": float(info.get("T_out", state[1])),
            "H_in": float(info.get("H_in", state[2])),
            "IT_load": float(info.get("IT_load", state[3])),
        }
    return {
        "T_in": float(state[0]),
        "T_out": float(state[1]),
        "H_in": float(state[2]),
        "IT_load": float(state[3]),
    }


def _run_datacenter_episode(
    env,
    controller,
    seed: int | None = None,
) -> Dict[str, float]:
    """执行单条数据中心轨迹"""
    obs, info = env.reset(seed=seed)
    ctrl_inputs = _extract_ctrl_inputs(info, obs)
    done = False
    ep_reward = 0.0
    episode_energy = 0.0
    violations = 0.0

    while not done:
        action = controller.get_action(**ctrl_inputs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        episode_energy = float(info.get("episode_energy", episode_energy))
        violations = float(info.get("episode_violations", violations))
        done = terminated or truncated
        if not done:
            ctrl_inputs = _extract_ctrl_inputs(info, obs)

    return {
        "reward": ep_reward,
        "energy": episode_energy,
        "violations": violations,
    }


def evaluate_datacenter(args: argparse.Namespace) -> None:
    """构建数据中心环境并执行多条回合"""
    from env.datacenter_env import DataCenterEnv
    from env.expert_controller import ExpertController

    env = DataCenterEnv(
        num_crac_units=args.num_crac,
        target_temp=args.target_temp,
        temp_tolerance=args.temp_tolerance,
        time_step=args.time_step,
        episode_length=args.episode_length,
        energy_weight=args.energy_weight,
        temp_weight=args.temp_weight,
        violation_penalty=args.violation_penalty,
        use_real_weather=args.use_real_weather,
        weather_file=args.weather_file,
        workload_file=args.workload_file,
        expert_type="pid",
    )
    controller = ExpertController(
        num_crac=args.num_crac,
        target_temp=args.target_temp,
        controller_type=args.controller,
    )

    rewards: List[float] = []
    energies: List[float] = []
    violations: List[float] = []

    reset_fn = getattr(controller, "reset", None)

    for idx in range(args.episodes):
        if callable(reset_fn):
            reset_fn()
        seed = args.seed + idx if args.seed is not None else None
        result = _run_datacenter_episode(env, controller, seed=seed)
        rewards.append(result["reward"])
        energies.append(result["energy"])
        violations.append(result["violations"])
        print(
            f"[Episode {idx + 1:02d}] reward={result['reward']:.2f}, "
            f"energy={result['energy']:.2f} kWh, violations={result['violations']:.0f}"
        )

    _print_summary(
        "SustainDC/DataCenter",
        args.controller,
        rewards,
        energies=energies,
        violations=violations,
    )


# -----------------------------------------------------------------------------#
# BEAR 建筑环境逻辑
# -----------------------------------------------------------------------------#
def _run_building_episode(
    env,
    controller,
    seed: int | None = None,
) -> Dict[str, float]:
    """执行单条 BEAR 轨迹"""
    obs, _ = env.reset(seed=seed)
    done = False
    ep_reward = 0.0
    episode_energy = 0.0
    comfort_sum = 0.0
    violation_sum = 0.0
    step_count = 0

    while not done:
        action = controller.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        if "episode_hvac_energy_kwh" in info:
            episode_energy = float(info["episode_hvac_energy_kwh"])
        else:
            episode_energy += float(info.get("hvac_energy_kwh", 0.0))
        comfort_sum += float(info.get("comfort_mean_abs_dev", 0.0))
        violation_sum += float(info.get("comfort_violations", 0.0))
        step_count += 1
        done = terminated or truncated

    avg_comfort = comfort_sum / step_count if step_count else 0.0
    avg_violations = violation_sum / step_count if step_count else 0.0

    return {
        "reward": ep_reward,
        "energy": episode_energy,
        "avg_comfort": avg_comfort,
        "avg_violations": avg_violations,
    }


def evaluate_building(args: argparse.Namespace) -> None:
    """构建 BEAR 环境并执行多条回合"""
    from env.building_env_wrapper import BearEnvWrapper
    from env.building_expert_controller import create_expert_controller

    env = BearEnvWrapper(
        building_type=args.building_type,
        weather_type=args.building_weather,
        location=args.building_location,
        target_temp=args.building_target_temp,
        temp_tolerance=args.building_temp_tolerance,
        max_power=args.building_max_power,
        time_resolution=args.building_time_resolution,
        energy_weight=args.building_energy_weight,
        temp_weight=args.building_temp_weight,
        episode_length=args.building_episode_length,
        reward_scale=args.building_reward_scale,
        add_violation_penalty=args.building_add_violation_penalty,
        violation_penalty=args.building_violation_penalty,
    )
    controller_kwargs = {}
    controller_name = args.controller.lower()
    if controller_name == "mpc":
        controller_kwargs["planning_steps"] = args.building_mpc_planning_steps
    elif controller_name == "pid":
        controller_kwargs.update(
            {
                "kp": args.pid_kp,
                "ki": args.pid_ki,
                "kd": args.pid_kd,
                "integral_limit": args.pid_integral_limit,
                "deadband": args.pid_deadband,
            }
        )
    controller = create_expert_controller(args.controller, env, **controller_kwargs)

    rewards: List[float] = []
    energies: List[float] = []
    comforts: List[float] = []
    violations: List[float] = []

    reset_fn = getattr(controller, "reset", None)

    for idx in range(args.episodes):
        if callable(reset_fn):
            reset_fn()
        seed = args.seed + idx if args.seed is not None else None
        result = _run_building_episode(env, controller, seed=seed)
        rewards.append(result["reward"])
        energies.append(result["energy"])
        comforts.append(result["avg_comfort"])
        violations.append(result["avg_violations"])
        print(
            f"[Episode {idx + 1:02d}] 奖励={result['reward']:.2f}, "
            f"HVAC能耗={result['energy']:.2f} kWh, "
            f"平均温差={result['avg_comfort']:.3f} C, "
            f"平均越界={result['avg_violations']:.3f}"
        )

    _print_summary(
        "BEAR Building",
        args.controller,
        rewards,
        energies=energies,
        comforts=comforts,
        violations=violations,
    )


# -----------------------------------------------------------------------------#
# SustainDC 环境逻辑
# -----------------------------------------------------------------------------#


def _safe_scalar(value) -> Optional[float]:
    """将嵌套结构转换为标量浮点数"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _safe_scalar(value[0]) if value else None
    if isinstance(value, dict):
        for _, v in value.items():
            scalar = _safe_scalar(v)
            if scalar is not None:
                return scalar
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, np.ndarray):
            return _safe_scalar(value.flatten()[0]) if value.size else None
    return None


class SustainDCRuleController:
    """
    SustainDC 简易规则控制器

    - 动作 0: agent_ls（任务调度） -> 负载高时处理/低时延迟
    - 动作 1: agent_dc（HVAC 设定） -> 根据室外温度调整
    - 动作 2: agent_bat（电池） -> 根据碳强度充放电
    """

    def __init__(
        self,
        action_dim: int,
        target_temp: float = 24.0,
        temp_band: float = 2.0,
        workload_low: float = 0.3,
        workload_high: float = 0.7,
        ci_low: float = 0.35,
        ci_high: float = 0.65,
    ) -> None:
        self.action_dim = action_dim
        self.target_temp = target_temp
        self.temp_band = temp_band
        self.workload_low = workload_low
        self.workload_high = workload_high
        self.workload_mid = (workload_low + workload_high) / 2.0
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.reset()

    def reset(self) -> None:
        self._last_workload = 0.5
        self._last_temp = self.target_temp
        self._last_ci = self.ci_low

    def get_action(self, obs: np.ndarray, info: Dict[str, float]) -> np.ndarray:
        workload = _safe_scalar(info.get("workload"))
        if workload is None:
            workload = self._last_workload
        else:
            self._last_workload = workload

        weather = info.get("weather", {})
        outdoor_temp = None
        if isinstance(weather, dict):
            for key in ("dry_bulb", "temperature", "outdoor_temp", "T_out"):
                if key in weather:
                    outdoor_temp = _safe_scalar(weather[key])
                    break
        if outdoor_temp is None:
            outdoor_temp = self._last_temp
        else:
            self._last_temp = outdoor_temp

        ci_value = _safe_scalar(info.get("carbon_intensity"))
        if ci_value is None:
            ci_value = self._last_ci
        else:
            self._last_ci = ci_value

        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_dim >= 1:
            action[0] = self._load_shedding_action(workload, ci_value)
        if self.action_dim >= 2:
            action[1] = self._hvac_action(outdoor_temp)
        if self.action_dim >= 3:
            action[2] = self._battery_action(workload, ci_value)
        return action

    def _load_shedding_action(self, workload: float, ci_value: float) -> float:
        if workload > self.workload_high or ci_value > self.ci_high:
            return 0.8  # 优先处理任务，避免过载
        if workload < self.workload_low and ci_value < self.ci_high:
            return -0.6  # 延迟任务以利用低负载时段
        return 0.0

    def _hvac_action(self, outdoor_temp: float) -> float:
        if outdoor_temp > self.target_temp + self.temp_band:
            return -0.8  # 增强制冷
        if outdoor_temp < self.target_temp - self.temp_band:
            return 0.7  # 放松制冷
        return 0.0

    def _battery_action(self, workload: float, ci_value: float) -> float:
        if ci_value < self.ci_low:
            return -0.7  # 碳强度低 -> 充电
        if ci_value > self.ci_high and workload > self.workload_mid:
            return 0.7  # 碳强度高且负载大 -> 放电辅助
        return 0.0


def evaluate_sustaindc(args: argparse.Namespace) -> None:
    """构建 SustainDC 环境并执行多条回合"""
    try:
        from env.sustaindc_env_wrapper import make_sustaindc_env
    except ImportError as exc:  # pragma: no cover - 仅在依赖缺失时触发
        raise RuntimeError("SustainDC 依赖未安装，无法运行该基线。") from exc

    env_cfg = dict(SDC_DEFAULT_WRAPPER)
    env_cfg.update(
        {
            "location": args.sdc_location,
            "month": args.sdc_month,
            "days_per_episode": args.sdc_days,
            "timezone_shift": args.sdc_timezone_shift,
            "datacenter_capacity_mw": args.sdc_dc_capacity,
            "max_bat_cap_Mw": args.sdc_max_bat_cap,
            "flexible_load": args.sdc_flexible_load,
            "individual_reward_weight": args.sdc_individual_reward_weight,
            "cintensity_file": args.sdc_cintensity_file,
            "weather_file": args.sdc_weather_file,
            "workload_file": args.sdc_workload_file,
            "dc_config_file": args.sdc_dc_config_file,
        }
    )

    env, _, _ = make_sustaindc_env(
        training_num=1,
        test_num=1,
        vector_env_type="dummy",
        env_config=env_cfg,
        reward_aggregation=args.sdc_reward_aggregation,
        action_threshold=args.sdc_action_threshold,
    )

    controller = SustainDCRuleController(
        action_dim=env.action_space.shape[0],
        target_temp=args.sdc_target_temp,
        temp_band=args.sdc_temp_band,
        workload_low=args.sdc_workload_low,
        workload_high=args.sdc_workload_high,
        ci_low=args.sdc_ci_low,
        ci_high=args.sdc_ci_high,
    )

    rewards: List[float] = []
    ci_means: List[float] = []
    workload_means: List[float] = []

    for idx in range(args.episodes):
        obs, info = env.reset(seed=(args.seed + idx) if args.seed is not None else None)
        controller.reset()
        done = False
        ep_reward = 0.0
        ci_samples: List[float] = []
        workload_samples: List[float] = []

        while not done:
            action = controller.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            ci_value = _safe_scalar(info.get("carbon_intensity"))
            if ci_value is not None:
                ci_samples.append(ci_value)
            workload_value = _safe_scalar(info.get("workload"))
            if workload_value is not None:
                workload_samples.append(workload_value)
            done = terminated or truncated

        rewards.append(ep_reward)
        if ci_samples:
            ci_means.append(float(np.mean(ci_samples)))
        if workload_samples:
            workload_means.append(float(np.mean(workload_samples)))

        print(
            f"[Episode {idx + 1:02d}] reward={ep_reward:.2f}, "
            f"avg CI={np.mean(ci_samples) if ci_samples else float('nan'):.3f}, "
            f"avg workload={np.mean(workload_samples) if workload_samples else float('nan'):.3f}"
        )

    extras = {}
    if ci_means:
        extras["碳强度"] = ci_means
    if workload_means:
        extras["工作负载"] = workload_means
    _print_summary("SustainDC", args.controller, rewards, extras=extras)


# -----------------------------------------------------------------------------#
# 公共工具
# -----------------------------------------------------------------------------#
def _print_summary(
    env_name: str,
    controller_name: str,
    rewards: List[float],
    energies: Optional[List[float]] = None,
    comforts: Optional[List[float]] = None,
    violations: Optional[List[float]] = None,
    extras: Optional[Dict[str, List[float]]] = None,
) -> None:
    """打印平均表现（根据可用指标自适应）"""
    def _format_stats(values: List[float], unit: str = "") -> str:
        arr = np.asarray(values, dtype=np.float32)
        stats = f"{arr.mean():.2f} ± {arr.std():.2f}"
        stats += f" (min={arr.min():.2f}, max={arr.max():.2f})"
        if unit:
            stats += f" {unit}"
        return stats

    print("\n========== 平均表现 ==========")
    print(f"环境: {env_name}")
    print(f"控制器: {controller_name}")
    print(f"平均奖励: {_format_stats(rewards)}")

    if energies:
        print(f"平均能耗: {_format_stats(energies, 'kWh')}")
    if comforts:
        print(f"平均温度偏差: {_format_stats(comforts, 'C')}")
    if violations:
        print(f"平均越界次数: {_format_stats(violations)}")
    if extras:
        for key, values in extras.items():
            if values:
                print(f"平均{key}: {_format_stats(values)}")


# -----------------------------------------------------------------------------#
# 参数定义与入口
# -----------------------------------------------------------------------------#
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行专家控制器基线用于 RL 对比")
    parser.add_argument(
        "--env-target",
        choices=["bear", "building", "sustaindc", "datacenter"],
        default=DEFAULT_ENV_SELECTION,
        help="选择评估的环境类型，亦可修改文件顶部 DEFAULT_ENV_SELECTION",
    )
    parser.add_argument("--controller", type=str, default="pid", help="控制器类型（按环境自动匹配）")
    parser.add_argument("--episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（每回合递增）")

    # -------------------- 数据中心 / SustainDC 参数 --------------------
    parser.add_argument("--num-crac", type=int, default=4, help="数据中心 CRAC 数量")
    parser.add_argument("--target-temp", type=float, default=24.0, help="数据中心目标温度 (°C)")
    parser.add_argument("--temp-tolerance", type=float, default=2.0, help="温度容差 (°C)")
    parser.add_argument("--time-step", type=float, default=5.0, help="仿真步长 (分钟)")
    parser.add_argument("--episode-length", type=int, default=288, help="回合步数")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="奖励能耗惩罚权重")
    parser.add_argument("--temp-weight", type=float, default=10.0, help="奖励温度偏差权重")
    parser.add_argument("--violation-penalty", type=float, default=100.0, help="温度越界惩罚")
    parser.add_argument("--use-real-weather", action="store_true", help="是否使用真实气象 CSV")
    parser.add_argument("--weather-file", type=str, default=None, help="气象 CSV 路径")
    parser.add_argument("--workload-file", type=str, default=None, help="负载 CSV 路径")

    # -------------------- SustainDC 参数 --------------------
    parser.add_argument("--sdc-location", type=str, default=SDC_DEFAULT_LOCATION, help="SustainDC 区域标识")
    parser.add_argument("--sdc-month", type=int, default=SDC_DEFAULT_MONTH_INDEX, help="0 基月份索引")
    parser.add_argument("--sdc-days", type=int, default=SDC_DEFAULT_DAYS, help="单个 episode 天数")
    parser.add_argument("--sdc-timezone-shift", type=int, default=SDC_DEFAULT_TZ_SHIFT, help="时间偏移量")
    parser.add_argument("--sdc-dc-capacity", type=float, default=SDC_DEFAULT_CAPACITY, help="数据中心容量缩放系数")
    parser.add_argument("--sdc-max-bat-cap", type=float, default=SDC_DEFAULT_BAT_CAP, help="电池容量 (MW)")
    parser.add_argument("--sdc-flexible-load", type=float, default=SDC_DEFAULT_FLEX_RATIO, help="可移峰负载比例")
    parser.add_argument(
        "--sdc-individual-reward-weight",
        type=float,
        default=SDC_DEFAULT_INDIVIDUAL_WEIGHT,
        help="单体/全局奖励混合权重",
    )
    parser.add_argument(
        "--sdc-reward-aggregation",
        choices=["mean", "sum"],
        default=SDC_DEFAULT_REWARD_AGG,
        help="SustainDC 奖励聚合方式",
    )
    parser.add_argument("--sdc-cintensity-file", type=str, default=SDC_DEFAULT_WRAPPER["cintensity_file"])
    parser.add_argument("--sdc-weather-file", type=str, default=SDC_DEFAULT_WRAPPER["weather_file"])
    parser.add_argument("--sdc-workload-file", type=str, default=SDC_DEFAULT_WRAPPER["workload_file"])
    parser.add_argument("--sdc-dc-config-file", type=str, default=SDC_DEFAULT_WRAPPER["dc_config_file"])
    parser.add_argument("--sdc-action-threshold", type=float, default=0.33, help="连续动作映射阈值")
    parser.add_argument("--sdc-target-temp", type=float, default=24.0, help="DC 控制目标温度 (°C)")
    parser.add_argument("--sdc-temp-band", type=float, default=2.0, help="温度死区 (°C)")
    parser.add_argument("--sdc-workload-low", type=float, default=0.3, help="低负载阈值 (0-1 归一化)")
    parser.add_argument("--sdc-workload-high", type=float, default=0.7, help="高负载阈值")
    parser.add_argument("--sdc-ci-low", type=float, default=0.35, help="碳强度低阈值 (0-1)")
    parser.add_argument("--sdc-ci-high", type=float, default=0.65, help="碳强度高阈值")

    # -------------------- BEAR 建筑参数 --------------------
    parser.add_argument("--building-type", type=str, default=DEFAULT_BUILDING_TYPE, help="BEAR 建筑类型")
    parser.add_argument("--building-weather", type=str, default=DEFAULT_WEATHER_TYPE, help="气候类型")
    parser.add_argument("--building-location", type=str, default=DEFAULT_LOCATION, help="地理位置 / 数据集关键字")
    parser.add_argument(
        "--building-target-temp",
        type=float,
        default=DEFAULT_BUILDING_TARGET_TEMP,
        help="建筑目标温度 (°C)",
    )
    parser.add_argument(
        "--building-temp-tolerance",
        type=float,
        default=DEFAULT_BUILDING_TEMP_TOLERANCE,
        help="建筑温度容差 (°C)",
    )
    parser.add_argument(
        "--building-energy-weight",
        type=float,
        default=DEFAULT_BUILDING_ENERGY_WEIGHT,
        help="建筑奖励能耗权重",
    )
    parser.add_argument(
        "--building-temp-weight",
        type=float,
        default=DEFAULT_BUILDING_TEMP_WEIGHT,
        help="建筑奖励舒适度权重",
    )
    parser.add_argument(
        "--building-max-power",
        type=float,
        default=DEFAULT_BUILDING_MAX_POWER,
        help="HVAC 最大功率 (W)",
    )
    parser.add_argument(
        "--building-time-resolution",
        type=int,
        default=DEFAULT_BUILDING_TIME_RESOLUTION,
        help="时间分辨率 (秒)",
    )
    parser.add_argument(
        "--building-episode-length",
        type=int,
        default=None,
        help="建筑回合步数（默认 None 表示整段数据）",
    )
    parser.add_argument(
        "--building-reward-scale",
        type=float,
        default=DEFAULT_BUILDING_REWARD_SCALE,
        help="建筑奖励缩放系数",
    )
    parser.add_argument(
        "--building-add-violation-penalty",
        dest="building_add_violation_penalty",
        action="store_true",
        default=True,
        help="启用温度越界惩罚（默认开启）",
    )
    parser.add_argument(
        "--building-no-violation-penalty",
        dest="building_add_violation_penalty",
        action="store_false",
        help="关闭温度越界惩罚",
    )
    parser.add_argument(
        "--building-violation-penalty",
        type=float,
        default=DEFAULT_BUILDING_VIOLATION_PENALTY,
        help="越界惩罚系数",
    )
    parser.add_argument(
        "--building-mpc-planning-steps",
        type=int,
        default=DEFAULT_BUILDING_MPC_STEPS,
        help=f"MPC 控制器规划步数 (默认 {DEFAULT_BUILDING_MPC_STEPS})",
    )
    parser.add_argument(
        "--pid-kp",
        type=float,
        default=DEFAULT_PID_KP,
        help=f"PID 比例系数 (默认 {DEFAULT_PID_KP})",
    )
    parser.add_argument(
        "--pid-ki",
        type=float,
        default=DEFAULT_PID_KI,
        help=f"PID 积分系数 (默认 {DEFAULT_PID_KI})",
    )
    parser.add_argument(
        "--pid-kd",
        type=float,
        default=DEFAULT_PID_KD,
        help=f"PID 微分系数 (默认 {DEFAULT_PID_KD})",
    )
    parser.add_argument(
        "--pid-integral-limit",
        type=float,
        default=DEFAULT_PID_INTEGRAL_LIMIT,
        help=f"PID 积分项限制 (默认 {DEFAULT_PID_INTEGRAL_LIMIT})",
    )
    parser.add_argument(
        "--pid-deadband",
        type=float,
        default=None,
        help="PID 死区范围 (默认等于温度容差)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env_choice = args.env_target.lower()

    if env_choice in {"bear", "building"}:
        valid_ctrl = {"mpc", "pid", "rule", "rulebased", "bangbang"}
        if args.controller.lower() not in valid_ctrl:
            raise ValueError(
                f"BEAR 环境仅支持 {sorted(valid_ctrl)}, "
                f"收到 '{args.controller}'."
            )
        evaluate_building(args)
    elif env_choice in {"datacenter"}:
        valid_ctrl = {"pid", "mpc", "rule_based", "rule"}
        if args.controller.lower() not in valid_ctrl:
            raise ValueError(
                f"数据中心环境仅支持 {sorted(valid_ctrl)}, "
                f"收到 '{args.controller}'."
            )
        evaluate_datacenter(args)
    else:
        valid_ctrl = {"rule", "sustaindc_rule"}
        if args.controller.lower() not in valid_ctrl:
            raise ValueError(
                f"SustainDC 基线暂时仅支持 {sorted(valid_ctrl)}, "
                f"收到 '{args.controller}'."
            )
        evaluate_sustaindc(args)


if __name__ == "__main__":
    main()
