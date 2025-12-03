# ========================================
# SustainDC 环境配置
# ========================================
# SustainDC 接入 DROPT 时集中管理默认配置项。

# ========== SustainDC 核心默认参数 ==========
DEFAULT_LOCATION = "ny"  # 默认碳强度 / 天气区域 key
DEFAULT_MONTH_INDEX = 0  # SustainDC 使用的 0 基月份索引（0=1月，11=12月）
DEFAULT_DAYS_PER_EPISODE = 7  # 单次 rollout 默认持续天数
DEFAULT_TIMEZONE_SHIFT = 0  # 时间序列向前/向后平移量
DEFAULT_DATACENTER_CAPACITY_MW = 1.0  # PyE+ 数据中心负载缩放系数
DEFAULT_MAX_BAT_CAP_MW = 2.0  # 电池装机容量（MW）
DEFAULT_FLEXIBLE_LOAD_RATIO = 0.1  # 可移峰负载比例
DEFAULT_INDIVIDUAL_REWARD_WEIGHT = 0.8  # 单体奖励与全局奖励的混合系数
DEFAULT_REWARD_AGGREGATION = "mean"  # 可选值 {mean,sum}
DEFAULT_QUEUE_MAX_LEN = 1000  # SustainDC 内部队列长度（任务缓冲区）

# ========== SustainDC 数据文件默认值 ==========
# 下列文件名称与 dc-rl-main 相同，便于直接共用数据资产。
DEFAULT_CI_FILE = "NYIS_NG_&_avgCI.csv"
DEFAULT_WEATHER_FILE = "USA_NY_New.York-Kennedy.epw"
DEFAULT_WORKLOAD_FILE = "Alibaba_CPU_Data_Hourly_1.csv"
DEFAULT_DC_CONFIG_FILE = "dc_config.json"

# ========== SustainDC 封装层默认配置 ==========
# CLI / 脚本通过覆盖此字典中的键即可快速修改环境行为。
DEFAULT_WRAPPER_CONFIG = {
    "location": DEFAULT_LOCATION,
    "month": DEFAULT_MONTH_INDEX,
    "days_per_episode": DEFAULT_DAYS_PER_EPISODE,
    "timezone_shift": DEFAULT_TIMEZONE_SHIFT,
    "datacenter_capacity_mw": DEFAULT_DATACENTER_CAPACITY_MW,
    "max_bat_cap_Mw": DEFAULT_MAX_BAT_CAP_MW,
    "flexible_load": DEFAULT_FLEXIBLE_LOAD_RATIO,
    "individual_reward_weight": DEFAULT_INDIVIDUAL_REWARD_WEIGHT,
    "cintensity_file": DEFAULT_CI_FILE,
    "weather_file": DEFAULT_WEATHER_FILE,
    "workload_file": DEFAULT_WORKLOAD_FILE,
    "dc_config_file": DEFAULT_DC_CONFIG_FILE,
}

# 任何未在命令行显式指定的参数，均会 fallback 到该配置字典。

