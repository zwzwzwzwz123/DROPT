# ========================================
# SustainDC Environment Configuration
# ========================================
# Central place to store the default knobs when wiring SustainDC into DROPT.

DEFAULT_LOCATION = "ny"
DEFAULT_MONTH_INDEX = 0  # 0-based month used by SustainDC (0=Jan, 11=Dec)
DEFAULT_DAYS_PER_EPISODE = 7
DEFAULT_TIMEZONE_SHIFT = 0
DEFAULT_DATACENTER_CAPACITY_MW = 1.0
DEFAULT_MAX_BAT_CAP_MW = 2.0
DEFAULT_FLEXIBLE_LOAD_RATIO = 0.1
DEFAULT_INDIVIDUAL_REWARD_WEIGHT = 0.8
DEFAULT_REWARD_AGGREGATION = "mean"  # {mean,sum}
DEFAULT_QUEUE_MAX_LEN = 1000

# SustainDC ships with several CSV / EPW files. The defaults below match the
# values used inside dc-rl-main.
DEFAULT_CI_FILE = "NYIS_NG_&_avgCI.csv"
DEFAULT_WEATHER_FILE = "USA_NY_New.York-Kennedy.epw"
DEFAULT_WORKLOAD_FILE = "Alibaba_CPU_Data_Hourly_1.csv"
DEFAULT_DC_CONFIG_FILE = "dc_config.json"

# Wrapper-wide defaults exposed to CLI / higher level scripts.
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

