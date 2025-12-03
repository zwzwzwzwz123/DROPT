"""
Gym wrapper around the SustainDC multi-agent environment from dc-rl-main.

This adapter flattens the multi-agent observations/rewards into a single
continuous control interface, so that DROPT's Diffusion-based policy can be
trained without touching the upstream codebase.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .sustaindc_config import (
    DEFAULT_REWARD_AGGREGATION,
    DEFAULT_WRAPPER_CONFIG,
)

DC_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dc-rl-main"))
_PATH_ADDED = False
if DC_RL_ROOT not in sys.path:
    sys.path.insert(0, DC_RL_ROOT)
    _PATH_ADDED = True

# SustainDC relies on OpenMP internally (PyTorch + EnergyPlus). Without this env
# var multiple OpenMP runtimes may crash the interpreter on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from sustaindc_env import SustainDC  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "Unable to import SustainDC. Make sure dc-rl-main is present inside the "
        "repository root and all dependencies from dc-rl-main/requirements.txt "
        "are installed."
    ) from exc
finally:
    if _PATH_ADDED and sys.path and sys.path[0] == DC_RL_ROOT:
        sys.path.pop(0)


def _concat_space_bounds(spaces_in_order: Iterable[spaces.Box]) -> spaces.Box:
    lows: List[np.ndarray] = []
    highs: List[np.ndarray] = []
    for space in spaces_in_order:
        if not isinstance(space, spaces.Box):
            raise TypeError(
                f"SustainDC wrapper expects Box observation spaces, got {type(space)}"
            )
        lows.append(space.low.astype(np.float32, copy=False))
        highs.append(space.high.astype(np.float32, copy=False))
    low = np.concatenate(lows, axis=0).astype(np.float32, copy=False)
    high = np.concatenate(highs, axis=0).astype(np.float32, copy=False)
    return spaces.Box(low=low, high=high, dtype=np.float32)


class SustainDCEnvWrapper(gym.Env[np.ndarray, np.ndarray]):
    """
    Flatten SustainDC's three-agent interface into a single Gymnasium env.

    - Observations: concatenation of agent_ls, agent_dc, agent_bat states.
    - Action space: continuous vector in [-1, 1]^3 which is mapped to the
      discrete decisions expected by SustainDC's agents.
    - Reward: aggregate (mean or sum) of the three agent rewards.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        reward_aggregation: str = DEFAULT_REWARD_AGGREGATION,
        action_threshold: float = 0.33,
    ) -> None:
        super().__init__()
        cfg = dict(DEFAULT_WRAPPER_CONFIG)
        if env_config:
            cfg.update(env_config)
        self._inner_env = SustainDC(cfg)
        self._agent_order = list(self._inner_env.agents)
        self._agent_obs_spaces = {
            name: space for name, space in zip(self._inner_env.agents, self._inner_env.observation_space)
        }
        assert self._agent_order, "SustainDC is expected to expose at least one agent."

        # Build a flattened observation space.
        obs_spaces_in_order = [self._agent_obs_spaces[name] for name in self._agent_order]
        self.observation_space = _concat_space_bounds(obs_spaces_in_order)

        # Continuous proxy actions that will be discretised before calling SustainDC.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self._agent_order),), dtype=np.float32
        )
        self._action_threshold = float(action_threshold)
        self._reward_mode = reward_aggregation.lower()
        self._last_obs: Optional[np.ndarray] = None

        # Mapping between continuous action buckets and discrete SustainDC actions.
        self._action_templates: Dict[str, tuple[int, int, int]] = {
            "agent_ls": (0, 1, 2),  # defer / idle / process
            "agent_dc": (0, 1, 2),  # decrease / hold / increase setpoint
            "agent_bat": (0, 2, 1),  # charge / idle / discharge
        }

    # --------------------------------------------------------------------- Utils
    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        buffers: List[np.ndarray] = []
        for agent in self._agent_order:
            arr = obs_dict.get(agent)
            if arr is None:
                continue
            buffers.append(np.asarray(arr, dtype=np.float32))
        if not buffers:
            # SustainDC returns empty obs when truncated. Reuse previous obs so
            # the trainer receives a correctly shaped tensor.
            if self._last_obs is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            return self._last_obs.copy()
        flat = np.concatenate(buffers, axis=0).astype(np.float32, copy=False)
        self._last_obs = flat
        return flat

    def _aggregate_reward(self, reward_dict: Dict[str, float]) -> float:
        rewards = [float(reward_dict.get(agent, 0.0)) for agent in self._agent_order]
        if not rewards:
            return 0.0
        if self._reward_mode == "sum":
            return float(np.sum(rewards))
        # default -> mean for better numerical stability
        return float(np.mean(rewards))

    def _continuous_to_discrete(self, value: float, agent: str) -> int:
        low, mid, high = self._action_templates.get(agent, (0, 1, 2))
        if value <= -self._action_threshold:
            return low
        if value >= self._action_threshold:
            return high
        return mid

    def _map_action(self, action: np.ndarray) -> Dict[str, int]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self._agent_order):
            raise ValueError(
                f"SustainDC wrapper expects {len(self._agent_order)} actions, "
                f"received shape {arr.shape}"
            )
        return {agent: self._continuous_to_discrete(value, agent) for agent, value in zip(self._agent_order, arr)}

    def _build_info(
        self,
        info_dict: Optional[Dict[str, Any]],
        reward_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        info_dict = info_dict or {}
        reward_dict = reward_dict or {}
        common = info_dict.get("__common__", {})
        flattened_rewards = {agent: float(reward_dict.get(agent, 0.0)) for agent in self._agent_order}
        aggregated = {
            "raw_info": info_dict,
            "reward_components": flattened_rewards,
            "time_encoding": common.get("time"),
            "workload": common.get("workload"),
            "weather": common.get("weather"),
            "carbon_intensity": common.get("ci"),
        }
        return aggregated

    # ----------------------------------------------------------------- Gym hooks
    def reset(  # type: ignore[override]
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._inner_env.seed(seed)
        obs_dict = self._inner_env.reset()
        obs = self._flatten_obs(obs_dict)
        info = self._build_info(getattr(self._inner_env, "infos", None))
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        mapped_action = self._map_action(action)
        obs_dict, reward_dict, terminateds, truncateds, info_dict = self._inner_env.step(mapped_action)
        obs = self._flatten_obs(obs_dict)
        reward = self._aggregate_reward(reward_dict)
        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(truncateds.get("__all__", False))
        info = self._build_info(info_dict, reward_dict)
        return obs, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._inner_env.seed(seed)

    def close(self) -> None:
        close_fn = getattr(self._inner_env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                # The upstream close() references an attribute that does not
                # exist; swallowing the error keeps shutdown graceful.
                pass


def make_sustaindc_env(
    training_num: int = 1,
    test_num: int = 1,
    vector_env_type: str = "dummy",
    env_config: Optional[Dict[str, Any]] = None,
    reward_aggregation: str = DEFAULT_REWARD_AGGREGATION,
    action_threshold: float = 0.33,
) -> tuple[gym.Env, gym.vector.VectorEnv, gym.vector.VectorEnv]:
    """
    Factory helper mirroring make_building_env/make_datacenter_env.
    """
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    def env_factory() -> SustainDCEnvWrapper:
        return SustainDCEnvWrapper(
            env_config=env_config,
            reward_aggregation=reward_aggregation,
            action_threshold=action_threshold,
        )

    env = env_factory()
    vector_cls = DummyVectorEnv if vector_env_type != "subproc" else SubprocVectorEnv
    train_envs = vector_cls([env_factory for _ in range(training_num)])
    test_envs = vector_cls([env_factory for _ in range(test_num)])
    return env, train_envs, test_envs
