"""
多任务建筑控制 + ICL 训练脚本（带 Transformer 上下文编码）

要点：
1) 多任务采样：跨 building/weather/location 组合训练，提升泛化。
2) 支持集提示：每个 episode 前用专家/随机短滚动生成 K 步 (s, a, r) 轨迹及统计摘要。
3) 上下文编码：支持集重建为序列，经位置编码和 TransformerEncoder，再与摘要/原始状态融合，供扩散 Actor 与双 Q 使用。
4) 训练流程延续 main_building.py，便于对比。
"""

import argparse
import os
import pprint
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from env.building_env_wrapper import make_building_env, BearEnvWrapper
from dropt_utils.logger_formatter import EnhancedTensorboardLogger
from dropt_utils.tianshou_compat import offpolicy_trainer
from dropt_utils.paper_logging import add_paper_logging_args, run_paper_logging
from env.building_config import (
    DEFAULT_REWARD_SCALE,
    DEFAULT_ENERGY_WEIGHT,
    DEFAULT_TEMP_WEIGHT,
    DEFAULT_VIOLATION_PENALTY,
    DEFAULT_TARGET_TEMP,
    DEFAULT_TEMP_TOLERANCE,
    DEFAULT_MAX_POWER,
    DEFAULT_TIME_RESOLUTION,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_TRAINING_NUM,
    DEFAULT_TEST_NUM,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GAMMA,
    DEFAULT_N_STEP,
    DEFAULT_STEP_PER_EPOCH,
    DEFAULT_STEP_PER_COLLECT,
    DEFAULT_EPISODE_PER_TEST,
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_BETA_SCHEDULE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ACTOR_LR,
    DEFAULT_CRITIC_LR,
    DEFAULT_EXPLORATION_NOISE,
    DEFAULT_LOG_DIR,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_MPC_PLANNING_STEPS,
)
from policy.diffusion_opt_icl import DiffusionOPT
from diffusion import Diffusion

import torch.nn as nn
import math

# ==============================
# Context-aware models (ICL)
# ==============================


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ICLActor(nn.Module):
    """
    实体化 Token 的扩散 Actor：房间/全局实体 + 支持集上下文。
    """

    def __init__(
        self,
        base_state_dim: int,
        full_state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        t_dim: int = 16,
        activation: str = "mish",
        support_steps: int = 8,
        token_dim: int = 0,
        summary_dim: int = 0,
        context_hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        _act = nn.Mish if activation == "mish" else nn.ReLU

        self.base_state_dim = base_state_dim
        self.full_state_dim = full_state_dim
        self.support_steps = support_steps
        self.token_dim = token_dim
        self.traj_flat_dim = support_steps * token_dim
        self.summary_dim = summary_dim
        self.d_model = hidden_dim
        self.base_roomnum = max(1, (base_state_dim - 2) // 3)
        self.action_dim = action_dim

        # 房间 token 编码：温度、GHI、负荷
        self.room_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 全局 token 编码：室外温度、地面温度
        self.global_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 支持集 token 编码
        self.context_proj = nn.Linear(token_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
            dropout=dropout,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=support_steps + 5)

        self.summary_mlp = nn.Sequential(
            nn.Linear(summary_dim, context_hidden_dim),
            _act(),
            nn.Linear(context_hidden_dim, hidden_dim),
        )

        self.mask_mlp = nn.Sequential(
            nn.Linear(base_state_dim + action_dim, context_hidden_dim),
            _act(),
            nn.Linear(context_hidden_dim, hidden_dim),
        )

        self.state_ln = nn.LayerNorm(hidden_dim)
        self.context_ln = nn.LayerNorm(hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim),
            _act(),
            nn.Linear(t_dim, t_dim),
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x, time, state_full):
        base_state = state_full[:, : self.base_state_dim]
        context_flat = state_full[:, self.base_state_dim :]

        # 支持集编码
        traj_flat = context_flat[:, : self.traj_flat_dim]
        traj_tokens = traj_flat.view(-1, self.support_steps, self.token_dim)
        traj_emb = self.context_proj(traj_tokens)
        traj_emb = self.pos_encoder(traj_emb)
        traj_emb = self.context_encoder(traj_emb)
        traj_ctx = traj_emb.mean(dim=1)

        # 摘要
        summary = context_flat[:, self.traj_flat_dim : self.traj_flat_dim + self.summary_dim]
        summary_ctx = self.summary_mlp(summary)

        # 掩码
        mask_start = self.traj_flat_dim + self.summary_dim
        masks = context_flat[:, mask_start:]
        mask_ctx = self.mask_mlp(masks)
        state_mask = masks[:, : self.base_state_dim]

        # 房间/全局实体编码
        temps = base_state[:, : self.base_roomnum]
        out_temp = base_state[:, self.base_roomnum]
        ghi = base_state[:, self.base_roomnum + 1 : self.base_roomnum * 2 + 1]
        ground_temp = base_state[:, self.base_roomnum * 2 + 1]
        occ = base_state[:, self.base_roomnum * 2 + 2 : self.base_roomnum * 3 + 2]

        room_feats = torch.stack([temps, ghi, occ], dim=-1)  # (B, R, 3)
        room_emb = self.room_encoder(room_feats)

        room_mask = state_mask[:, : self.base_roomnum].unsqueeze(-1)
        eps = 1e-6
        room_emb = room_emb * room_mask
        room_ctx = room_emb.sum(dim=1) / (room_mask.sum(dim=1) + eps)

        global_token = torch.stack([out_temp, ground_temp], dim=-1)
        global_ctx = self.global_encoder(global_token)

        s = self.state_ln(room_ctx + global_ctx)
        c = self.context_ln(traj_ctx + summary_ctx + mask_ctx)
        t_in = time.float().unsqueeze(-1)
        t = self.time_mlp(t_in)
        x = torch.cat([x, t, s + c], dim=1)
        return self.mid_layer(x)


class ICLDoubleCritic(nn.Module):
    """
    实体化 Token 的双 Q：房间/全局实体 + 支持集上下文，动作融入房间 token。
    """

    def __init__(
        self,
        base_state_dim: int,
        full_state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "relu",
        support_steps: int = 8,
        token_dim: int = 0,
        summary_dim: int = 0,
        context_hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        _act = nn.ReLU if activation == "relu" else nn.Mish
        self.base_state_dim = base_state_dim
        self.support_steps = support_steps
        self.token_dim = token_dim
        self.traj_flat_dim = support_steps * token_dim
        self.summary_dim = summary_dim
        self.base_roomnum = max(1, (base_state_dim - 2) // 3)
        self.action_dim = action_dim

        self.room_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # Temp, GHI, Occ, Action
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.context_proj = nn.Linear(token_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
            dropout=dropout,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=support_steps + 5)

        self.summary_mlp = nn.Sequential(
            nn.Linear(summary_dim, context_hidden_dim),
            _act(),
            nn.Linear(context_hidden_dim, hidden_dim),
        )
        self.mask_mlp = nn.Sequential(
            nn.Linear(base_state_dim + action_dim, context_hidden_dim),
            _act(),
            nn.Linear(context_hidden_dim, hidden_dim),
        )

        self.state_ln = nn.LayerNorm(hidden_dim)
        self.context_ln = nn.LayerNorm(hidden_dim)

        def build_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                _act(),
                nn.Linear(hidden_dim, hidden_dim),
                _act(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1_net = build_head()
        self.q2_net = build_head()

    def forward(self, state_full, action):
        base_state = state_full[:, : self.base_state_dim]
        context_flat = state_full[:, self.base_state_dim :]

        traj_flat = context_flat[:, : self.traj_flat_dim]
        traj_tokens = traj_flat.view(-1, self.support_steps, self.token_dim)
        traj_emb = self.context_proj(traj_tokens)
        traj_emb = self.pos_encoder(traj_emb)
        traj_emb = self.context_encoder(traj_emb)
        traj_ctx = traj_emb.mean(dim=1)

        summary = context_flat[:, self.traj_flat_dim : self.traj_flat_dim + self.summary_dim]
        summary_ctx = self.summary_mlp(summary)

        mask_start = self.traj_flat_dim + self.summary_dim
        masks = context_flat[:, mask_start:]
        state_mask = masks[:, : self.base_state_dim]
        action_mask = masks[:, self.base_state_dim : self.base_state_dim + self.action_dim]
        mask_ctx = self.mask_mlp(masks)

        temps = base_state[:, : self.base_roomnum]
        out_temp = base_state[:, self.base_roomnum]
        ghi = base_state[:, self.base_roomnum + 1 : self.base_roomnum * 2 + 1]
        ground_temp = base_state[:, self.base_roomnum * 2 + 1]
        occ = base_state[:, self.base_roomnum * 2 + 2 : self.base_roomnum * 3 + 2]

        room_action = action[:, : self.base_roomnum]
        room_feats = torch.stack([temps, ghi, occ, room_action], dim=-1)
        room_emb = self.room_encoder(room_feats)

        room_mask = (state_mask[:, : self.base_roomnum] * action_mask[:, : self.base_roomnum]).unsqueeze(-1)
        eps = 1e-6
        room_emb = room_emb * room_mask
        room_ctx = room_emb.sum(dim=1) / (room_mask.sum(dim=1) + eps)

        global_token = torch.stack([out_temp, ground_temp], dim=-1)
        global_ctx = self.global_encoder(global_token)

        s = self.state_ln(room_ctx + global_ctx)
        c = self.context_ln(traj_ctx + summary_ctx + mask_ctx)
        x = s + c
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

    def q_min(self, obs, action):
        return torch.min(*self.forward(obs, action))

# 默认多任务组合（可通过 --tasks 覆盖）
DEFAULT_TASKS = [
    ("OfficeSmall", "Hot_Dry", "Tucson"),
    ("OfficeSmall", "Hot_Humid", "Tampa"),
    ("Hospital", "Cold_Humid", "Rochester"),
    ("SchoolPrimary", "Hot_Dry", "Tucson"),
]

# ICL 专用默认参数（不影响无 ICL 版本）
ICL_DEFAULT_SUPPORT_STEPS = 6
ICL_DEFAULT_BC_DECAY_STEPS = 150000
ICL_DEFAULT_TEMP_WEIGHT = 0.85
ICL_DEFAULT_VIOLATION_PENALTY = 20.0
ICL_DEFAULT_REWARD_SCALE = 0.00025


def parse_tasks(task_str: Optional[str]) -> List[tuple]:
    """
    Parse task list string: "OfficeSmall:Hot_Dry:Tucson,Hospital:Cold_Humid:Rochester"
    """
    if not task_str:
        return DEFAULT_TASKS
    tasks = []
    for chunk in task_str.split(","):
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid task spec '{chunk}', expected format building:weather:location")
        tasks.append(tuple(parts))
    return tasks


class ICLContextWrapper(gym.Wrapper):
    """
    Append a rich support-set context to every observation.

    Context = [support_trajectory_flat, summary_stats], where support_trajectory_flat
    is the concat of K steps of (state, action, reward).
    """

    def __init__(
        self,
        env: BearEnvWrapper,
        support_steps: int = 8,
        use_expert_support: bool = True,
        support_expert_type: Optional[str] = "mpc",
        target_base_state_dim: Optional[int] = None,
        target_action_dim: Optional[int] = None,
    ):
        super().__init__(env)
        self.support_steps = support_steps
        self.use_expert_support = use_expert_support
        self.support_expert_type = support_expert_type
        # 原始维度
        self.orig_state_dim = env.state_dim
        self.orig_action_dim = env.action_dim
        # 目标对齐维度（跨任务统一）
        self.base_state_dim = target_base_state_dim or self.orig_state_dim
        self.base_action_dim = target_action_dim or self.orig_action_dim
        # 掩码（用于区分真实维度与填充维度）
        self.state_mask = np.concatenate(
            [np.ones(self.orig_state_dim, dtype=np.float32),
             np.zeros(self.base_state_dim - self.orig_state_dim, dtype=np.float32)]
        )
        self.action_mask = np.concatenate(
            [np.ones(self.orig_action_dim, dtype=np.float32),
             np.zeros(self.base_action_dim - self.orig_action_dim, dtype=np.float32)]
        )

        # flattened trajectory: K * (s + a + r)
        self.traj_flat_dim = support_steps * (self.base_state_dim + self.base_action_dim + 1)
        self.support_steps = support_steps
        self.token_dim = self.base_state_dim + self.base_action_dim + 1
        # summary stats: mean/std for states/actions
        self.summary_dim = 2 * self.base_state_dim + 2 * self.base_action_dim
        self.mask_dim = self.base_state_dim + self.base_action_dim
        self.context_dim = self.traj_flat_dim + self.summary_dim + self.mask_dim

        # 观测空间按对齐后的状态维度构造，避免低维任务的 low/high 不足
        pad_state_low = np.full(self.base_state_dim, -np.inf, dtype=np.float32)
        pad_state_high = np.full(self.base_state_dim, np.inf, dtype=np.float32)
        low = np.concatenate([pad_state_low, np.full(self.context_dim, -np.inf, dtype=np.float32)])
        high = np.concatenate([pad_state_high, np.full(self.context_dim, np.inf, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.current_context = np.zeros(self.context_dim, dtype=np.float32)
        # 统一动作空间到目标维度
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.base_action_dim,), dtype=np.float32)
        self._last_obs_padded = np.zeros(self.base_state_dim + self.context_dim, dtype=np.float32)
        self._last_raw_obs = np.zeros(self.orig_state_dim, dtype=np.float32)

    def _maybe_init_expert(self):
        if self.use_expert_support and self.env.expert_controller is None and self.support_expert_type:
            try:
                self.env._init_expert_controller(self.support_expert_type)
            except Exception:
                self.use_expert_support = False

    def _pad_state(self, state: np.ndarray) -> np.ndarray:
        if state.shape[0] == self.base_state_dim:
            return state.astype(np.float32, copy=False)
        pad_len = self.base_state_dim - state.shape[0]
        if pad_len < 0:
            return state[: self.base_state_dim].astype(np.float32, copy=False)
        return np.concatenate([state.astype(np.float32, copy=False), np.zeros(pad_len, dtype=np.float32)], axis=0)

    def _pad_action(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] == self.base_action_dim:
            return action.astype(np.float32, copy=False)
        pad_len = self.base_action_dim - action.shape[0]
        if pad_len < 0:
            return action[: self.base_action_dim].astype(np.float32, copy=False)
        return np.concatenate([action.astype(np.float32, copy=False), np.zeros(pad_len, dtype=np.float32)], axis=0)

    def _build_support_context(self, seed: Optional[int] = None) -> None:
        self._maybe_init_expert()

        states = []
        actions = []
        rewards = []
        obs, _ = self.env.reset(seed=seed)
        steps = 0
        while steps < self.support_steps:
            if self.use_expert_support and self.env.expert_controller is not None:
                try:
                    action = self.env.expert_controller.get_action(obs)
                except Exception:
                    action = self.env.action_space.sample()
            else:
                action = self.env.action_space.sample()

            next_obs, rew, done, truncated, _ = self.env.step(action)
            states.append(self._pad_state(obs))
            actions.append(self._pad_action(action))
            rewards.append(rew)
            obs = next_obs
            steps += 1
            if done or truncated:
                obs, _ = self.env.reset()

        states = np.asarray(states, dtype=np.float32)  # (K, base_state_dim)
        actions = np.asarray(actions, dtype=np.float32)  # (K, base_action_dim)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)  # (K,1)

        # 掩码应用到 token
        masked_states = states * self.state_mask
        masked_actions = actions * self.action_mask
        traj_flat = np.concatenate([masked_states, masked_actions, rewards], axis=1).reshape(-1)

        # 掩码统计（避免填充维度影响均值/方差）
        state_den = max(1.0, float(self.state_mask.sum()))
        action_den = max(1.0, float(self.action_mask.sum()))
        state_mean = (masked_states.sum(axis=0) / state_den)
        state_var = ((masked_states - state_mean) ** 2).sum(axis=0) / state_den
        state_std = np.sqrt(state_var + 1e-6)
        action_mean = (masked_actions.sum(axis=0) / action_den)
        action_var = ((masked_actions - action_mean) ** 2).sum(axis=0) / action_den
        action_std = np.sqrt(action_var + 1e-6)
        summary = np.concatenate([state_mean, state_std, action_mean, action_std], axis=0)

        # 将掩码也拼入上下文，供模型识别真实维度
        self.current_context = np.ascontiguousarray(
            np.concatenate([traj_flat, summary, self.state_mask, self.action_mask]),
            dtype=np.float32,
        )

        # reset before real episode
        self.env.reset()

    def _append_context(self, obs: np.ndarray) -> np.ndarray:
        padded = self._pad_state(obs)
        return np.ascontiguousarray(np.concatenate([padded, self.current_context], axis=-1), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._build_support_context(seed=seed)
        obs, _ = self.env.reset(seed=seed, options=options)
        # 返回简洁 info，避免 Collector 堆叠不同形状
        appended = self._append_context(obs)
        self._last_obs_padded = appended
        self._last_raw_obs = obs.astype(np.float32, copy=False)
        return appended, {}

    def step(self, action):
        # 截取前 orig_action_dim 传给底层环境
        env_action = action[: self.orig_action_dim]
        # 获取专家动作（用于 BC），固定维度
        expert_act = np.zeros(self.base_action_dim, dtype=np.float32)
        if self.use_expert_support and self.env.expert_controller is not None:
            try:
                raw_expert = self.env.expert_controller.get_action(self._last_raw_obs)
                expert_act = self._pad_action(np.asarray(raw_expert, dtype=np.float32))
            except Exception:
                expert_act = np.zeros(self.base_action_dim, dtype=np.float32)

        obs, reward, done, truncated, _ = self.env.step(env_action)
        info = {"expert_action": expert_act}
        appended = self._append_context(obs)
        self._last_obs_padded = appended
        self._last_raw_obs = obs.astype(np.float32, copy=False)
        return appended, reward, done, truncated, info

    def consume_metrics(self):
        """
        优先返回底层 env 已完成 episode 的指标；若尚未完成任何 episode，则返回当前未完成
        episode 的临时均值（避免长 episode 下始终为空）。
        """
        base = getattr(self.env, "consume_metrics", None)
        if base:
            m = base()
            if m:
                return m
        # 回退：尚无完成的 episode 时，使用正在进行中的累计指标
        steps = getattr(self.env, "_metric_steps", 0)
        if steps and steps > 0:
            energy_sum = getattr(self.env, "_metric_energy_sum", None)
            comfort_sum = getattr(self.env, "_metric_comfort_sum", None)
            violation_sum = getattr(self.env, "_metric_violation_sum", None)
            result = {}
            if energy_sum is not None:
                # 当前 episode 的已累计能耗（kWh）
                result["avg_energy"] = float(energy_sum)
            if comfort_sum is not None:
                result["avg_comfort_mean"] = float(comfort_sum / steps)
            if violation_sum is not None:
                result["avg_violations"] = float(violation_sum / steps)
            return result if result else None
        return None


def get_args():
    parser = argparse.ArgumentParser(description="Multi-task Building Training with ICL prompt")
    # 环境/任务相关
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma separated list of tasks building:weather:location. "
                             "Default uses a preset multi-task list.")
    parser.add_argument("--support-steps", type=int, default=ICL_DEFAULT_SUPPORT_STEPS,
                        help="Expert rollout steps to build ICL context per episode")
    parser.add_argument("--support-expert-type", type=str, default="mpc",
                        choices=["mpc", "pid", "rule", "bangbang", None],
                        help="Expert type for support set generation")
    parser.add_argument("--use-expert-support", action="store_true", default=True,
                        help="Use expert for support rollout (default True)")

    # 继承原有参数
    parser.add_argument("--target-temp", type=float, default=DEFAULT_TARGET_TEMP)
    parser.add_argument("--temp-tolerance", type=float, default=DEFAULT_TEMP_TOLERANCE)
    parser.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER)
    parser.add_argument("--time-resolution", type=int, default=DEFAULT_TIME_RESOLUTION)
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--energy-weight", type=float, default=DEFAULT_ENERGY_WEIGHT)
    parser.add_argument("--temp-weight", type=float, default=ICL_DEFAULT_TEMP_WEIGHT,
                        help=f"Temperature penalty weight (default {ICL_DEFAULT_TEMP_WEIGHT})")
    parser.add_argument("--add-violation-penalty", dest="add_violation_penalty",
                        action="store_true", default=True)
    parser.add_argument("--no-add-violation-penalty", dest="add_violation_penalty",
                        action="store_false")
    parser.add_argument("--violation-penalty", type=float, default=ICL_DEFAULT_VIOLATION_PENALTY,
                        help=f"Penalty per comfort violation (default {ICL_DEFAULT_VIOLATION_PENALTY})")
    parser.add_argument("--reward-scale", type=float, default=ICL_DEFAULT_REWARD_SCALE,
                        help=f"Reward scaling factor (default {ICL_DEFAULT_REWARD_SCALE})")
    parser.add_argument("--reward-normalization", dest="reward_normalization", action="store_true")
    parser.add_argument("--no-reward-normalization", dest="reward_normalization", action="store_false")
    parser.set_defaults(reward_normalization=True)

    parser.add_argument("--training-num", type=int, default=DEFAULT_TRAINING_NUM)
    parser.add_argument("--test-num", type=int, default=DEFAULT_TEST_NUM)
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--n-step", type=int, default=DEFAULT_N_STEP)
    parser.add_argument("--step-per-epoch", type=int, default=DEFAULT_STEP_PER_EPOCH)
    parser.add_argument("--step-per-collect", type=int, default=DEFAULT_STEP_PER_COLLECT)
    parser.add_argument("--episode-per-test", type=int, default=DEFAULT_EPISODE_PER_TEST)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--prior-alpha", type=float, default=0.6)
    parser.add_argument("--prior-beta", type=float, default=0.4)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument("--beta-schedule", type=str, default=DEFAULT_BETA_SCHEDULE)
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-prefix", type=str, default="icl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-e", "--epoch", type=int, default=20000)
    parser.add_argument("--vector-env-type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--log-update-interval", type=int, default=50)
    # 缓和更新频率，降低梯度波动（仅 ICL 版本调整）
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--lr-decay", action="store_true", default=False)
    parser.add_argument("--exploration-noise", type=float, default=DEFAULT_EXPLORATION_NOISE)
    parser.add_argument("--watch", action="store_true", default=False)
    parser.add_argument("--save-interval", type=int, default=10,
                        help="checkpoint 保存间隔（轮次，默认10）")
    parser.add_argument("--mpc-planning-steps", type=int, default=DEFAULT_MPC_PLANNING_STEPS)
    parser.add_argument("--bc-coef", action="store_true", default=True)
    parser.add_argument("--bc-weight", type=float, default=0.8)
    parser.add_argument("--bc-weight-final", type=float, default=0.1)
    parser.add_argument("--bc-weight-decay-steps", type=int, default=ICL_DEFAULT_BC_DECAY_STEPS,
                        help=f"Linear decay steps for BC weight (default {ICL_DEFAULT_BC_DECAY_STEPS})")
    add_paper_logging_args(parser)
    args = parser.parse_args()

    if args.reward_normalization and args.n_step > 1:
        # n_step 与奖励归一化不兼容，这里保持行为与原脚本一致
        print("警告: n_step>1 与 reward_normalization 不兼容，已自动关闭 reward_normalization")
        args.reward_normalization = False

    if args.bc_weight_final is None:
        args.bc_weight_final = args.bc_weight

    return args


def make_multitask_envs(args, tasks: List[tuple]):
    """
    Create vectorized environments, each bound to a specific (building, weather, location) task,
    and wrapped with ICLContextWrapper to inject support-set context.
    """
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    # 预探测各任务的原始维度，统一到最大维度
    max_state_dim = 0
    max_action_dim = 0
    for (bld, wea, loc) in tasks:
        probe = BearEnvWrapper(
            building_type=bld,
            weather_type=wea,
            location=loc,
            target_temp=args.target_temp,
            temp_tolerance=args.temp_tolerance,
            max_power=args.max_power,
            time_resolution=args.time_resolution,
            energy_weight=args.energy_weight,
            temp_weight=args.temp_weight,
            episode_length=args.episode_length,
            add_violation_penalty=args.add_violation_penalty,
            violation_penalty=args.violation_penalty,
            reward_scale=args.reward_scale,
            expert_type=None,
            expert_kwargs=None,
        )
        max_state_dim = max(max_state_dim, probe.state_dim)
        max_action_dim = max(max_action_dim, probe.action_dim)

    def make_env(task_cfg):
        building_type, weather_type, location = task_cfg
        env = BearEnvWrapper(
            building_type=building_type,
            weather_type=weather_type,
            location=location,
            target_temp=args.target_temp,
            temp_tolerance=args.temp_tolerance,
            max_power=args.max_power,
            time_resolution=args.time_resolution,
            energy_weight=args.energy_weight,
            temp_weight=args.temp_weight,
            episode_length=args.episode_length,
            add_violation_penalty=args.add_violation_penalty,
            violation_penalty=args.violation_penalty,
            reward_scale=args.reward_scale,
            expert_type=args.support_expert_type if args.use_expert_support else None,
            expert_kwargs={"planning_steps": args.mpc_planning_steps} if args.support_expert_type == "mpc" else None,
        )
        return ICLContextWrapper(
            env,
            support_steps=args.support_steps,
            use_expert_support=args.use_expert_support,
            support_expert_type=args.support_expert_type,
            target_base_state_dim=max_state_dim,
            target_action_dim=max_action_dim,
        )

    def build_vec(num_envs: int):
        env_refs = []

        def make_factory(task_cfg):
            def _factory():
                inst = make_env(task_cfg)
                # 保存环境引用，供 logger 聚合环境指标
                env_refs.append(inst)
                return inst

            return _factory

        factories = [make_factory(tasks[i % len(tasks)]) for i in range(num_envs)]
        vec_cls = DummyVectorEnv if args.vector_env_type != "subproc" else SubprocVectorEnv
        vec = vec_cls(factories)
        if env_refs:
            setattr(vec, "_env_list", env_refs)
        return vec

    probe_env = make_env(tasks[0])
    train_envs = build_vec(args.training_num)
    test_envs = build_vec(args.test_num)
    train_eval_envs = build_vec(len(tasks))
    return probe_env, train_envs, test_envs, train_eval_envs


def main():
    args = get_args()
    tasks = parse_tasks(args.tasks)

    # 确保测试环境数量足以覆盖所有任务
    if args.test_num < len(tasks):
        print(
            f"[提示] test_num={args.test_num} 小于任务数量 {len(tasks)}，已自动调整为覆盖全部任务"
        )
        args.test_num = len(tasks)
    # device & seeds
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{args.log_prefix}_icl_{timestamp}"
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    print("\n" + "=" * 60)
    print("  Multi-task Building + ICL Training")
    print("=" * 60)
    print(f"\n任务列表 ({len(tasks)}):")
    for t in tasks:
        print(f"  - building={t[0]}, weather={t[1]}, location={t[2]}")
    print("\n配置参数:")
    pprint.pprint(vars(args))
    print()

    # environments
    env, train_envs, test_envs, train_eval_envs = make_multitask_envs(args, tasks)
    print("环境创建成功")
    print(f"  原始状态维度: {env.base_state_dim}")
    print(f"  上下文维度: {env.context_dim}")
    print(f"  拼接后状态维度: {env.observation_space.shape[0]}")
    print(f"  动作维度: {env.action_space.shape[0]}")

    # metrics getter (only works for dummy vector env where _env_list is accessible)
    def _aggregate_metrics(vector_env):
        """
        聚合向量环境中各实例的 episode 级环境指标。
        优先使用 DummyVectorEnv 注入的 _env_list；若缺失则回退到 envs 属性。
        """
        if vector_env is None:
            return None
        env_list = getattr(vector_env, "_env_list", None)
        if not env_list:
            env_list = getattr(vector_env, "envs", None)
        if not env_list:
            return None
        values = []
        for env_inst in env_list:
            consume = getattr(env_inst, "consume_metrics", None)
            if consume:
                m = consume()
                if m:
                    values.append(m)
        if not values:
            return None
        result = {}
        for key in ("avg_energy", "avg_comfort_mean", "avg_violations", "avg_pue"):
            nums = [m[key] for m in values if m.get(key) is not None]
            if nums:
                result[key] = float(np.mean(nums))
        return result if result else None

    def metrics_getter(mode: str):
        target_env = train_envs if mode == "train" else test_envs
        return _aggregate_metrics(target_env)

    # networks (context-aware)
    full_state_dim = env.observation_space.shape[0]
    base_state_dim = env.base_state_dim  # original state without context
    action_dim = env.action_space.shape[0]

    actor = ICLActor(
        base_state_dim=base_state_dim,
        full_state_dim=full_state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        t_dim=16,
        support_steps=env.support_steps,
        token_dim=env.token_dim,
        summary_dim=env.summary_dim,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=1e-4)

    critic = ICLDoubleCritic(
        base_state_dim=base_state_dim,
        full_state_dim=full_state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        support_steps=env.support_steps,
        token_dim=env.token_dim,
        summary_dim=env.summary_dim,
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)

    diffusion = Diffusion(
        state_dim=full_state_dim,
        action_dim=action_dim,
        model=actor,
        max_action=1.0,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.diffusion_steps,
    ).to(args.device)

    policy = DiffusionOPT(
        state_dim=full_state_dim,
        actor=diffusion,
        actor_optim=actor_optim,
        action_dim=action_dim,
        critic=critic,
        critic_optim=critic_optim,
        device=args.device,
        tau=0.005,
        gamma=args.gamma,
        exploration_noise=args.exploration_noise,
        bc_coef=args.bc_coef,
        bc_weight=args.bc_weight,
        bc_weight_final=args.bc_weight_final,
        bc_weight_decay_steps=args.bc_weight_decay_steps,
        action_space=env.action_space,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        reward_normalization=args.reward_normalization,
    )

    print("网络创建成功")
    print(f"  Actor 参数量: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Critic 参数量: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"  扩散步数: {args.diffusion_steps}, beta schedule: {args.beta_schedule}")
    if args.bc_coef:
        print(f"  行为克隆权重: 初始 {args.bc_weight}, 目标 {args.bc_weight_final}")

    # collectors
    buffer_num = max(1, args.training_num)
    if args.prioritized_replay:
        replay_buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size, buffer_num=buffer_num, alpha=args.prior_alpha, beta=args.prior_beta
        )
    else:
        replay_buffer = VectorReplayBuffer(args.buffer_size, buffer_num)

    # Collector 不再额外注入噪声，使用策略自身的探索机制，保证训练/测试指标可对齐
    train_collector = Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = Collector(policy, test_envs)
    train_eval_collector = Collector(policy, train_eval_envs)

    logger = EnhancedTensorboardLogger(
        writer=writer,
        total_epochs=args.epoch,
        reward_scale=args.reward_scale,
        log_interval=1,
        verbose=True,
        diffusion_steps=args.diffusion_steps,
        update_log_interval=args.log_update_interval,
        step_per_epoch=args.step_per_epoch,
        metrics_getter=metrics_getter,
        context_info={
            "support_steps": args.support_steps,
            "state_dim": base_state_dim,
            "action_dim": action_dim,
            "token_dim": env.token_dim,
            "summary_dim": env.summary_dim,
            "context_dim": env.context_dim,
        },
        train_eval_collector=train_eval_collector,
        train_eval_episodes=args.episode_per_test,
        png_interval=5,
    )

    print("\n" + "=" * 60)
    print("  开始 ICL 多任务训练")
    print("=" * 60)
    print(f"[提示] 奖励已缩放 {args.reward_scale}x")

    latest_ckpt_path = os.path.join(log_path, "checkpoint_latest.pth")

    def _save_checkpoint():
        torch.save(
            {
                "model": policy.state_dict(),
                "optim_actor": actor_optim.state_dict(),
                "optim_critic": critic_optim.state_dict(),
            },
            latest_ckpt_path,
        )

    last_paper_epoch = {"value": None}

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            _save_checkpoint()
        if args.paper_log and args.paper_log_interval > 0 and epoch % args.paper_log_interval == 0:
            try:
                print(f"\n[paper-log] Epoch {epoch}: collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                last_paper_epoch["value"] = epoch
            except Exception as exc:
                print(f"[paper-log] Failed at epoch {epoch}: {exc}")
        return None

    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        logger=logger,
        save_best_fn=lambda pol: torch.save(pol.state_dict(), os.path.join(log_path, "policy_best.pth")),
        save_checkpoint_fn=save_checkpoint_fn,
    )

    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)
    pprint.pprint(result)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

    if args.paper_log:
        try:
            if args.paper_log_interval > 0 and last_paper_epoch["value"] == args.epoch:
                print("[paper-log] Skipped final logging (already captured at last epoch).")
            else:
                print("\n[paper-log] Collecting trajectories and plots ...")
                run_paper_logging(
                    env=env,
                    policy=policy,
                    actor=actor,
                    guidance_fn=None,
                    args=args,
                    log_path=log_path,
                )
                print(f"[paper-log] Saved to: {os.path.join(log_path, 'paper_data')}")
        except Exception as exc:
            print(f"[paper-log] Failed: {exc}")
    print(f"\n[提示] 模型已保存到: {log_path}")


if __name__ == "__main__":
    main()
