# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Custom reward functions for SoFTA-style smooth locomotion.

Ports reward terms from LeCAR-Lab/SoFTA focused on end-effector stabilization
and smooth force-torque-aware locomotion.

Reference: https://github.com/LeCAR-Lab/SoFTA
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize end-effector linear acceleration (SoFTA: -1.0).

    Core SoFTA reward: minimizes hand acceleration for smooth EE motion.
    Uses body acceleration of wrist links.
    """
    asset = env.scene[asset_cfg.name]
    ee_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # linear accel
    return torch.sum(torch.sum(torch.square(ee_acc), dim=-1), dim=1)


def ee_angular_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize end-effector angular acceleration (SoFTA: -0.5).

    Prevents rotational jitter of hands during walking.
    """
    asset = env.scene[asset_cfg.name]
    ee_ang_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, 3:]  # angular accel
    return torch.sum(torch.sum(torch.square(ee_ang_acc), dim=-1), dim=1)


def ee_zero_acceleration_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential reward for near-zero EE acceleration (SoFTA: 2.0).

    Gaussian kernel: exp(-||a_EE||^2 / (2 * std^2))
    Encourages perfectly still hands.
    """
    asset = env.scene[asset_cfg.name]
    ee_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]
    acc_norm_sq = torch.sum(torch.sum(torch.square(ee_acc), dim=-1), dim=1)
    return torch.exp(-acc_norm_sq / (2.0 * std * std))


def gravity_alignment_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize end-effector gravity misalignment (SoFTA: -0.3).

    Ensures wrist orientation keeps the hand upright (e.g., holding a cup).
    Measures how much the EE z-axis deviates from world up.
    """
    asset = env.scene[asset_cfg.name]
    ee_quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    w, x, y, z = ee_quats[..., 0], ee_quats[..., 1], ee_quats[..., 2], ee_quats[..., 3]
    # z-component of rotated [0,0,1]: 1 - 2(x^2 + y^2)
    ee_up_z = 1.0 - 2.0 * (x * x + y * y)
    return torch.sum(torch.square(1.0 - ee_up_z), dim=1)


def gait_smoothness_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize non-smooth gait transitions (SoFTA: -0.1).

    Measures jerk in joint velocities for smoother walking.
    """
    asset = env.scene["robot"]
    joint_vel = asset.data.joint_vel
    # Approximate jerk as change in velocity (first-order finite diff)
    # Use action rate as proxy for joint velocity smoothness
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(actions - prev_actions), dim=1)
