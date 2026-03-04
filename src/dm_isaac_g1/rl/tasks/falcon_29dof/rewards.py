# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Custom reward functions for FALCON-style force-adaptive locomotion.

Ports reward terms from LeCAR-Lab/FALCON that are not available in upstream
unitree_rl_lab or Isaac Lab mdp modules.

Reference: https://github.com/LeCAR-Lab/FALCON
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hip_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive hip joint positions (FALCON: -2.5).

    Prevents over-extension of hip joints during force adaptation.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)


def negative_knee_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize negative (hyperextended) knee angles (FALCON: -1.0).

    Safety constraint to prevent joint damage.
    """
    asset = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(-knee_pos, min=0.0), dim=1)


def stance_foot_lateral_tap(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize lateral foot tapping during stance (FALCON: -5.0).

    Detects rapid lateral foot movements when foot should be planted.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Get foot contact forces
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_stance = foot_forces > threshold  # (N, 2)
    # Get lateral foot velocity
    foot_vel = asset.data.body_vel_w[:, asset_cfg.body_ids, 1]  # y-velocity
    lateral_tap = torch.sum(torch.square(foot_vel) * in_stance.float(), dim=1)
    return lateral_tap


def root_lateral_drift(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral root drift during stance (FALCON: -5.0).

    Maintains stability during force adaptation by keeping COM centered.
    """
    asset = env.scene[asset_cfg.name]
    # Lateral velocity of root in body frame
    root_vel_b = asset.data.root_lin_vel_b[:, 1]  # y-component in body frame
    return torch.square(root_vel_b)


def contact_loss_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize unexpected loss of ground contact (FALCON: -0.15).

    Ensures feet maintain planned contact during force disturbances.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    total_contact = torch.sum(foot_forces, dim=1)
    return (total_contact < threshold).float()


def walking_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float = 0.78,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential reward for maintaining target walking height (FALCON: 2.0).

    Uses Gaussian kernel: exp(-||h - h_target||^2 / (2 * std^2))
    """
    asset = env.scene["robot"]
    height = asset.data.root_pos_w[:, 2]
    return torch.exp(-torch.square(height - target_height) / (2.0 * std * std))


def upper_body_joint_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of upper body joint default positions (FALCON: 4.0).

    Encourages upper body to maintain default pose while lower body adapts.
    Uses exponential kernel for smooth gradients.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    return torch.exp(-error * 0.5)
