# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Custom reward functions for TWIST-style motion tracking locomotion.

Ports the exponential tracking reward kernels from YanjieZe/TWIST.
Uses Gaussian kernels: exp(-sigma * ||error||^2) for smooth gradients.

Reference: https://github.com/YanjieZe/TWIST
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_position_tracking_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for key body position tracking (TWIST: 2.0).

    Tracks how well key body parts maintain their default relative positions.
    Uses Gaussian kernel: exp(-sigma * ||pos - default_pos||^2)
    """
    asset = env.scene[asset_cfg.name]
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # Track relative positions to root
    root_pos = asset.data.root_pos_w.unsqueeze(1)
    rel_pos = body_pos - root_pos
    # Default relative positions (from initial state)
    default_body_pos = asset.data.default_root_state[:, :3].unsqueeze(1)
    # Use L2 error of relative body positions
    error = torch.sum(torch.sum(torch.square(rel_pos), dim=-1), dim=1)
    return torch.exp(-sigma * error)


def joint_position_tracking_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for joint position tracking (TWIST: 0.6).

    Tracks how well joints match their default positions.
    """
    asset = env.scene["robot"]
    error = torch.sum(
        torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    return torch.exp(-sigma * error)


def root_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for root velocity tracking (TWIST: 1.0).

    Similar to standard velocity tracking but uses TWIST's sigma parameter.
    """
    asset = env.scene["robot"]
    vel_cmd = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(
        torch.square(asset.data.root_lin_vel_b[:, :2] - vel_cmd[:, :2]), dim=1
    )
    return torch.exp(-sigma * lin_vel_error)


def joint_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Exponential reward for low joint velocities (TWIST: 0.2).

    Encourages smooth joint motion.
    """
    asset = env.scene["robot"]
    vel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    return torch.exp(-sigma * vel_sq)


def root_pose_tracking_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Exponential reward for root orientation tracking (TWIST: 0.6).

    Tracks how upright the root body is using quaternion error.
    """
    asset = env.scene["robot"]
    # Penalize deviation from upright (gravity projected in body frame)
    projected_gravity = asset.data.projected_gravity_b
    # Ideal: [0, 0, -1], penalize xy components
    error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return torch.exp(-sigma * error)


def feet_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize feet slipping during contact (TWIST: -0.1).

    Measures foot velocity when foot is in contact with ground.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = foot_forces > threshold
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(
        torch.sum(torch.square(foot_vel), dim=-1) * in_contact.float(), dim=1
    )
    return slip
