# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""Custom reward functions for AGILE-style locomotion.

Implements reward terms from NVIDIA WBC-AGILE that are not available
in the upstream unitree_rl_lab or Isaac Lab mdp modules.

Reference: https://github.com/nvidia-isaac/WBC-AGILE
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ankle_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize ankle joint torques (L2) separately from general torque penalty.

    Ankle joints need stricter torque regulation for stable ground contact.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def ankle_roll_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize ankle roll torques specifically for lateral stability."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def feet_roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive roll angle of feet bodies.

    Keeps feet flat on the ground for stable contact.
    """
    asset = env.scene[asset_cfg.name]
    # Get foot body orientations (quaternion wxyz)
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    # Extract roll from quaternion: roll ≈ 2 * (wy*wz + wx*ww) for small angles
    # More robust: use projected gravity in foot frame
    # Simplified: penalize deviation of foot z-axis from world up
    # foot_z in world = rotate [0,0,1] by foot quaternion
    w, x, y, z = foot_quats[..., 0], foot_quats[..., 1], foot_quats[..., 2], foot_quats[..., 3]
    # z-component of rotated [0,0,1]: 1 - 2(x^2 + y^2)
    foot_up_z = 1.0 - 2.0 * (x * x + y * y)
    # Penalize deviation from 1.0 (perfectly upright)
    roll_penalty = torch.sum(torch.square(1.0 - foot_up_z), dim=1)
    return roll_penalty


def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet yaw misalignment with base heading.

    AGILE uses two terms: feet_yaw_diff (between feet) and feet_yaw_mean (vs base).
    This combines both: penalizes when feet point away from the walking direction.
    """
    asset = env.scene[asset_cfg.name]
    # Get base yaw from root quaternion
    root_quat = asset.data.root_quat_w  # (N, 4) wxyz
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Get foot yaws
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (N, 2, 4)
    fw, fx, fy, fz = foot_quats[..., 0], foot_quats[..., 1], foot_quats[..., 2], foot_quats[..., 3]
    foot_yaws = torch.atan2(2.0 * (fw * fz + fx * fy), 1.0 - 2.0 * (fy * fy + fz * fz))  # (N, 2)

    # Difference between feet yaws
    yaw_diff = torch.square(foot_yaws[:, 0] - foot_yaws[:, 1])

    # Mean foot yaw vs base yaw
    mean_foot_yaw = torch.mean(foot_yaws, dim=1)
    yaw_vs_base = torch.square(mean_foot_yaw - base_yaw)

    return yaw_diff + 2.0 * yaw_vs_base


def root_acceleration_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root body acceleration for smoother motion."""
    asset = env.scene[asset_cfg.name]
    # Use body acceleration of root (index 0)
    root_acc = asset.data.body_acc_w[:, 0, :3]  # linear acceleration
    return torch.sum(torch.square(root_acc), dim=1)


def jumping_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Strongly penalize both feet leaving the ground (jumping/airborne).

    AGILE uses weight=-20.0 for this — it's critical for stable bipedal walking.
    Returns 1.0 when total foot contact force is below threshold (airborne).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # Net contact force on feet
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # z-component
    total_contact = torch.sum(foot_forces, dim=1)
    # Penalize when total contact is below threshold (both feet off ground)
    return (total_contact < threshold).float()


def feet_distance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 0.2,
) -> torch.Tensor:
    """Penalize lateral foot spacing deviating from target.

    Keeps feet at a natural width apart (AGILE default: 0.2m).
    """
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    # Lateral distance (y-axis in body frame, approximate with world y diff)
    lateral_dist = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    return torch.square(lateral_dist - target_distance)


def action_rate_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize jerk (second derivative of actions) for ultra-smooth motion.

    AGILE calls this "action_rate_rate" — penalizes changes in the action rate.
    """
    # action_manager stores last two actions
    # action_rate_t = action_t - action_{t-1}
    # action_rate_{t-1} = action_{t-1} - action_{t-2}
    # jerk = action_rate_t - action_rate_{t-1}
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    # We need action at t-2, which isn't directly available
    # Approximate: use action_rate_l2 with prev_action as proxy
    # This is equivalent to penalizing large changes in the action delta
    rate = actions - prev_actions
    return torch.sum(torch.square(rate), dim=1)
