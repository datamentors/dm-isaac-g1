# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""SoFTA-style 29-DOF smooth locomotion config.

Ports the reward structure from LeCAR-Lab/SoFTA for gentle humanoid locomotion
with end-effector stabilization. Key features:
- End-effector acceleration penalties (keep hands still while walking)
- Gravity alignment (upright hand orientation)
- Gait smoothness (minimize jerk)
- Strong orientation tracking for stable base

Note: The original SoFTA uses dual-agent async control (100Hz upper / 50Hz lower).
In Isaac Lab we approximate this with a single agent but keep the EE-focused rewards.

Reference: https://github.com/LeCAR-Lab/SoFTA
PPO config: [512, 256, 128] (matching SoFTA MLP architecture)
"""

import importlib

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
_mdp_module = importlib.import_module("unitree_rl_lab.tasks.locomotion.mdp")
mdp = _mdp_module

from dm_isaac_g1.rl.tasks.softa_29dof import rewards as softa_rewards
from dm_isaac_g1.rl.tasks.agile_locomotion_29dof import rewards as agile_rewards

_UpstreamRobotEnvCfg = _module.RobotEnvCfg
_UpstreamRobotPlayEnvCfg = _module.RobotPlayEnvCfg


@configclass
class SoFTAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config matching SoFTA architecture."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "DM-G1-29dof-SoFTA"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RewardsCfg:
    """SoFTA reward terms — smooth locomotion with EE stabilization.

    Categories:
    1. Task rewards (velocity tracking)
    2. End-effector stabilization (acceleration, gravity alignment)
    3. Base stability
    4. Joint regularization
    5. Gait quality
    """

    # -- 1. TASK: Velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )

    # -- 2. END-EFFECTOR STABILIZATION (core SoFTA contribution)
    ee_accel = RewTerm(
        func=softa_rewards.ee_acceleration_penalty,
        weight=-1.0,  # SoFTA: strong hand acceleration penalty
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]
            ),
        },
    )

    ee_ang_accel = RewTerm(
        func=softa_rewards.ee_angular_acceleration_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]
            ),
        },
    )

    ee_zero_accel = RewTerm(
        func=softa_rewards.ee_zero_acceleration_exp,
        weight=2.0,  # Positive: reward near-zero acceleration
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]
            ),
            "std": 0.5,
        },
    )

    gravity_alignment = RewTerm(
        func=softa_rewards.gravity_alignment_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]
            ),
        },
    )

    # -- 3. BASE STABILITY
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-2.5, params={"target_height": 0.78}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)
    root_acc = RewTerm(
        func=agile_rewards.root_acceleration_l2,
        weight=-1e-5,
    )

    # -- 4. JOINT REGULARIZATION
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)

    # Keep arms near default (SoFTA: arms should be relatively still)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,  # Moderate: arms should stay still but not rigid
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    # -- 5. GAIT QUALITY
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)
    action_jerk = RewTerm(
        func=softa_rewards.gait_smoothness_penalty,
        weight=-0.1,
    )

    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)


@configclass
class SoFTAEventCfg(_module.EventCfg):
    """SoFTA domain randomization — moderate perturbations."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 7.0),  # SoFTA: less frequent (gentle)
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RobotEnvCfg(_UpstreamRobotEnvCfg):
    """SoFTA smooth locomotion training environment."""

    rewards: RewardsCfg = RewardsCfg()
    events: SoFTAEventCfg = SoFTAEventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
