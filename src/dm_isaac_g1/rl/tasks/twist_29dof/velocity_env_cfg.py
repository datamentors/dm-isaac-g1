# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""TWIST-style 29-DOF motion-tracking locomotion config.

Ports the reward structure from YanjieZe/TWIST for full-body motion tracking
with exponential Gaussian reward kernels. Key features:
- Exponential tracking kernels (exp(-sigma * error^2)) for all tracking terms
- Full-body position tracking (not just velocity)
- Joint position + velocity tracking
- Root pose tracking (upright orientation)
- Feet slip penalty with contact detection
- Smooth action rate penalties

Note: Original TWIST uses teacher-student distillation. We implement the teacher
reward structure here for direct PPO training.

Reference: https://github.com/YanjieZe/TWIST
PPO config: [512, 512, 256, 128] (TWIST uses deeper network with SiLU)
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

from dm_isaac_g1.rl.tasks.twist_29dof import rewards as twist_rewards

_UpstreamRobotEnvCfg = _module.RobotEnvCfg
_UpstreamRobotPlayEnvCfg = _module.RobotPlayEnvCfg


@configclass
class TWISTPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config matching TWIST architecture.

    TWIST uses deeper MLP [512, 512, 256, 128] with SiLU activation.
    We use ELU (Isaac Lab default) but keep the deeper architecture.
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "DM-G1-29dof-TWIST"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],  # TWIST: deeper network
        critic_hidden_dims=[512, 512, 256, 128],
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
    """TWIST reward terms — exponential tracking kernels.

    Categories:
    1. Motion tracking (exponential kernels)
    2. Base stability
    3. Joint regularization
    4. Feet contact quality
    5. Action smoothness
    """

    # -- 1. MOTION TRACKING (exponential Gaussian kernels)
    # Key body position tracking (TWIST: 2.0)
    body_pos_tracking = RewTerm(
        func=twist_rewards.body_position_tracking_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "pelvis",
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "torso_link",
                ],
            ),
            "sigma": 0.25,
        },
    )

    # Joint position tracking (TWIST: 0.6)
    joint_pos_tracking = RewTerm(
        func=twist_rewards.joint_position_tracking_exp,
        weight=0.6,
        params={"sigma": 0.25},
    )

    # Root velocity tracking (TWIST: 1.0)
    root_vel_tracking = RewTerm(
        func=twist_rewards.root_velocity_tracking_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "sigma": 0.25},
    )

    # Joint velocity tracking (TWIST: 0.2)
    joint_vel_tracking = RewTerm(
        func=twist_rewards.joint_velocity_tracking_exp,
        weight=0.2,
        params={"sigma": 0.1},
    )

    # Root pose tracking (TWIST: 0.6)
    root_pose_tracking = RewTerm(
        func=twist_rewards.root_pose_tracking_exp,
        weight=0.6,
        params={"sigma": 0.25},
    )

    # Standard velocity tracking (for curriculum compatibility)
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

    # -- 2. BASE STABILITY
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.78}
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)

    # -- 3. JOINT REGULARIZATION
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.05)  # TWIST: -0.05
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)

    # -- 4. FEET CONTACT
    feet_slip = RewTerm(
        func=twist_rewards.feet_slip_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
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

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # -- 5. ACTION SMOOTHNESS
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # TWIST: -0.01

    # -- 6. TERMINATION
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)


@configclass
class TWISTEventCfg(_module.EventCfg):
    """TWIST domain randomization."""

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
        interval_range_s=(2.0, 5.0),
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
    )


@configclass
class RobotEnvCfg(_UpstreamRobotEnvCfg):
    """TWIST motion-tracking locomotion training environment."""

    rewards: RewardsCfg = RewardsCfg()
    events: TWISTEventCfg = TWISTEventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
