# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""FALCON-style 29-DOF force-adaptive locomotion config.

Ports the reward structure from LeCAR-Lab/FALCON for robust humanoid locomotion
with force adaptation capabilities. Key features:
- Hip position regularization for stable stance
- Anti-knee-hyperextension safety
- Stance-phase lateral stability penalties
- Upper body joint tracking (arms maintain pose during disturbances)
- Strong domain randomization (mass, friction, pushes)
- Wider push perturbations simulating external forces

Reference: https://github.com/LeCAR-Lab/FALCON
PPO config: [512, 256, 128] actor/critic (FALCON defaults)
"""

import importlib

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Load the upstream 29dof module
_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
_mdp_module = importlib.import_module("unitree_rl_lab.tasks.locomotion.mdp")
mdp = _mdp_module

from dm_isaac_g1.rl.tasks.falcon_29dof import rewards as falcon_rewards
from dm_isaac_g1.rl.tasks.agile_locomotion_29dof import rewards as agile_rewards

_UpstreamRobotEnvCfg = _module.RobotEnvCfg
_UpstreamRobotPlayEnvCfg = _module.RobotPlayEnvCfg


@configclass
class FALCONPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config matching FALCON architecture."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "DM-G1-29dof-FALCON"
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
    """FALCON reward terms — force-adaptive locomotion.

    Categories:
    1. Task rewards (velocity tracking, height)
    2. Stability penalties (hip, knee, stance lateral)
    3. Joint regularization
    4. Upper body tracking
    5. Action smoothness
    6. Contact management
    """

    # -- 1. TASK: Velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=4.0,  # FALCON: stronger turning control
        params={"command_name": "base_velocity", "std": 0.2},
    )

    # Walking height maintenance
    walking_height = RewTerm(
        func=falcon_rewards.walking_height_tracking,
        weight=2.0,
        params={"target_height": 0.78, "std": 0.1},
    )

    # -- 2. STABILITY
    hip_position = RewTerm(
        func=falcon_rewards.hip_position_penalty,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"],
            )
        },
    )

    negative_knee = RewTerm(
        func=falcon_rewards.negative_knee_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
        },
    )

    stance_lateral_tap = RewTerm(
        func=falcon_rewards.stance_foot_lateral_tap,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    root_lateral = RewTerm(
        func=falcon_rewards.root_lateral_drift,
        weight=-5.0,
    )

    contact_loss = RewTerm(
        func=falcon_rewards.contact_loss_penalty,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)

    # Ankle roll penalty (FALCON: -2.0)
    ankle_roll = RewTerm(
        func=agile_rewards.feet_roll_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        },
    )

    # -- 3. JOINT REGULARIZATION
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.5)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # -- 4. UPPER BODY: keep waist/arms near default
    waist_tracking = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,  # FALCON: waist DOF tracking
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )

    upper_body_tracking = RewTerm(
        func=falcon_rewards.upper_body_joint_tracking,
        weight=4.0,  # FALCON: strong upper body default pose tracking
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )

    # -- 5. ACTION SMOOTHNESS
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)

    # -- 6. GAIT + CONTACT
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
class FALCONEventCfg(_module.EventCfg):
    """FALCON-style strong domain randomization for force adaptation."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.5),
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

    # FALCON: stronger and more frequent pushes (simulating external forces)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),  # FALCON: frequent perturbations
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class FALCONCommandsCfg(_module.CommandsCfg):
    """FALCON velocity commands."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.15, 0.15)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class RobotEnvCfg(_UpstreamRobotEnvCfg):
    """FALCON force-adaptive locomotion training environment."""

    rewards: RewardsCfg = RewardsCfg()
    events: FALCONEventCfg = FALCONEventCfg()
    commands: FALCONCommandsCfg = FALCONCommandsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
