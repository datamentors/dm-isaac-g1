# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""AGILE-style 29-DOF velocity-tracking locomotion config.

Adapts NVIDIA WBC-AGILE's 24-term reward structure for robust, natural G1
locomotion with sim-to-real transfer characteristics.

Key differences from military march:
- 24 reward terms (vs ~20) with AGILE-specific aesthetic rewards
- Anti-jumping penalty (-20.0) — critical for stable bipedal walking
- Feet yaw alignment — prevents toe-out walking
- Action jerk penalty — smoother actuator commands
- Ankle-specific torque penalties — stable ground contact
- Stronger orientation tracking (5.0) — upright posture
- Higher base height target (0.78m) with AGILE-style exp reward
- Wider velocity command range for agile maneuvers
- Stronger domain randomization (friction, mass, pushes)

Reference: https://github.com/nvidia-isaac/WBC-AGILE
PPO config: [256, 256, 128] actor, [512, 256, 128] critic (AGILE defaults)
"""

import importlib
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Load the upstream 29dof module
_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)

# Import mdp from upstream (has all base reward functions)
_mdp_module = importlib.import_module("unitree_rl_lab.tasks.locomotion.mdp")
mdp = _mdp_module

# Import our custom AGILE reward functions
from dm_isaac_g1.rl.tasks.agile_locomotion_29dof import rewards as agile_rewards

# Import base classes
_UpstreamRobotEnvCfg = _module.RobotEnvCfg
_UpstreamRobotPlayEnvCfg = _module.RobotPlayEnvCfg


# ---------------------------------------------------------------------------
# PPO Runner Config — AGILE architecture
# ---------------------------------------------------------------------------
@configclass
class AgilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config matching AGILE architecture.

    Actor: [256, 256, 128] — smaller than default for faster inference
    Critic: [512, 256, 128] — asymmetric (larger for value estimation)
    Entropy: 0.005 with implicit decay via adaptive LR schedule
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "DM-G1-29dof-AgileLocomotion"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
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


# ---------------------------------------------------------------------------
# Rewards — AGILE 24-term structure
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """AGILE-style reward terms — 24 terms for robust natural locomotion.

    Organized into categories following AGILE convention:
    1. Task rewards (velocity tracking)
    2. Base stability (orientation, height, acceleration)
    3. Joint regularization (torques, velocities, limits)
    4. Action smoothness (rate, jerk)
    5. Feet aesthetics (slip, roll, yaw, distance, jumping)
    6. Termination
    """

    # ── 1. TASK: Velocity tracking ──────────────────────────────────────
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,  # AGILE: 5.0 (strong forward drive)
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,  # AGILE: 5.0
        params={"command_name": "base_velocity", "std": 0.2},
    )

    # ── 2. BASE STABILITY ───────────────────────────────────────────────
    # Orientation — strong upright posture enforcement
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,  # AGILE: 5.0 (exp-based, we use L2 penalty equivalent)
    )

    # Base height — target pelvis height 0.78m
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.5,  # AGILE: 2.5 (exp reward at std=0.1)
        params={"target_height": 0.78},
    )

    # Vertical bounce penalty
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.25,  # AGILE: -0.25
    )

    # Roll/pitch angular velocity penalty
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.25,  # AGILE: -0.25
    )

    # Root body acceleration (smoother motion)
    root_acc = RewTerm(
        func=agile_rewards.root_acceleration_l2,
        weight=-1e-5,  # AGILE: -1e-5
    )

    # ── 3. JOINT REGULARIZATION ─────────────────────────────────────────
    # General torque penalty
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-5,  # AGILE: -5e-5
    )

    # Ankle-specific torque penalties (stricter for ground contact stability)
    ankle_torques = RewTerm(
        func=agile_rewards.ankle_torques_l2,
        weight=-1e-4,  # AGILE: -1e-4
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*ankle_pitch_joint", ".*ankle_roll_joint"],
            )
        },
    )

    # Joint velocity penalty
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,  # AGILE: -1e-4
    )

    # Joint position limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,  # AGILE: -0.5
    )

    # Energy penalty (efficiency)
    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-5,
    )

    # Joint deviation — keep non-active joints near default
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,  # Light penalty — arms should be free to balance
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
            )
        },
    )

    # ── 4. ACTION SMOOTHNESS ────────────────────────────────────────────
    # Action rate (first derivative)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.25,  # AGILE: -0.25
    )

    # Action jerk (second derivative) — smoother actuator commands
    action_rate_rate = RewTerm(
        func=agile_rewards.action_rate_rate_l2,
        weight=-0.025,  # AGILE: -0.025
    )

    # ── 5. FEET AESTHETICS & CONTACT ────────────────────────────────────
    # Feet slip penalty
    feet_slip = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,  # AGILE: -0.05
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # Feet roll — keep feet flat
    feet_roll = RewTerm(
        func=agile_rewards.feet_roll_penalty,
        weight=-0.05,  # AGILE: -0.05
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        },
    )

    # Feet yaw alignment — prevent toe-out walking
    feet_yaw = RewTerm(
        func=agile_rewards.feet_yaw_alignment,
        weight=-0.5,  # Combined feet_yaw_diff (-0.1) + feet_yaw_mean (-2.0)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
        },
    )

    # Feet distance — lateral spacing
    feet_distance = RewTerm(
        func=agile_rewards.feet_distance_penalty,
        weight=-0.1,  # AGILE: -0.1
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "target_distance": 0.2,
        },
    )

    # ANTI-JUMPING — critical for stable bipedal walking
    jumping = RewTerm(
        func=agile_rewards.jumping_penalty,
        weight=-20.0,  # AGILE: -20.0 (very strong!)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 10.0,
        },
    )

    # Gait periodicity
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

    # Feet air time
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # Undesired contacts (knees, torso hitting ground)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # ── 6. TERMINATION ──────────────────────────────────────────────────
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,  # AGILE: -100.0
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)


# ---------------------------------------------------------------------------
# Event Config — AGILE-style domain randomization (stronger than default)
# ---------------------------------------------------------------------------
@configclass
class AgileEventCfg(_module.EventCfg):
    """Stronger domain randomization following AGILE.

    Key additions over baseline:
    - Wider friction range (0.2-1.5 vs 0.3-1.0)
    - Larger mass perturbation (-1 to 5kg vs -1 to 3kg)
    - More frequent push perturbations (2-5s vs 5s)
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),   # AGILE: wider range
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 5.0),  # AGILE: up to 5kg
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),  # AGILE: 2-5s (more frequent)
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},  # stronger pushes
    )


# ---------------------------------------------------------------------------
# Commands — wider velocity range for agile maneuvers
# ---------------------------------------------------------------------------
@configclass
class AgileCommandsCfg(_module.CommandsCfg):
    """Wider velocity commands for agile locomotion."""

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
            lin_vel_x=(-0.8, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )


# ---------------------------------------------------------------------------
# Environment Configs
# ---------------------------------------------------------------------------
@configclass
class RobotEnvCfg(_UpstreamRobotEnvCfg):
    """AGILE locomotion training environment.

    Overrides rewards, events, and commands from upstream 29dof config.
    Scene, observations, actions, terminations, and curriculum inherited.
    """

    rewards: RewardsCfg = RewardsCfg()
    events: AgileEventCfg = AgileEventCfg()
    commands: AgileCommandsCfg = AgileCommandsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
