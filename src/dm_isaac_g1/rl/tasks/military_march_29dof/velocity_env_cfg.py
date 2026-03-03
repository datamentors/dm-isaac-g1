# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""29-DOF military march velocity-tracking environment config v2.

Overrides reward weights from upstream unitree_rl_lab for more natural walking:
- Enable arm_leg_coordination for contralateral arm swing
- Reduce arm deviation penalty so arms can actually swing
- Increase feet clearance target for higher steps
- Reduce base_height penalty to avoid forced crouching
- Reduce termination penalty to encourage exploration
- Increase gait reward for stronger periodic stepping signal
"""

import importlib
import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Load the upstream module (directory starts with digit, needs importlib)
_module = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof_military_march.velocity_env_cfg"
)

# Import mdp from upstream (has all reward functions including arm_leg_coordination)
_mdp_module = importlib.import_module("unitree_rl_lab.tasks.locomotion.mdp")
mdp = _mdp_module

# Import base classes
_UpstreamRobotEnvCfg = _module.RobotEnvCfg
_UpstreamRobotPlayEnvCfg = _module.RobotPlayEnvCfg


@configclass
class RewardsCfg:
    """Reward terms v2 — tuned for more natural military march walking.

    Key changes from v1:
    1. arm_leg_coordination ENABLED (+0.3) — the signature arm swing
    2. joint_deviation_arms REDUCED (-0.1 -> -0.02) — let arms swing freely
    3. feet_clearance target INCREASED (0.1 -> 0.15m) — higher steps
    4. base_height penalty REDUCED (-10.0 -> -5.0) — less forced crouching
    5. termination_penalty REDUCED (-200 -> -100) — less conservative
    6. gait weight INCREASED (0.5 -> 0.8) — stronger periodic signal
    7. track_lin_vel_xy INCREASED (1.5 -> 2.0) — stronger forward drive
    8. feet_air_time INCREASED (0.25 -> 0.4) — reward longer swing phase
    """

    # -- VELOCITY TRACKING (stronger forward drive)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,  # was 1.5
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Feet air time — increased for longer swing phase
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.4,  # was 0.25
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # Termination — reduced from -200 to -100 for less conservative exploration
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # -- Base penalties
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)

    # Arms — greatly reduced penalty so arms can swing for coordination
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # was -0.1 (5x reduction)
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # Base height — reduced penalty to avoid forced crouching
    base_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.78})  # was -10.0

    # Gait — stronger periodic signal for clear stepping pattern
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.8,  # was 0.5
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
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # Feet clearance — higher target for visible step height
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,  # was 0.1 — higher steps for military march
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={"threshold": 0.4, "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )

    air_time_symmetry = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )

    lateral_sway = RewTerm(
        func=mdp.lateral_velocity_penalty,
        weight=-0.3,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    # NEW: Arm-leg coordination — contralateral swing (military march signature)
    arm_leg_coordination = RewTerm(
        func=mdp.arm_leg_coordination,
        weight=-0.3,  # negative because function returns penalty for SAME-sign motion
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "left_arm_joints": ["left_shoulder_pitch_joint"],
            "right_arm_joints": ["right_shoulder_pitch_joint"],
            "left_leg_joints": ["left_hip_pitch_joint"],
            "right_leg_joints": ["right_hip_pitch_joint"],
        },
    )


@configclass
class RobotEnvCfg(_UpstreamRobotEnvCfg):
    """Override only the rewards — everything else inherited from upstream."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
