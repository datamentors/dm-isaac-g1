"""Centralized robot configurations for Unitree G1.

Single source of truth for joint names, DOF counts, actuator specs,
value conversion functions, and hand type definitions.

Source: unitree_sim_isaaclab/robots/unitree.py (Isaac Lab ArticulationCfg)
        unitree_sim_isaaclab/tools/data_convert.py (gripper value mapping)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# G1 Body joints (29 DOF) — shared across all hand variants
# =============================================================================

G1_BODY_JOINT_NAMES: Dict[str, List[str]] = {
    "left_leg": [
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "right_leg": [
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

G1_BODY_DOF = 29

# Body part index ranges in the full state vector (body-only, no hands)
G1_BODY_INDEX_RANGES: Dict[str, Tuple[int, int]] = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
}


# =============================================================================
# Hand type definitions
# =============================================================================

@dataclass
class HandType:
    """Definition of a hand/gripper type for Unitree G1.

    Attributes:
        name: Short identifier (dex1, dex3, inspire).
        dof_per_hand: Degrees of freedom per hand.
        joint_names: Joint names per side {"left": [...], "right": [...]}.
        joint_type: "prismatic" or "revolute".
        joint_range: Physical joint limits (min, max). For revolute hands
            with varying ranges per joint, this is a representative range.
        stiffness: Default actuator stiffness (kp).
        damping: Default actuator damping (kd).
        friction: Actuator friction (Dex1 only).
        effort_limit: Torque/force limit per joint.
        velocity_limit: Velocity limit per joint.
        armature: Reflected inertia.
        training_range: Value range in GROOT training data space.
    """
    name: str
    dof_per_hand: int
    joint_names: Dict[str, List[str]]
    joint_type: str  # "prismatic" | "revolute"
    joint_range: Tuple[float, float]
    stiffness: float
    damping: float
    friction: float = 0.0
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None
    armature: Optional[float] = None
    training_range: Tuple[float, float] = (0.0, 5.4)

    @property
    def total_dof(self) -> int:
        return self.dof_per_hand * 2

    @property
    def all_joint_names(self) -> List[str]:
        return self.joint_names.get("left", []) + self.joint_names.get("right", [])


# --- Dex1: Prismatic parallel-jaw grippers (2 DOF/hand) ---
# Source: unitree_sim_isaaclab G129_CFG_WITH_DEX1_BASE_FIX
DEX1 = HandType(
    name="dex1",
    dof_per_hand=2,
    joint_names={
        "left": ["left_hand_Joint1_1", "left_hand_Joint2_1"],
        "right": ["right_hand_Joint1_1", "right_hand_Joint2_1"],
    },
    joint_type="prismatic",
    joint_range=(-0.02, 0.0245),
    stiffness=800.0,
    damping=3.0,
    friction=200.0,
    training_range=(0.0, 5.4),
)

# --- Dex3: 3-finger dexterous hands (7 DOF/hand) ---
# Source: unitree_sim_isaaclab G129_CFG_WITH_DEX3_BASE_FIX
DEX3 = HandType(
    name="dex3",
    dof_per_hand=7,
    joint_names={
        "left": [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ],
        "right": [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ],
    },
    joint_type="revolute",
    joint_range=(0.0, 1.57),  # representative range (varies per joint)
    stiffness=100.0,
    damping=10.0,
    armature=0.1,
    effort_limit=300.0,
    velocity_limit=100.0,
)

# --- Inspire: 12 DOF dexterous hands ---
# Source: unitree_sim_isaaclab G129_CFG_WITH_INSPIRE_HAND
INSPIRE = HandType(
    name="inspire",
    dof_per_hand=12,
    joint_names={
        "left": [
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
        ],
        "right": [
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
        ],
    },
    joint_type="revolute",
    joint_range=(0.0, 1.57),  # representative range (varies per joint)
    stiffness=1000.0,
    damping=15.0,
    armature=0.0,
    effort_limit=100.0,
    velocity_limit=50.0,
)

# --- "Gripper" is an alias for Dex1 ---
# The pre-registered UNITREE_G1 embodiment in GROOT uses the Dex1 prismatic
# gripper. Training datasets (G1_Fold_Towel, etc.) all use this hand type.
# GROOT sees 1 DOF/hand (single value in [0, 5.4]) even though Dex1 has
# 2 physical joints/hand (Joint1_1 and Joint2_1 mirror each other).
GRIPPER = DEX1

# --- Hand type registry ---
HAND_TYPES: Dict[str, Optional[HandType]] = {
    "dex1": DEX1,
    "gripper": DEX1,  # alias — same hardware
    "dex3": DEX3,
    "inspire": INSPIRE,
    "none": None,
}


# =============================================================================
# Actuator specifications — per body group
# =============================================================================

@dataclass
class ActuatorSpec:
    """Actuator parameters for a joint group.

    Source: unitree_sim_isaaclab/robots/unitree.py
    """
    stiffness: Dict[str, float]
    damping: Dict[str, float]
    effort_limit: Optional[Dict[str, float]] = None
    velocity_limit: Optional[Dict[str, float]] = None
    armature: Optional[float] = None


# "base_fix" variant — simpler PD control, used for manipulation tasks
G1_ACTUATORS_BASE_FIX: Dict[str, ActuatorSpec] = {
    "legs": ActuatorSpec(
        stiffness={},  # uses USD defaults
        damping={},
    ),
    "waist": ActuatorSpec(
        stiffness={
            "waist_yaw_joint": 10000.0,
            "waist_roll_joint": 10000.0,
            "waist_pitch_joint": 10000.0,
        },
        damping={
            "waist_yaw_joint": 10000.0,
            "waist_roll_joint": 10000.0,
            "waist_pitch_joint": 10000.0,
        },
        effort_limit={"*": 1000.0},
    ),
    "feet": ActuatorSpec(
        stiffness={},
        damping={},
    ),
    "arms": ActuatorSpec(
        stiffness={
            "shoulder": 300.0,
            "elbow": 400.0,
            "wrist": 400.0,
        },
        damping={
            "shoulder": 3.0,
            "elbow": 2.5,
            "wrist": 2.5,
        },
    ),
}

# "wholebody" variant — detailed limits, used for WBC locomotion + manipulation
G1_ACTUATORS_WHOLEBODY: Dict[str, ActuatorSpec] = {
    "legs": ActuatorSpec(
        stiffness={
            "hip_yaw": 150.0,
            "hip_roll": 150.0,
            "hip_pitch": 200.0,
            "knee": 200.0,
            "waist": 200.0,
        },
        damping={
            "hip_yaw": 5.0,
            "hip_roll": 5.0,
            "hip_pitch": 5.0,
            "knee": 5.0,
            "waist": 5.0,
        },
        effort_limit={
            "hip_yaw": 88.0,
            "hip_roll": 139.0,
            "hip_pitch": 88.0,
            "knee": 139.0,
            "waist_yaw": 88.0,
            "waist_roll": 35.0,
            "waist_pitch": 35.0,
        },
        velocity_limit={
            "hip_yaw": 32.0,
            "hip_roll": 20.0,
            "hip_pitch": 32.0,
            "knee": 20.0,
            "waist_yaw": 32.0,
            "waist_roll": 30.0,
            "waist_pitch": 30.0,
        },
        armature=0.01,
    ),
    "feet": ActuatorSpec(
        stiffness={"ankle_pitch": 20.0, "ankle_roll": 20.0},
        damping={"ankle_pitch": 2.0, "ankle_roll": 2.0},
        effort_limit={"ankle_pitch": 35.0, "ankle_roll": 35.0},
        velocity_limit={"ankle_pitch": 30.0, "ankle_roll": 30.0},
        armature=0.01,
    ),
    "shoulders": ActuatorSpec(
        stiffness={"shoulder_pitch": 100.0, "shoulder_roll": 100.0},
        damping={"shoulder_pitch": 2.0, "shoulder_roll": 2.0},
        effort_limit={"shoulder_pitch": 25.0, "shoulder_roll": 25.0},
        velocity_limit={"shoulder_pitch": 37.0, "shoulder_roll": 37.0},
        armature=0.01,
    ),
    "arms": ActuatorSpec(
        stiffness={"shoulder_yaw": 50.0, "elbow": 50.0},
        damping={"shoulder_yaw": 2.0, "elbow": 2.0},
        effort_limit={"shoulder_yaw": 25.0, "elbow": 25.0},
        velocity_limit={"shoulder_yaw": 37.0, "elbow": 37.0},
        armature=0.01,
    ),
    "wrist": ActuatorSpec(
        stiffness={"wrist_yaw": 40.0, "wrist_roll": 40.0, "wrist_pitch": 40.0},
        damping={"wrist_yaw": 2.0, "wrist_roll": 2.0, "wrist_pitch": 2.0},
        effort_limit={"wrist_yaw": 5.0, "wrist_roll": 25.0, "wrist_pitch": 5.0},
        velocity_limit={"wrist_yaw": 22.0, "wrist_roll": 37.0, "wrist_pitch": 22.0},
        armature=0.01,
    ),
}


# =============================================================================
# Value conversion: Dex1 physical <-> training space
# =============================================================================
# Source: unitree_sim_isaaclab/tools/data_convert.py
#
# Physical Dex1 range: [-0.02, 0.024] meters
# Training data range: [0.0, 5.4] (inverted: -0.02=open→5.4, 0.024=closed→0.0)

DEX1_PHYSICAL_MIN = -0.02   # fully open
DEX1_PHYSICAL_MAX = 0.024   # fully closed (note: URDF limit is 0.0245)
DEX1_TRAINING_MIN = 0.0     # fully closed
DEX1_TRAINING_MAX = 5.4     # fully open


def dex1_physical_to_training(value: float) -> float:
    """Convert Dex1 physical joint position to training-data space.

    Physical [-0.02, 0.024] -> Training [5.4, 0.0]
    (inverted: open in physical = high in training)

    Args:
        value: Physical joint position in meters.

    Returns:
        Training-data value in [0.0, 5.4] range.
    """
    try:
        value = round(float(value), 3)
    except Exception:
        pass
    # Clamp to physical range
    value = max(DEX1_PHYSICAL_MIN, min(DEX1_PHYSICAL_MAX, value))
    # Linear map (inverted)
    result = DEX1_TRAINING_MIN + (DEX1_TRAINING_MAX - DEX1_TRAINING_MIN) * (
        DEX1_PHYSICAL_MAX - value
    ) / (DEX1_PHYSICAL_MAX - DEX1_PHYSICAL_MIN)
    return round(result, 3)


def dex1_training_to_physical(value: float) -> float:
    """Convert training-data value to Dex1 physical joint position.

    Training [0.0, 5.4] -> Physical [0.024, -0.02]
    (0.0=closed=0.024m, 5.4=open=-0.02m)

    Args:
        value: Training-data value in [0.0, 5.4] range.

    Returns:
        Physical joint position in meters.
    """
    value = max(DEX1_TRAINING_MIN, min(DEX1_TRAINING_MAX, float(value)))
    return DEX1_PHYSICAL_MAX + (DEX1_PHYSICAL_MIN - DEX1_PHYSICAL_MAX) * (
        value - DEX1_TRAINING_MIN
    ) / (DEX1_TRAINING_MAX - DEX1_TRAINING_MIN)


# =============================================================================
# GROOT UNITREE_G1 state/action layout
# =============================================================================
# The pre-registered UNITREE_G1 embodiment uses 31 DOF state, 23 DOF action.

GROOT_STATE_LAYOUT: Dict[str, Tuple[int, int]] = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (22, 29),
    "left_hand": (29, 30),
    "right_hand": (30, 31),
}
GROOT_STATE_DOF = 31
GROOT_ACTION_DOF = 23

# GROOT arm action order (training order — yaw, roll, pitch for wrists)
# vs WBC/menagerie order (roll, pitch, yaw).
# Maps GROOT action index -> WBC joint index within a 7-joint arm.
GROOT_TO_WBC_ARM_REMAP = [0, 1, 2, 3, 6, 4, 5]

# WBC joint ordering (first 15 actuated joints)
WBC_JOINT_NAMES = [
    # Left leg (0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

WBC_NUM_ACTIONS = 15  # Lower body DOF controlled by WBC
WBC_ARM_START = 15    # Index where arms start in WBC joint list
WBC_ARM_COUNT = 14    # 7 per arm


# =============================================================================
# Initial pose defaults
# =============================================================================

# Default initial joint positions for base_fix variant
G1_INIT_POS_BASE_FIX: Dict[str, float] = {
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.15,
}

# Default initial joint positions for wholebody variant
G1_INIT_POS_WHOLEBODY: Dict[str, float] = {
    "left_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_hip_pitch_joint": -0.20,
    "right_knee_joint": 0.42,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_joint": 0.87,
    "right_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_pitch_joint": 0.35,
}


# =============================================================================
# Complete robot configuration
# =============================================================================

@dataclass
class G1RobotConfig:
    """Complete G1 robot configuration with a specific hand type.

    Attributes:
        hand_type: The hand type (or None for no hands).
        actuator_variant: "base_fix" or "wholebody".
        body_dof: Body DOF (always 29 for G1).
    """
    hand_type: Optional[HandType] = None
    actuator_variant: str = "base_fix"
    body_dof: int = 29

    @property
    def total_dof(self) -> int:
        if self.hand_type is None:
            return self.body_dof
        return self.body_dof + self.hand_type.total_dof

    @property
    def hand_dof(self) -> int:
        if self.hand_type is None:
            return 0
        return self.hand_type.total_dof

    @property
    def all_joint_names(self) -> List[str]:
        """All joint names in order (body + hands)."""
        names: List[str] = []
        for part in ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]:
            names.extend(G1_BODY_JOINT_NAMES[part])
        if self.hand_type is not None:
            names.extend(self.hand_type.joint_names.get("left", []))
            names.extend(self.hand_type.joint_names.get("right", []))
        return names

    @property
    def joint_index_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Index ranges for each body part in the full state vector."""
        ranges = dict(G1_BODY_INDEX_RANGES)
        offset = self.body_dof
        if self.hand_type is not None:
            n = self.hand_type.dof_per_hand
            ranges["left_hand"] = (offset, offset + n)
            ranges["right_hand"] = (offset + n, offset + 2 * n)
        return ranges

    @property
    def actuators(self) -> Dict[str, ActuatorSpec]:
        if self.actuator_variant == "wholebody":
            return G1_ACTUATORS_WHOLEBODY
        return G1_ACTUATORS_BASE_FIX


# Pre-built configurations matching unitree_sim_isaaclab/robots/unitree.py
G1_DEX1 = G1RobotConfig(hand_type=DEX1, actuator_variant="base_fix")
G1_DEX1_WHOLEBODY = G1RobotConfig(hand_type=DEX1, actuator_variant="wholebody")
G1_GRIPPER = G1_DEX1  # alias — UNITREE_G1 pre-registered embodiment uses Dex1
G1_DEX3 = G1RobotConfig(hand_type=DEX3, actuator_variant="base_fix")
G1_DEX3_WHOLEBODY = G1RobotConfig(hand_type=DEX3, actuator_variant="wholebody")
G1_INSPIRE = G1RobotConfig(hand_type=INSPIRE, actuator_variant="base_fix")
G1_INSPIRE_WHOLEBODY = G1RobotConfig(hand_type=INSPIRE, actuator_variant="wholebody")
G1_NO_HANDS = G1RobotConfig(hand_type=None)
