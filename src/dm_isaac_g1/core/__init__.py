"""Core utilities for dm-isaac-g1."""

from dm_isaac_g1.core.config import Config, load_config
from dm_isaac_g1.core.robot import (
    G1Robot,
    G1InspireRobot,
    JOINT_INDEX_RANGES,
    G1_INSPIRE_JOINT_NAMES,
)
from dm_isaac_g1.core.robot_configs import (
    # Hand types
    HandType,
    DEX1,
    DEX3,
    INSPIRE,
    GRIPPER,
    HAND_TYPES,
    # Body definitions
    G1_BODY_JOINT_NAMES,
    G1_BODY_DOF,
    G1_BODY_INDEX_RANGES,
    # Robot configs
    G1RobotConfig,
    G1_DEX1,
    G1_DEX3,
    G1_INSPIRE,
    G1_GRIPPER,
    G1_NO_HANDS,
    # Actuator specs
    ActuatorSpec,
    G1_ACTUATORS_BASE_FIX,
    G1_ACTUATORS_WHOLEBODY,
    # Value conversion
    dex1_physical_to_training,
    dex1_training_to_physical,
    # GROOT layout
    GROOT_STATE_LAYOUT,
    GROOT_STATE_DOF,
    GROOT_ACTION_DOF,
    WBC_JOINT_NAMES,
)
from dm_isaac_g1.core.remote import WorkstationConnection

__all__ = [
    # Config
    "Config",
    "load_config",
    # Robot classes
    "G1Robot",
    "G1InspireRobot",
    "JOINT_INDEX_RANGES",
    "G1_INSPIRE_JOINT_NAMES",
    # Hand types
    "HandType",
    "DEX1",
    "DEX3",
    "INSPIRE",
    "GRIPPER",
    "HAND_TYPES",
    # Body definitions
    "G1_BODY_JOINT_NAMES",
    "G1_BODY_DOF",
    "G1_BODY_INDEX_RANGES",
    # Robot configs
    "G1RobotConfig",
    "G1_DEX1",
    "G1_DEX3",
    "G1_INSPIRE",
    "G1_GRIPPER",
    "G1_NO_HANDS",
    # Actuator specs
    "ActuatorSpec",
    "G1_ACTUATORS_BASE_FIX",
    "G1_ACTUATORS_WHOLEBODY",
    # Value conversion
    "dex1_physical_to_training",
    "dex1_training_to_physical",
    # GROOT layout
    "GROOT_STATE_LAYOUT",
    "GROOT_STATE_DOF",
    "GROOT_ACTION_DOF",
    "WBC_JOINT_NAMES",
    # Remote
    "WorkstationConnection",
]
