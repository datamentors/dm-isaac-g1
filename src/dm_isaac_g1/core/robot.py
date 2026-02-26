"""Robot definitions for Unitree G1.

Provides the G1Robot class (generic, works with any hand type) and
backward-compatible G1InspireRobot alias.

All joint names, hand types, and actuator specs are defined in robot_configs.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dm_isaac_g1.core.robot_configs import (
    G1_BODY_JOINT_NAMES,
    G1_BODY_DOF,
    G1_BODY_INDEX_RANGES,
    DEX1,
    DEX3,
    INSPIRE,
    HAND_TYPES,
    HandType,
    G1RobotConfig,
)


# Legacy constants (kept for backward compatibility with finetuning configs)
G1_INSPIRE_JOINT_NAMES: Dict[str, List[str]] = {
    **G1_BODY_JOINT_NAMES,
    "left_inspire_hand": INSPIRE.joint_names["left"],
    "right_inspire_hand": INSPIRE.joint_names["right"],
}

JOINT_INDEX_RANGES: Dict[str, Tuple[int, int]] = {
    **G1_BODY_INDEX_RANGES,
    "left_inspire_hand": (29, 41),
    "right_inspire_hand": (41, 53),
}


@dataclass
class G1Robot:
    """Unitree G1 robot with configurable hand type.

    Attributes:
        hand_type: The hand type (DEX1, DEX3, INSPIRE, or None).
        body_dof: Body DOF (always 29).
    """

    hand_type: Optional[HandType] = None
    body_dof: int = G1_BODY_DOF

    @property
    def hand_dof_per_hand(self) -> int:
        return self.hand_type.dof_per_hand if self.hand_type else 0

    @property
    def total_dof(self) -> int:
        return self.body_dof + (self.hand_type.total_dof if self.hand_type else 0)

    @property
    def joint_names(self) -> Dict[str, List[str]]:
        names = dict(G1_BODY_JOINT_NAMES)
        if self.hand_type:
            names["left_hand"] = self.hand_type.joint_names["left"]
            names["right_hand"] = self.hand_type.joint_names["right"]
        return names

    @property
    def joint_indices(self) -> Dict[str, Tuple[int, int]]:
        indices = dict(G1_BODY_INDEX_RANGES)
        if self.hand_type:
            n = self.hand_type.dof_per_hand
            offset = self.body_dof
            indices["left_hand"] = (offset, offset + n)
            indices["right_hand"] = (offset + n, offset + 2 * n)
        return indices

    def get_joint_indices(self, body_part: str) -> Tuple[int, int]:
        """Get the index range for a body part."""
        return self.joint_indices[body_part]

    def get_joint_names(self, body_part: str) -> List[str]:
        """Get joint names for a body part."""
        return self.joint_names[body_part]

    def validate_state_vector(self, state: List[float]) -> bool:
        """Validate that a state vector has correct dimensions."""
        return len(state) == self.total_dof

    @property
    def all_joint_names(self) -> List[str]:
        """Get all joint names in order."""
        names: List[str] = []
        for part in ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]:
            names.extend(G1_BODY_JOINT_NAMES[part])
        if self.hand_type:
            names.extend(self.hand_type.joint_names["left"])
            names.extend(self.hand_type.joint_names["right"])
        return names


@dataclass
class G1InspireRobot:
    """G1 EDU 2 robot with Inspire Robotics Dexterous Hands (53 DOF).

    Backward-compatible wrapper around G1Robot(hand_type=INSPIRE).
    """

    body_dof: int = 29
    hand_dof_per_hand: int = 12
    total_dof: int = 53

    joint_names: Dict[str, List[str]] = field(
        default_factory=lambda: G1_INSPIRE_JOINT_NAMES.copy()
    )
    joint_indices: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: JOINT_INDEX_RANGES.copy()
    )

    def get_joint_indices(self, body_part: str) -> Tuple[int, int]:
        """Get the index range for a body part."""
        return self.joint_indices[body_part]

    def get_joint_names(self, body_part: str) -> List[str]:
        """Get joint names for a body part."""
        return self.joint_names[body_part]

    def validate_state_vector(self, state: List[float]) -> bool:
        """Validate that a state vector has correct dimensions."""
        return len(state) == self.total_dof

    @property
    def all_joint_names(self) -> List[str]:
        """Get all joint names in order."""
        names = []
        for part in [
            "left_leg", "right_leg", "waist",
            "left_arm", "right_arm",
            "left_inspire_hand", "right_inspire_hand",
        ]:
            names.extend(self.joint_names[part])
        return names
