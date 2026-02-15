"""Robot definitions for G1 + Inspire Hands.

Defines the joint configuration for the Unitree G1 EDU 2 robot
with Inspire Robotics Dexterous Hands (53 DOF total).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Complete joint names for G1 + Inspire (53 DOF)
G1_INSPIRE_JOINT_NAMES: Dict[str, List[str]] = {
    # Legs (12 DOF)
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
    # Waist (3 DOF)
    "waist": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
    # Arms (14 DOF)
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
    # Inspire Hands (24 DOF)
    "left_inspire_hand": [
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
    "right_inspire_hand": [
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
}

# Index ranges in the 53 DOF state vector
JOINT_INDEX_RANGES: Dict[str, Tuple[int, int]] = {
    "left_leg": (0, 6),  # indices 0-5
    "right_leg": (6, 12),  # indices 6-11
    "waist": (12, 15),  # indices 12-14
    "left_arm": (15, 22),  # indices 15-21
    "right_arm": (22, 29),  # indices 22-28
    "left_inspire_hand": (29, 41),  # indices 29-40
    "right_inspire_hand": (41, 53),  # indices 41-52
}


@dataclass
class G1InspireRobot:
    """G1 EDU 2 robot with Inspire Robotics Dexterous Hands.

    Attributes:
        body_dof: Degrees of freedom for body (legs + waist + arms).
        hand_dof_per_hand: Degrees of freedom per Inspire hand.
        total_dof: Total degrees of freedom.
        joint_names: Dictionary mapping body parts to joint names.
        joint_indices: Dictionary mapping body parts to index ranges.
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
        """Get the index range for a body part.

        Args:
            body_part: Name of body part (e.g., "left_arm", "right_inspire_hand").

        Returns:
            Tuple of (start_index, end_index).

        Raises:
            KeyError: If body part not found.
        """
        return self.joint_indices[body_part]

    def get_joint_names(self, body_part: str) -> List[str]:
        """Get joint names for a body part.

        Args:
            body_part: Name of body part.

        Returns:
            List of joint names.
        """
        return self.joint_names[body_part]

    def validate_state_vector(self, state: List[float]) -> bool:
        """Validate that a state vector has correct dimensions.

        Args:
            state: State vector to validate.

        Returns:
            True if valid, False otherwise.
        """
        return len(state) == self.total_dof

    @property
    def all_joint_names(self) -> List[str]:
        """Get all joint names in order."""
        names = []
        for part in [
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_inspire_hand",
            "right_inspire_hand",
        ]:
            names.extend(self.joint_names[part])
        return names
