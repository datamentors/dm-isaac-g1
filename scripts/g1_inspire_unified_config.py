"""
G1 + Inspire Hand Unified Configuration for GROOT N1.6 Training

This configuration defines the 53 DOF embodiment for the Unitree G1 EDU 2
robot with Inspire Robotics Dexterous Hands.

Joint Layout (53 DOF total):
- Left Leg:  6 DOF (indices 0-5)
- Right Leg: 6 DOF (indices 6-11)
- Waist:     3 DOF (indices 12-14)
- Left Arm:  7 DOF (indices 15-21)
- Right Arm: 7 DOF (indices 22-28)
- Left Hand: 12 DOF (indices 29-40)  - Inspire
- Right Hand: 12 DOF (indices 41-52) - Inspire

Usage:
    Copy this file to /workspace/Isaac-GR00T/ and use with training:

    python gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-path /workspace/datasets/G1_Inspire_Combined \
        --embodiment-tag G1_INSPIRE_53DOF \
        --modality-config-path /workspace/Isaac-GR00T/g1_inspire_unified_config.py \
        --output-dir /workspace/checkpoints/groot_g1_inspire \
        --max-steps 10000
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)
from gr00t.data.embodiment_tags import EmbodimentTag

# =============================================================================
# Joint Configuration
# =============================================================================

# Complete joint names for documentation and validation
G1_INSPIRE_JOINT_NAMES = {
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
JOINT_INDEX_RANGES = {
    "left_leg": (0, 6),           # indices 0-5
    "right_leg": (6, 12),         # indices 6-11
    "waist": (12, 15),            # indices 12-14
    "left_arm": (15, 22),         # indices 15-21
    "right_arm": (22, 29),        # indices 22-28
    "left_inspire_hand": (29, 41),  # indices 29-40
    "right_inspire_hand": (41, 53), # indices 41-52
}

# =============================================================================
# GROOT Modality Configuration
# =============================================================================

g1_inspire_unified_config = {
    # Video modality - uses camera from the dataset
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.images.cam_left_high"],  # Primary camera
    ),

    # State modality - 53 DOF observation
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",          # 6 DOF
            "right_leg",         # 6 DOF
            "waist",             # 3 DOF
            "left_arm",          # 7 DOF
            "right_arm",         # 7 DOF
            "left_inspire_hand", # 12 DOF
            "right_inspire_hand", # 12 DOF
        ],
    ),

    # Action modality - 53 DOF actions with 16-step horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_inspire_hand",
            "right_inspire_hand",
        ],
        action_configs=[
            # Each body part uses absolute position control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ] * 7,  # One config per modality_key
    ),

    # Language modality - task description
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

# =============================================================================
# Register Configuration
# =============================================================================

# Register as a new embodiment tag
# Use EmbodimentTag.NEW_EMBODIMENT or create a custom tag
register_modality_config(
    g1_inspire_unified_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate the configuration dimensions."""
    total_dof = 0
    for key, (start, end) in JOINT_INDEX_RANGES.items():
        dof = end - start
        total_dof += dof
        joint_count = len(G1_INSPIRE_JOINT_NAMES[key])
        assert dof == joint_count, f"{key}: index range {dof} != joint count {joint_count}"

    assert total_dof == 53, f"Total DOF should be 53, got {total_dof}"
    print(f"Configuration validated: {total_dof} DOF")

    # Print summary
    print("\nJoint Configuration Summary:")
    print("-" * 40)
    for key, joints in G1_INSPIRE_JOINT_NAMES.items():
        start, end = JOINT_INDEX_RANGES[key]
        print(f"  {key}: {len(joints)} DOF (indices {start}-{end-1})")
    print("-" * 40)
    print(f"  TOTAL: {total_dof} DOF")


if __name__ == "__main__":
    validate_config()
