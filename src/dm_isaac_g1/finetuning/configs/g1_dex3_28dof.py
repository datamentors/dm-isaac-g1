"""
G1 + Dex3 Hand Native Configuration for GROOT N1.6 Training

This configuration fine-tunes on native Dex3 datasets exactly as downloaded
from HuggingFace — no format conversion. The raw datasets have a flat 28 DOF
observation.state and action column.

Joint Layout (28 DOF — flat vector in observation.state / action):
  Indices  0- 6: Left arm   [ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw]
  Indices  7-13: Right arm  [ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw]
  Indices 14-20: Left Dex3  [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1]
  Indices 21-27: Right Dex3 [Thumb0, Thumb1, Thumb2, Index0, Index1, Middle0, Middle1]

Source datasets (unitreerobotics HuggingFace):
  - G1_Dex3_BlockStacking_Dataset  (301 episodes, 281k frames)
  - G1_Dex3_ToastedBread_Dataset

Cameras (4): cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist (480x640, AV1)
Note: This config specifies cam_left_high for the dataset loader, but the base
model's NEW_EMBODIMENT processor config includes all 4 cameras during training.

Usage (on workstation):
    # Copy config to Isaac-GR00T workspace
    cp /workspace/dm-isaac-g1/src/dm_isaac_g1/finetuning/configs/g1_dex3_28dof.py \\
       /workspace/Isaac-GR00T/g1_dex3_28dof_config.py

    # Run fine-tuning
    conda run --no-capture-output -n unitree_sim_env \\
    python /workspace/Isaac-GR00T/gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.6-3B \\
        --dataset-path /workspace/datasets/G1_Dex3_Combined \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path /workspace/Isaac-GR00T/g1_dex3_28dof_config.py \\
        --output-dir /workspace/checkpoints/groot_g1_dex3_28dof \\
        --max-steps 10000 \\
        --save-steps 1000 \\
        --save-total-limit 2 \\
        --global-batch-size 8 \\
        --dataloader-num-workers 0 \\
        --num-gpus 1
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
# GROOT Modality Configuration
# =============================================================================
# Uses single flat keys matching raw parquet column names.
# The loader reads observation.state (28 floats) and action (28 floats) directly.

g1_dex3_28dof_config = {
    # cam_left_high is the primary camera in both Dex3 datasets
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.images.cam_left_high"],
    ),

    # observation.state: flat 28 DOF vector (arms + Dex3 hands)
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.state"],
    ),

    # action: flat 28 DOF vector, 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # Language task description
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

# =============================================================================
# Register Configuration
# =============================================================================

register_modality_config(
    g1_dex3_28dof_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

# =============================================================================
# Reference: DOF breakdown (for documentation only, not used by loader)
# Centralized Dex3 joint definitions: dm_isaac_g1.core.robot_configs.DEX3
# =============================================================================

from dm_isaac_g1.core.robot_configs import DEX3, G1_BODY_JOINT_NAMES

# Flat index: joint name from raw dataset info.json
# Note: Dataset uses "k" prefix naming, robot_configs uses Isaac Lab naming.
# The mapping is: kLeftShoulderPitch -> left_shoulder_pitch_joint, etc.
G1_DEX3_DOF_LAYOUT = {
    # Flat index: joint name from raw dataset info.json
    0:  "kLeftShoulderPitch",
    1:  "kLeftShoulderRoll",
    2:  "kLeftShoulderYaw",
    3:  "kLeftElbow",
    4:  "kLeftWristRoll",
    5:  "kLeftWristPitch",
    6:  "kLeftWristYaw",
    7:  "kRightShoulderPitch",
    8:  "kRightShoulderRoll",
    9:  "kRightShoulderYaw",
    10: "kRightElbow",
    11: "kRightWristRoll",
    12: "kRightWristPitch",
    13: "kRightWristYaw",
    14: "kLeftHandThumb0",
    15: "kLeftHandThumb1",
    16: "kLeftHandThumb2",
    17: "kLeftHandMiddle0",
    18: "kLeftHandMiddle1",
    19: "kLeftHandIndex0",
    20: "kLeftHandIndex1",
    21: "kRightHandThumb0",
    22: "kRightHandThumb1",
    23: "kRightHandThumb2",
    24: "kRightHandIndex0",
    25: "kRightHandIndex1",
    26: "kRightHandMiddle0",
    27: "kRightHandMiddle1",
}

# Isaac Lab joint names (from centralized config)
G1_DEX3_JOINT_NAMES = {
    "left_arm": G1_BODY_JOINT_NAMES["left_arm"],
    "right_arm": G1_BODY_JOINT_NAMES["right_arm"],
    "left_dex3_hand": DEX3.joint_names["left"],
    "right_dex3_hand": DEX3.joint_names["right"],
}
