"""
G1 + Gripper (UNITREE_G1) Configuration for GROOT N1.6 Training

This configuration uses the PRE-REGISTERED UNITREE_G1 embodiment tag from
Isaac-GR00T. No custom modality config file is needed — the UNITREE_G1 tag
automatically loads the correct modality config.

Joint Layout (31 DOF state / 23 DOF action):

State (observation.state, 31 DOF):
  left_leg:   6 DOF (indices  0- 5) — hip pitch/roll/yaw, knee, ankle pitch/roll
  right_leg:  6 DOF (indices  6-11) — same as left
  waist:      3 DOF (indices 12-14) — yaw, roll, pitch
  left_arm:   7 DOF (indices 15-21) — shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
  right_arm:  7 DOF (indices 22-28) — same as left
  left_hand:  1 DOF (index   29)    — gripper (binary open/close)
  right_hand: 1 DOF (index   30)    — gripper (binary open/close)

Action (23 DOF):
  left_arm:            7 DOF (indices  0- 6) — RELATIVE
  right_arm:           7 DOF (indices  7-13) — RELATIVE
  left_hand:           1 DOF (index   14)    — ABSOLUTE (gripper)
  right_hand:          1 DOF (index   15)    — ABSOLUTE (gripper)
  waist:               3 DOF (indices 16-18) — ABSOLUTE (yaw, pitch, roll)
  base_height_command: 1 DOF (index   19)    — ABSOLUTE
  navigate_command:    3 DOF (indices 20-22) — ABSOLUTE (VX, VY, AngZ)

Camera: 1 ego-view (observation.images.ego_view), mapped from cam_left_high
Action horizon: 30 steps (official UNITREE_G1 default)

Source datasets (unitreerobotics HuggingFace):
  - G1_Fold_Towel       (200 episodes)
  - G1_Clean_Table       (200 episodes)
  - G1_Wipe_Table        (200 episodes)
  - G1_Prepare_Fruit     (200 episodes)
  - G1_Pour_Medicine     (200 episodes)
  - G1_Organize_Tools    (200 episodes)
  - G1_Pack_PingPong     (200 episodes)

Usage:
    # No --modality-config-path needed — UNITREE_G1 is pre-registered
    python -m dm_isaac_g1.finetuning.launcher \\
        --datasets /workspace/datasets/groot/G1_Fold_Towel \\
        --output /workspace/checkpoints/groot-g1-gripper-fold-towel \\
        --embodiment-tag UNITREE_G1

    # Or directly with Isaac-GR00T:
    conda run --no-capture-output -n unitree_sim_env \\
    torchrun --nproc_per_node=1 --master_port=29500 \\
        gr00t/experiment/launch_finetune.py \\
        --base_model_path nvidia/GR00T-N1.6-3B \\
        --dataset_path /workspace/datasets/groot/G1_Fold_Towel \\
        --embodiment_tag UNITREE_G1 \\
        --output_dir /workspace/checkpoints/groot-g1-gripper-fold-towel \\
        --max_steps 10000 \\
        --save_steps 2000 \\
        --save_total_limit 2 \\
        --global_batch_size 64 \\
        --learning_rate 1e-4

Note:
    Unlike g1_inspire_53dof.py and g1_dex3_28dof.py which use NEW_EMBODIMENT
    and require a custom modality config, UNITREE_G1 is pre-registered in
    Isaac-GR00T's embodiment configs. This file is documentation-only — it
    is NOT loaded or registered during training.
"""

# This config is NOT registered — UNITREE_G1 is already pre-registered in
# Isaac-GR00T. This file documents the expected layout for reference.

# State layout (31 DOF)
STATE_LAYOUT = {
    "left_leg":   {"start": 0,  "end": 6,  "dof": 6},
    "right_leg":  {"start": 6,  "end": 12, "dof": 6},
    "waist":      {"start": 12, "end": 15, "dof": 3},
    "left_arm":   {"start": 15, "end": 22, "dof": 7},
    "right_arm":  {"start": 22, "end": 29, "dof": 7},
    "left_hand":  {"start": 29, "end": 30, "dof": 1},
    "right_hand": {"start": 30, "end": 31, "dof": 1},
}
TOTAL_STATE_DOF = 31

# Action layout (23 DOF)
ACTION_LAYOUT = {
    "left_arm":            {"start": 0,  "end": 7,  "dof": 7,  "rep": "RELATIVE"},
    "right_arm":           {"start": 7,  "end": 14, "dof": 7,  "rep": "RELATIVE"},
    "left_hand":           {"start": 14, "end": 15, "dof": 1,  "rep": "ABSOLUTE"},
    "right_hand":          {"start": 15, "end": 16, "dof": 1,  "rep": "ABSOLUTE"},
    "waist":               {"start": 16, "end": 19, "dof": 3,  "rep": "ABSOLUTE"},
    "base_height_command": {"start": 19, "end": 20, "dof": 1,  "rep": "ABSOLUTE"},
    "navigate_command":    {"start": 20, "end": 23, "dof": 3,  "rep": "ABSOLUTE"},
}
TOTAL_ACTION_DOF = 23

# Training hyperparameters (validated on Blackwell RTX PRO 6000, 98 GB VRAM)
RECOMMENDED_HYPERPARAMS = {
    "max_steps": 10000,
    "save_steps": 2000,
    "save_total_limit": 2,
    "global_batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_ratio": 0.05,
    "dataloader_num_workers": 4,
    "color_jitter": True,
}
