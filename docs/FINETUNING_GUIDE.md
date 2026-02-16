# GROOT N1.6 Fine-tuning Guide

This guide covers fine-tuning the GROOT N1.6 model for G1+Inspire robot manipulation tasks.

## Overview

Fine-tuning adapts the pre-trained GROOT N1.6 model to your specific robot embodiment and tasks. The process uses NVIDIA's Isaac-GR00T framework on the workstation.

## Prerequisites

- **Workstation**: Blackwell (192.168.1.205) with NVIDIA GPU
- **Container**: `isaac-sim` with Isaac-GR00T installed
- **Dataset**: LeRobot v2 format with observations and actions
- **Model**: `nvidia/GR00T-N1.6-3B` base model

## Quick Start

```bash
# SSH to workstation
ssh datamentors@192.168.1.205

# Enter container
docker exec -it isaac-sim bash

# Activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate grootenv

# Run fine-tuning
cd /workspace/Isaac-GR00T
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/datasets/my_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/Isaac-GR00T/g1_inspire_unified_config.py \
    --output-dir /workspace/checkpoints/my_model \
    --max-steps 5000 \
    --save-steps 1000 \
    --global-batch-size 8
```

## Dataset Format

GROOT expects datasets in LeRobot v2 format:

```
dataset/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── modality.json       # Joint mapping
│   ├── episodes.jsonl      # Episode metadata
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── cam_left_high/
            └── episode_000000.mp4
```

### Required Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.state` | (53,) | Joint positions (body + arms + hands) |
| `action` | (53,) | Target joint positions |
| `observation.images.cam_left_high` | (256, 256, 3) | Camera image |
| `task` | str | Language instruction |

### G1+Inspire Joint Order (53 DOF)

```python
# Body: 29 joints
body_joints = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Left hand: 12 joints
left_hand = ["left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", "left_thumb_rot", ...]

# Right hand: 12 joints
right_hand = ["right_thumb", "right_index", "right_middle", "right_ring", "right_pinky", "right_thumb_rot", ...]
```

## Modality Config

Create a config file for your embodiment:

```python
# g1_inspire_unified_config.py
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig, ActionConfig,
    ActionRepresentation, ActionType, ActionFormat
)
from gr00t.configs.data.embodiment_configs import register_modality_config

config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_left_high"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["body", "left_arm", "right_arm", "left_gripper", "right_gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon
        modality_keys=["body", "left_arm", "right_arm", "left_gripper", "right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 5000 | Total training steps |
| `--save-steps` | 1000 | Checkpoint frequency |
| `--global-batch-size` | 8 | Batch size across all GPUs |
| `--learning-rate` | 1e-4 | Learning rate |
| `--num-gpus` | 1 | Number of GPUs |
| `--dataloader-num-workers` | 4 | Data loading workers |

### Component Tuning Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tune-llm` | False | Fine-tune language model |
| `--tune-visual` | False | Fine-tune visual encoder |
| `--tune-projector` | True | Fine-tune multimodal projector |
| `--tune-diffusion-model` | True | Fine-tune diffusion action head |

## Step-by-Step Workflow

### 1. Prepare Dataset

```bash
# Convert to GROOT format if needed
python /workspace/Isaac-GR00T/scripts/convert_g1_format.py \
    --input /path/to/original \
    --output /workspace/datasets/my_dataset_GROOT
```

### 2. Validate Dataset

```bash
# Check structure
cat /workspace/datasets/my_dataset/meta/info.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Features:', list(d['features'].keys()))
"

# Check parquet columns
python3 -c "
import pyarrow.parquet as pq
import glob
files = glob.glob('/workspace/datasets/my_dataset/data/chunk-000/*.parquet')
if files:
    t = pq.read_table(files[0])
    print('Columns:', t.column_names)
"
```

### 3. Run Test Training

```bash
# Quick test (100 steps)
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/datasets/my_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/Isaac-GR00T/g1_inspire_unified_config.py \
    --output-dir /workspace/checkpoints/test_run \
    --max-steps 100 \
    --save-steps 50 \
    --global-batch-size 8
```

### 4. Run Full Training

```bash
# Full training (background)
nohup python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/datasets/my_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/Isaac-GR00T/g1_inspire_unified_config.py \
    --output-dir /workspace/checkpoints/my_model \
    --max-steps 5000 \
    --save-steps 1000 \
    --global-batch-size 8 \
    --learning-rate 1e-4 \
    > /tmp/finetune.log 2>&1 &

# Monitor
tail -f /tmp/finetune.log
```

### 5. Upload Model

```bash
# Upload to HuggingFace
huggingface-cli upload datamentorshf/my-model \
    /workspace/checkpoints/my_model/checkpoint-5000 . \
    --repo-type model --private
```

## Troubleshooting

### KeyError: 'action'

Dataset uses `action.action` instead of `action`. Re-convert the dataset.

### Language modality must have exactly one key

Add language config to modality config:
```python
"language": ModalityConfig(delta_indices=[0], modality_keys=["task"])
```

### Out of Memory

- Reduce `--global-batch-size`
- Use `--tune-llm False` and `--tune-visual False`

### Training Loss Not Decreasing

- Check dataset quality (valid observations and actions)
- Verify joint order matches modality config
- Try higher learning rate

## Trained Models

| Model | HuggingFace | Dataset | Steps |
|-------|-------------|---------|-------|
| G1 Inspire 9-Dataset | `datamentorshf/groot-g1-inspire-9datasets` | Combined | 5000 |
| G1 Loco-Manipulation | `datamentorshf/groot-g1-loco-manip` | Isaac Sim | 5000 |
| G1 Teleop | `datamentorshf/groot-g1-teleop` | Real robot | 4000 |

## References

- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
- [GROOT N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
