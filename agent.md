# Agent Workflow Rules

This document defines the standard workflows and best practices for working with the GROOT fine-tuning and G1 robot development environment.

---

## PyTorch + CUDA Compatibility Reference

**Source of truth**: https://pytorch.org/get-started/previous-versions/

| PyTorch | Correct CUDA index | pip flag |
|---------|-------------------|----------|
| 2.7.x   | CUDA 12.8         | `--index-url https://download.pytorch.org/whl/cu128` |
| 2.6.x   | CUDA 12.4         | `--index-url https://download.pytorch.org/whl/cu124` |

**Workstation (Blackwell RTX PRO 6000)**:
- Driver: 590.48.01 → supports up to CUDA 13.1
- Docker base image: `nvidia/cuda:12.8.0-runtime-ubuntu22.04`
- PyTorch install: `torch==2.7.0 ... --index-url https://download.pytorch.org/whl/cu128`

**Rule**: Always check the official previous-versions page above before changing the PyTorch CUDA index. Never use cu126 for PyTorch 2.7.x — cu128 is the correct matching index.

---

## Single Repository Workflow

**CRITICAL**: Both local (Mac) and workstation use the **same** `dm-isaac-g1` repository.

### Design Principles

1. **Single Source of Truth**: One repo cloned on all machines
2. **UV for Dependencies**: Fast, reproducible Python environment management
3. **Docker for Isolation**: Containers mount the repo as a volume
4. **Git for Sync**: GitHub token enables workstation to pull/push

### Environment Overview

| Machine | IP | Purpose | Environment |
|---------|-----|---------|-------------|
| **Local Mac** | - | Development, editing | Local venv with UV |
| **Blackwell** | 192.168.1.205 | Training + Simulation | `environments/workstation/` |
| **Spark** | 192.168.1.237 | GROOT Inference | `environments/spark/` |

See [environments/README.md](environments/README.md) for detailed setup instructions.

---

## Git Workflow

### Workflow Diagram

```mermaid
sequenceDiagram
    participant Local as Local Mac
    participant GitHub as GitHub
    participant WS as Workstation
    participant Docker as Docker Container

    Note over Local: Make code changes
    Local->>Local: git add && git commit
    Local->>GitHub: git push origin main

    Note over Local: Sync to workstation
    Local->>WS: SSH: cd dm-isaac-g1 && git pull
    WS->>GitHub: git pull origin main
    GitHub-->>WS: Updated code

    Note over Docker: Code auto-available via mount
    Docker-->>WS: Volume mount sees changes
```

### Workstation Git Setup (One-time)

The workstation needs a GitHub token to pull/push:

```bash
# 1. Create GitHub Personal Access Token (PAT) with repo scope
# 2. Add to .env on workstation
echo "GITHUB_TOKEN=ghp_xxxxx" >> /home/datamentors/dm-isaac-g1/.env

# 3. Configure git credentials
cd /home/datamentors/dm-isaac-g1
source .env
git config --global credential.helper store
echo "https://oauth2:${GITHUB_TOKEN}@github.com" > ~/.git-credentials

# 4. Test
git pull origin main
```

### Daily Workflow

**Local → Workstation:**
```bash
# On local Mac
git add -A && git commit -m "Description" && git push origin main

# Sync to workstation
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 \
    "cd /home/datamentors/dm-isaac-g1 && git pull origin main"
```

**Workstation → Local:**
```bash
# On workstation (SSH)
cd /home/datamentors/dm-isaac-g1
git add -A && git commit -m "Description" && git push origin main

# On local Mac
git pull origin main
```

### Quick Sync Commands

```bash
# Push local and pull on workstation
git push origin main && source .env && sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "cd /home/datamentors/dm-isaac-g1 && git pull"

# Pull from workstation changes
source .env && sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205 "cd /home/datamentors/dm-isaac-g1 && git push" && git pull
```

---

## Environment Management with UV

### Local Development (Mac)

```bash
cd dm-isaac-g1

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv sync
uv pip install -e .

# Run commands
uv run dm-g1 --help
```

### Workstation Container

The container mounts the repo, so code changes are immediately available:

```bash
# Start container (from workstation)
cd /home/datamentors/dm-isaac-g1/environments/workstation
docker compose up -d

# Enter container
docker exec -it dm-workstation bash

# Inside container - repo is at /workspace/dm-isaac-g1
cd /workspace/dm-isaac-g1
uv sync  # If needed
```

### SSH Connection to Workstation

```bash
source /path/to/dm-isaac-g1/.env
sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no datamentors@192.168.1.205 "command"
```

### Docker Container Access

The main work environment is inside the Docker container:

```bash
docker exec -it dm-workstation bash -c 'cd /workspace/dm-isaac-g1 && your_command'
```

### Isaac Sim with VNC Display (NOT Headless)

**IMPORTANT**: When running Isaac Sim inference, ALWAYS use VNC display mode unless specifically asked for headless. This allows viewing the simulation.

**NOTE**: Our `dm-workstation` image does NOT have `/isaac-sim/python.sh`. Isaac Sim is installed as a Python package inside the `unitree_sim_env` conda environment. Always use `conda run -n unitree_sim_env python`.

**Complete VNC Launch Command (from local Mac):**
```bash
source .env

# Start VNC if not running
sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no datamentors@192.168.1.205 \
  "docker exec dm-workstation /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 2>/dev/null || true"

# Launch inference in background
sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no datamentors@192.168.1.205 '
docker exec -d dm-workstation bash -c "
  export DISPLAY=:1
  export PROJECT_ROOT=/workspace/unitree_sim_isaaclab
  export GR00T_STATS=/workspace/checkpoints/groot_g1_inspire_9datasets/processor/statistics.json
  export PYTHONPATH=/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:\$PYTHONPATH
  cd /workspace/dm-isaac-g1
  conda run --no-capture-output -n unitree_sim_env python scripts/policy_inference_groot_g1.py \
    --server 192.168.1.237:5555 \
    --scene pickplace_g1_inspire \
    --language \"pick up the cylinder\" \
    --num_action_steps 30 \
    --action_scale 0.1 \
    --enable_cameras \
    > /tmp/inference.log 2>&1
"
'

# Connect VNC: open vnc://192.168.1.205:5901 (password: datament)

# Check log:
sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no datamentors@192.168.1.205 \
  "docker exec dm-workstation tail -50 /tmp/inference.log"
```

**Key Points:**
- Do NOT use `--headless` flag if you want to see the simulation
- `--enable_cameras` is REQUIRED for `pickplace_g1_inspire` (scene has cameras defined)
- `PROJECT_ROOT` MUST be set — without it USD asset paths become `None/assets/...` and scene crashes
- VNC password: `datament` (8-char max for TurboVNC)
- Connect VNC client to `192.168.1.205:5901`

### Pink/Pinocchio IK Library Configuration

Pink and pinocchio are installed directly in the `unitree_sim_env` conda environment (baked into the `dm-workstation` image during the builder stage). No special path configuration is needed.

**Working PYTHONPATH for inference:**
```bash
export PYTHONPATH=/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:$PYTHONPATH
# The conda env already has IsaacLab and unitree_sim_isaaclab on its path
# DO NOT manually add /workspace/IsaacLab/... — that path is NOT used by the conda env
```

See [docs/WORKSTATION_ENVIRONMENT_MAP.md](docs/WORKSTATION_ENVIRONMENT_MAP.md) for full details on the environment architecture and why certain paths impact others.

### Container Environment Notes

**IMPORTANT: Avoid Container Recreation Issues**

The Docker container loses all manual installations when recreated (`docker compose down && up`). To prevent recurring library issues:

1. **Use `docker compose stop/start`** instead of `down/up` when possible - this preserves container state
2. **The working container has these libraries pre-installed in IsaacLab env:**
   - `pink` (pin-pink-3.1.0)
   - `pinocchio` (in cmeel.prefix)
   - `hpp-fcl` (in cmeel.prefix)
3. **If you must recreate the container**, the libraries are already in IsaacLab's env_isaaclab directory which is mounted, so no reinstallation is needed
4. **CUDA/Driver Compatibility**: The Blackwell GPU uses CUDA 13.1 driver, but Isaac Sim's torch is built for CUDA 12.8. This can cause symbol errors if paths are configured incorrectly.

**Quick Container Commands:**
```bash
# PREFERRED - preserves container state
docker compose stop && docker compose start

# ONLY if needed - recreates container (loses pip installs not in mounted volumes)
docker compose down && docker compose up -d
```

## Directory Structure

### Local Repository (Mac)
```
/Users/elianomarques/Documents/DataScienceProjects/Datamentors/dm-isaac-g1/
├── .env                    # Environment variables (SSH passwords, HF tokens)
├── agent.md               # This file
├── docs/                  # Documentation
│   ├── FINETUNING_LOG.md # Training session logs
│   └── G1_INSPIRE_TRAINING_PLAN.md
├── cleanup/               # Cleanup reference files
│   ├── README.md
│   └── grootenv_main.yml
└── scripts/               # Local utility scripts
```

### Workstation Container (/workspace)
```
/workspace/
├── Isaac-GR00T/           # GROOT fine-tuning framework
│   ├── datasets/          # Datasets (must be in /workspace for container access)
│   ├── scripts/           # Conversion and utility scripts
│   └── *.py configs       # Modality configs (g1_fold_towel_config.py, etc.)
├── checkpoints/           # Model checkpoints
│   ├── groot_g1_full/     # Simulated training checkpoint
│   ├── groot_g1_teleop/   # Real robot data checkpoint
│   └── groot_g1_fold_towel/  # Hospitality task checkpoint
└── IsaacLab/              # Isaac Lab for RL training
```

## Dataset Handling

### GROOT Expected Format

GROOT N1.6 expects datasets in LeRobot v2 format with specific field names:
- `observation.state` - Concatenated state vector
- `action` - Concatenated action vector (NOT `action.action`)
- `observation.images.*` - Video frames

### Converting Unitree Hospitality Datasets

Unitree hospitality datasets (G1_Fold_Towel, G1_Clean_Table, etc.) use individual fields:
- `observation.left_arm`, `observation.right_arm`, etc.
- `action.left_arm`, `action.right_arm`, etc.

Use the conversion script:
```bash
python /workspace/Isaac-GR00T/scripts/convert_g1_format.py \
  --input /path/to/original_dataset \
  --output /path/to/converted_dataset_GROOT
```

### Required Meta Files

Each dataset must have in `meta/`:
1. `info.json` - Dataset info with features
2. `modality.json` - Joint mapping for GROOT
3. `episodes.jsonl` - Episode metadata
4. `tasks.jsonl` - Task descriptions

## Fine-tuning Workflow

### 1. Validate Dataset Before Training

```bash
# Check info.json features
cat /workspace/Isaac-GR00T/datasets/DATASET_NAME/meta/info.json | python3 -c "import json,sys; d=json.load(sys.stdin); print([k for k in d['features'].keys()])"

# Check parquet columns
python3 -c "
import pyarrow.parquet as pq
import glob
files = sorted(glob.glob('/workspace/Isaac-GR00T/datasets/DATASET_NAME/data/chunk-000/*.parquet'))
if files:
    t = pq.read_table(files[0])
    print('Columns:', t.column_names)
"
```

### 2. Run Test Training First

Always run a small test (100 steps) before full training:
```bash
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /workspace/Isaac-GR00T/datasets/DATASET_GROOT \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/Isaac-GR00T/config.py \
  --output-dir /workspace/checkpoints/test_run \
  --max-steps 100 \
  --save-steps 50 \
  --global-batch-size 8
```

### 3. Clean Test Checkpoints

Delete test checkpoints before full training to save disk space:
```bash
rm -rf /workspace/checkpoints/test_run
```

### 4. Run Full Training

```bash
nohup python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path /workspace/Isaac-GR00T/datasets/DATASET_GROOT \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/Isaac-GR00T/config.py \
  --num-gpus 1 \
  --output-dir /workspace/checkpoints/groot_TASK_NAME \
  --max-steps 5000 \
  --save-steps 1000 \
  --global-batch-size 8 \
  --learning-rate 1e-4 \
  --dataloader-num-workers 4 \
  > /tmp/finetune_TASK.log 2>&1 &
```

### 5. Monitor Training

```bash
tail -f /tmp/finetune_TASK.log
```

## Disk Space Management

### Check Disk Usage
```bash
df -h /workspace
du -sh /workspace/*
```

### Safe Cleanup Targets
- `/root/.cache/uv` - Python package cache
- `/root/.cache/pip` - Pip cache
- `/opt/conda/pkgs` - Conda package cache (`conda clean --all -y`)
- Intermediate checkpoints (keep only final)
- Test training outputs
- Old HuggingFace model caches in `/root/.cache/huggingface/hub/`

### DO NOT Delete
- Base model: `models--nvidia--GR00T-N1.6-3B`
- Final checkpoints (checkpoint-5000)
- Active datasets

## HuggingFace Upload

### Model Upload
```bash
huggingface-cli upload datamentorshf/groot-TASK-NAME /workspace/checkpoints/groot_TASK/checkpoint-5000 . --repo-type model --private
```

### Dataset Download
Use git LFS for large datasets to avoid rate limiting:
```bash
git lfs install
git clone https://huggingface.co/datasets/unitreerobotics/DATASET_NAME
```

## Common Issues & Solutions

### Issue: SSH Permission Denied
**Solution**: Use `sshpass` with password from `.env`

### Issue: Dataset path not found in container
**Solution**: Copy dataset to `/workspace/` (home directory is not mounted in container)

### Issue: `KeyError: 'action'` during training
**Solution**: Dataset uses `action.action` instead of `action`. Re-convert dataset.

### Issue: `Language modality must have exactly one key`
**Solution**: Add `"language": ModalityConfig(delta_indices=[0], modality_keys=["task"])` to config

### Issue: HuggingFace rate limiting (429)
**Solution**: Use git LFS clone instead of `huggingface-cli download`

## Modality Config Template

```python
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag

config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_name"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["body", "left_arm", "right_arm", "left_gripper", "right_gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["body", "left_arm", "right_arm", "left_gripper", "right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # ... one per modality_key
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
```

## Trained Models

| Model | HuggingFace Repo | Dataset | Steps |
|-------|-----------------|---------|-------|
| G1 Loco-Manipulation | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | unitree_g1.LMPnPAppleToPlateDC | 5000 |
| G1 Teleop | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | g1-pick-apple | 4000 |
| G1 Fold Towel | (training in progress) | G1_Fold_Towel | 5000 |

---

## Workstation Scripts Reference

Scripts that exist on the workstation container but should be migrated to this repo:

### Data Preparation Scripts (`/workspace/scripts/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `fix_episodes.py` | Create missing episodes.jsonl and tasks.jsonl | To migrate |
| `split_parquet.py` | Split combined parquet into per-episode files | To migrate |
| `compute_stats.py` | Compute normalization statistics for datasets | To migrate |
| `fix_video_structure.py` | Restructure video directories for GROOT | To migrate |
| `complete_split.py` | Complete partial parquet splits | To migrate |
| `convert_to_inspire.py` | Convert datasets to Inspire format | ✅ In repo |
| `combine_inspire_datasets.py` | Combine multiple datasets | ✅ In repo |

### Isaac-GR00T Scripts (`/workspace/Isaac-GR00T/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `download_dataset.py` | HuggingFace dataset downloader | To migrate |
| `download_all_meta.py` | Download dataset metadata | To migrate |
| `download_meta.py` | Download single dataset metadata | To migrate |
| `g1_teleop_config.py` | Teleop dataset modality config | To migrate |
| `g1_fold_towel_config.py` | Hospitality task modality config | To migrate |
| `g1_inspire_simple_config.py` | Simple Inspire config | To migrate |
| `g1_inspire_unified_config.py` | Full 53 DOF Inspire config | ✅ In repo |

### Note on Script Migration

See [RESTRUCTURING_PLAN.md](./RESTRUCTURING_PLAN.md) for the full plan to migrate these scripts into a proper Python package structure.

---

## Camera Configuration

### Overview

Camera configurations for the Unitree G1 robot are centralized in `src/dm_isaac_g1/configs/camera_configs.py`. This module provides standardized camera settings based on the official Unitree Isaac Sim/Lab integration.

**Reference Repository**: https://github.com/unitreerobotics/unitree_sim_isaaclab

### Key Concepts

1. **Head Camera (Hand-Agnostic)**: The front camera mounted on the robot's head (`d435_link`) uses the same configuration regardless of which hands (DEX3, Inspire, Gripper) are used. This camera is the primary observation for manipulation tasks.

2. **Wrist Cameras (Hand-Dependent)**: Cameras mounted on each hand have configurations that depend on the hand type, as different hands have different link structures and camera mount positions.

3. **World Camera**: External camera at a fixed position for third-person observation views.

### Supported Camera Links

| Link Name | Robot Part | Available In |
|-----------|------------|--------------|
| `d435_link` | Intel RealSense D435 on head | Full Unitree USD |
| `torso_link` | Torso (fallback) | All G1 scenes |
| `left_hand_camera_base_link` | DEX3 left wrist | DEX3 hand USD |
| `right_hand_camera_base_link` | DEX3 right wrist | DEX3 hand USD |
| `left_wrist_yaw_link` | Inspire/Gripper left wrist | Inspire/Gripper USD |
| `right_wrist_yaw_link` | Inspire/Gripper right wrist | Inspire/Gripper USD |

### Usage in Inference Script

The `policy_inference_groot_g1.py` script automatically:
1. Detects available camera links in the scene
2. Uses the primary head camera config (`d435_link`) if available
3. Falls back to torso-mounted camera for simpler robot models

```python
from dm_isaac_g1.configs.camera_configs import (
    get_head_camera_config,
    get_wrist_camera_configs,
    HandType,
    RobotType,
)

# Get head camera (hand-agnostic, same for all hand types)
head_cam = get_head_camera_config(RobotType.G1)

# Get wrist cameras (hand-dependent)
left_wrist, right_wrist = get_wrist_camera_configs(HandType.INSPIRE)
```

### Adding Cameras to Scenes

When a scene doesn't have the required camera links, the inference script dynamically adds a camera attached to an available link (e.g., `torso_link`). The camera configuration is based on the standard Unitree settings.

### Scene Camera Availability

| Scene | d435_link | Wrist Cameras | Fallback |
|-------|-----------|---------------|----------|
| `locomanipulation_g1` | ❌ | ❌ | torso_link |
| `pickplace_g1_inspire` | ✅ | ✅ (Inspire) | - |
| `fixed_base_ik_g1` | ❌ | ❌ | torso_link |

### Camera Position Notes

The head camera faces forward with a slight downward inclination to capture the hands and workspace. This matches the camera view used during training data collection.

- **Primary config (d435_link)**: Position (0, 0, 0), Rotation looking forward-down
- **Fallback config (torso_link)**: Position (15cm forward, 35cm up), ~30° pitch down

---
