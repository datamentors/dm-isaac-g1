# GROOT N1.6 Fine-tuning Guide

Fine-tuning the GROOT N1.6 model for the Unitree G1 robot with UNITREE_G1 gripper embodiment.

## Overview

We use the pre-registered `UNITREE_G1` embodiment tag in Isaac-GR00T, which expects:
- **State**: 31 DOF flat vector (`observation.state`)
- **Action**: 23 DOF flat vector (`action`) — arms RELATIVE, grippers/waist/nav ABSOLUTE
- **Camera**: 1 ego-view (`observation.images.ego_view`)
- **Action horizon**: 30 steps
- **No custom modality config needed** — UNITREE_G1 is built into Isaac-GR00T

## Prerequisites

- **Workstation**: 192.168.1.205 with NVIDIA RTX PRO 6000 (98 GB VRAM)
- **Container**: `dm-workstation` with conda env `unitree_sim_env`
- **Isaac-GR00T**: Installed at `/workspace/Isaac-GR00T/`
- **Base model**: `nvidia/GR00T-N1.6-3B` (auto-downloaded from HuggingFace)

## Step 1: Download Dataset

```bash
# SSH to workstation and enter container
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205
docker exec -it dm-workstation bash

# Download from HuggingFace
huggingface-cli download unitreerobotics/G1_Fold_Towel --repo-type dataset \
    --local-dir /workspace/datasets/hospitality/G1_Fold_Towel
```

Available datasets:
- `unitreerobotics/G1_Fold_Towel` (200 episodes, 310k frames)
- `unitreerobotics/G1_Clean_Table` (200 episodes, 196k frames)
- `unitreerobotics/G1_Wipe_Table` (200 episodes, 264k frames)
- `unitreerobotics/G1_Prepare_Fruit` (200 episodes, 123k frames)
- `unitreerobotics/G1_Pour_Medicine` (200 episodes, 158k frames)
- `unitreerobotics/G1_Organize_Tools` (200 episodes, 182k frames)
- `unitreerobotics/G1_Pack_PingPong` (200 episodes, 160k frames)

## Step 2: Convert to GR00T Format

The raw datasets use per-body-part columns. The conversion flattens them into `observation.state` (31 DOF) and `action` (23 DOF) and renames the camera to `observation.images.ego_view`.

```bash
python -m dm_isaac_g1.data.convert_to_groot \
    --input /workspace/datasets/hospitality/G1_Fold_Towel \
    --output /workspace/datasets/groot/G1_Fold_Towel \
    --ego-camera cam_left_high
```

Output structure:
```
datasets/groot/G1_Fold_Towel/
├── meta/
│   ├── info.json           # Dataset metadata (31 DOF state, 23 DOF action)
│   ├── modality.json       # Joint mapping for UNITREE_G1
│   ├── episodes.jsonl      # Episode metadata
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── observation.images.ego_view/    # ← renamed from cam_left_high
            └── episode_000000.mp4
```

## Step 3: Generate Normalization Stats

```bash
conda run --no-capture-output -n unitree_sim_env python -c "
from gr00t.utils.generate_rel_stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('/workspace/datasets/groot/G1_Fold_Towel', EmbodimentTag.UNITREE_G1)
"
```

This creates `meta/stats.json` and `meta/relative_stats.json`.

## Step 4: Run Training

### Single Dataset

```bash
bash /workspace/dm-isaac-g1/scripts/training/finetune_fold_towel.sh
```

Or manually:
```bash
cd /workspace/Isaac-GR00T
conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/groot/G1_Fold_Towel \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-gripper-fold-towel \
    --save_total_limit 2 \
    --max_steps 10000 \
    --save_steps 2000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /workspace/logs/finetune_fold_towel.log
```

### Multiple Datasets (Merged)

```bash
# First merge datasets
python /workspace/dm-isaac-g1/scripts/training/merge_datasets.py \
    --input-dir /workspace/datasets/groot \
    --output /workspace/datasets/groot_merged

# Generate stats for merged dataset
conda run --no-capture-output -n unitree_sim_env python -c "
from gr00t.utils.generate_rel_stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('/workspace/datasets/groot_merged', EmbodimentTag.UNITREE_G1)
"

# Run training
bash /workspace/dm-isaac-g1/scripts/training/finetune_hospitality_7ds.sh
```

### Resume from Checkpoint

```bash
# Download checkpoint from HuggingFace
huggingface-cli download datamentorshf/groot-g1-gripper-fold-towel \
    --local-dir /workspace/checkpoints/groot-g1-gripper-fold-towel

# Resume training for 4000 more steps
bash /workspace/dm-isaac-g1/scripts/training/finetune_resume.sh \
    /workspace/checkpoints/groot-g1-gripper-fold-towel \
    /workspace/datasets/groot/G1_Fold_Towel \
    /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    4000
```

## Step 5: Upload Model

```bash
# Create inference-only copy (removes optimizer state: ~9 GB vs ~22 GB)
mkdir -p /workspace/checkpoints/model-upload
cp /workspace/checkpoints/groot-g1-gripper-fold-towel/checkpoint-10000/model-*.safetensors \
   /workspace/checkpoints/groot-g1-gripper-fold-towel/checkpoint-10000/config.json \
   /workspace/checkpoints/groot-g1-gripper-fold-towel/checkpoint-10000/generation_config.json \
   /workspace/checkpoints/model-upload/

# Upload to HuggingFace
huggingface-cli upload datamentorshf/groot-g1-gripper-fold-towel \
    /workspace/checkpoints/model-upload . \
    --repo-type model --private
```

## Step 6: Deploy to Inference Server

```bash
# On Spark (192.168.1.237)
docker exec groot-server bash -c "
    huggingface-cli download datamentorshf/groot-g1-gripper-fold-towel \
        --local-dir /workspace/checkpoints/groot-g1-gripper-fold-towel
"

# Update docker-compose.yml or .env with new model path
# Restart server
docker restart groot-server
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_steps` | 10000 | Total training steps |
| `--save_steps` | 2000 | Checkpoint frequency |
| `--save_total_limit` | 2 | Keep only last N checkpoints |
| `--global_batch_size` | 64 | Batch size across all GPUs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-5 | Weight decay |
| `--warmup_ratio` | 0.05 | Warmup ratio (5% of total steps) |
| `--dataloader_num_workers` | 4 | Data loading workers |
| `--embodiment_tag` | UNITREE_G1 | Pre-registered embodiment |

### Tuning Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tune_llm` | False | Fine-tune language model backbone |
| `--tune_visual` | False | Fine-tune visual encoder |
| Projector | True | Fine-tune multimodal projector |
| Diffusion head | True | Fine-tune diffusion action head |

### Expected Training Time (Blackwell RTX PRO 6000)

| Dataset | Episodes | Speed | Duration |
|---------|----------|-------|----------|
| Single dataset (200 eps) | 200 | ~1.2 it/s | ~2.3 hours |
| All 7 datasets (1400 eps) | 1400 | ~1.2 it/s | ~2.3 hours |

## All Trained Models

Every model trained in this project, with the exact datasets used and how to reproduce.

### Model 1: G1 Gripper Hospitality 7-Dataset (RECOMMENDED)

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-gripper-hospitality-7ds](https://huggingface.co/datamentorshf/groot-g1-gripper-hospitality-7ds) |
| **Embodiment** | `UNITREE_G1` (pre-registered, 31 DOF state / 23 DOF action) |
| **Steps** | 10,000 |
| **Final Loss** | 0.055 |
| **Status** | Deployed to Spark inference server |

**Datasets used (7 datasets, 1400 episodes merged):**

| Dataset | HF Repo | Episodes | Frames |
|---------|---------|----------|--------|
| G1_Fold_Towel | `unitreerobotics/G1_Fold_Towel` | 200 | 310,000 |
| G1_Clean_Table | `unitreerobotics/G1_Clean_Table` | 200 | 196,000 |
| G1_Wipe_Table | `unitreerobotics/G1_Wipe_Table` | 200 | 264,000 |
| G1_Prepare_Fruit | `unitreerobotics/G1_Prepare_Fruit` | 200 | 123,000 |
| G1_Pour_Medicine | `unitreerobotics/G1_Pour_Medicine` | 200 | 158,000 |
| G1_Organize_Tools | `unitreerobotics/G1_Organize_Tools` | 200 | 182,000 |
| G1_Pack_PingPong | `unitreerobotics/G1_Pack_PingPong` | 200 | 160,000 |
| **Total** | | **1,400** | **1,393,000** |

**Reproduce:**
```bash
# SSH to workstation and enter container
source .env
sshpass -p "$WORKSTATION_PASSWORD" ssh datamentors@192.168.1.205
docker exec -it dm-workstation bash

# 1. Download all 7 datasets
for ds in G1_Fold_Towel G1_Clean_Table G1_Wipe_Table G1_Prepare_Fruit \
          G1_Pour_Medicine G1_Organize_Tools G1_Pack_PingPong; do
    huggingface-cli download "unitreerobotics/$ds" --repo-type dataset \
        --local-dir "/workspace/datasets/hospitality/$ds"
done

# 2. Convert all to GR00T UNITREE_G1 format
for ds in G1_Fold_Towel G1_Clean_Table G1_Wipe_Table G1_Prepare_Fruit \
          G1_Pour_Medicine G1_Organize_Tools G1_Pack_PingPong; do
    python -m dm_isaac_g1.data.convert_to_groot \
        --input "/workspace/datasets/hospitality/$ds" \
        --output "/workspace/datasets/groot/$ds"
done

# 3. Merge into single dataset
python /workspace/dm-isaac-g1/scripts/training/merge_datasets.py

# 4. Generate normalization stats
conda run -n unitree_sim_env python -c "
from gr00t.utils.generate_rel_stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('/workspace/datasets/groot_merged', EmbodimentTag.UNITREE_G1)
"

# 5. Train (10,000 steps, ~2.3 hours on RTX PRO 6000)
bash /workspace/dm-isaac-g1/scripts/training/finetune_hospitality_7ds.sh
```

---

### Model 2: G1 Gripper Fold Towel (Full)

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-gripper-fold-towel-full](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel-full) |
| **Embodiment** | `UNITREE_G1` (pre-registered) |
| **Steps** | 10,000 (6000 initial + 4000 resumed) |
| **Status** | Uploaded |

**Dataset used (1 dataset, 200 episodes):**

| Dataset | HF Repo | Episodes | Frames |
|---------|---------|----------|--------|
| G1_Fold_Towel | `unitreerobotics/G1_Fold_Towel` | 200 | 310,000 |

**Reproduce:**
```bash
# 1. Download dataset
huggingface-cli download unitreerobotics/G1_Fold_Towel --repo-type dataset \
    --local-dir /workspace/datasets/hospitality/G1_Fold_Towel

# 2. Convert
python -m dm_isaac_g1.data.convert_to_groot \
    --input /workspace/datasets/hospitality/G1_Fold_Towel \
    --output /workspace/datasets/groot/G1_Fold_Towel

# 3. Stats
conda run -n unitree_sim_env python -c "
from gr00t.utils.generate_rel_stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('/workspace/datasets/groot/G1_Fold_Towel', EmbodimentTag.UNITREE_G1)
"

# 4. Train (10,000 steps, ~2.3 hours on RTX PRO 6000)
bash /workspace/dm-isaac-g1/scripts/training/finetune_fold_towel.sh
```

Alternatively, to reproduce the exact two-phase approach (6000 + 4000):
```bash
# Phase 1: Train 6000 steps (or full 10000 — original crashed at 6123)
# Phase 2: Resume from checkpoint-6000
bash /workspace/dm-isaac-g1/scripts/training/finetune_resume.sh \
    /workspace/checkpoints/groot-g1-gripper-fold-towel \
    /workspace/datasets/groot/G1_Fold_Towel \
    /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    4000
```

---

### Model 3: G1 Gripper Fold Towel (Partial — checkpoint-6000)

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-gripper-fold-towel](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel) |
| **Embodiment** | `UNITREE_G1` (pre-registered) |
| **Steps** | 6,000 (training interrupted at step 6123 by disk pressure) |
| **Final Loss** | 0.029 |
| **Status** | Uploaded (superseded by fold-towel-full above) |

Same dataset and pipeline as Model 2 above. This is the intermediate checkpoint.

---

### Legacy Models (deprecated — not recommended for production)

These models used experimental configurations with `NEW_EMBODIMENT` tag. They predate the UNITREE_G1 gripper approach.

#### Legacy Model A: G1 Inspire 9-Dataset

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-inspire-9datasets](https://huggingface.co/datamentorshf/groot-g1-inspire-9datasets) |
| **Embodiment** | Inspire 53 DOF (`NEW_EMBODIMENT` with custom modality config) |
| **Steps** | ~10,000 |
| **Status** | Deprecated — no reproducible pipeline exists |

This model was trained on 9 mixed datasets with joint remapping from various hand types (gripper 1 DOF, Dex3 7 DOF, tri-finger 7 DOF) to Inspire 12 DOF per hand. The datasets used were a combination of:
- Simulated: `unitree_g1.LMPnPAppleToPlateDC`
- Real teleop: `g1-pick-apple`
- Dex3: `G1_Dex3_BlockStacking`, `G1_Dex3_ToastedBread`
- Hospitality: `G1_Fold_Towel`, `G1_Clean_Table`, `G1_Wipe_Table`, `G1_Prepare_Fruit`, `G1_Pour_Medicine`

**Not reproducible** — the joint remapping pipeline and exact training configuration were ad-hoc and not committed to the repo. The Inspire hand approach has been superseded by UNITREE_G1 gripper. See [G1_INSPIRE_TRAINING_PLAN.md](G1_INSPIRE_TRAINING_PLAN.md) for historical context.

#### Legacy Model B: G1 Loco-Manipulation

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) |
| **Embodiment** | `UNITREE_G1` |
| **Steps** | 5,000 |

**Dataset:** `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` subset `unitree_g1.LMPnPAppleToPlateDC` (103 episodes, simulated)

**Reproduce (for reference only):**
```bash
# Download the X-Embodiment sim dataset
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --include "unitree_g1.LMPnPAppleToPlateDC/**" \
    --repo-type dataset \
    --local-dir /workspace/datasets/gr00t_x_embodiment

# Train
cd /workspace/Isaac-GR00T
conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-loco-manip \
    --max_steps 5000 \
    --save_steps 1000 \
    --global_batch_size 8 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 4
```

#### Legacy Model C: G1 Teleop

| | |
|---|---|
| **HuggingFace** | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) |
| **Embodiment** | Tri-finger (`NEW_EMBODIMENT` with custom modality config) |
| **Steps** | 4,000 |

**Dataset:** `nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1` subset `g1-pick-apple` (311 episodes, real robot teleop)

**Reproduce (for reference only):**
```bash
# Clone via git LFS
git lfs install
git clone https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
    /workspace/datasets/g1_teleop

# Train (requires custom modality config for tri-finger hands)
cd /workspace/Isaac-GR00T
conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/g1_teleop/g1-pick-apple \
    --embodiment_tag NEW_EMBODIMENT \
    --modality_config_path /workspace/Isaac-GR00T/g1_teleop_config.py \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-teleop \
    --max_steps 5000 \
    --save_steps 1000 \
    --global_batch_size 8 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 4
```

Note: The `g1_teleop_config.py` modality config was created ad-hoc and is not in the repo. See FINETUNING_LOG.md Session 2 for details.

## Troubleshooting

### AssertionError: Original key observation.images.X not found
Video directory or info.json uses wrong camera key. Ensure `convert_to_groot.py` produces `observation.images.ego_view` (not `cam_left_high`).

### KeyError: 'action'
Dataset uses per-body-part columns. Run `convert_to_groot.py` to flatten to `observation.state` + `action`.

### Out of Memory
- Reduce `--global_batch_size` (64 works on 98 GB VRAM)
- Reduce `--dataloader_num_workers` (0 if RAM is limited)
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Disk Full During Training
- Each checkpoint is ~22 GB (model + optimizer)
- Use `--save_total_limit 2` (default)
- Clean HF cache: `rm -rf /root/.cache/huggingface/hub/datasets--*`
- Clean pip/conda caches: `rm -rf /root/.cache/pip /root/.cache/uv && conda clean --all -y`

### Training Loss Not Decreasing
- Verify dataset was converted correctly (check parquet columns match expected DOF)
- Ensure stats were generated with correct EmbodimentTag
- Try reducing learning rate

## References

- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
- [GROOT N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
