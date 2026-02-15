# GROOT N1.6 Fine-Tuning Plan

## Overview

This document outlines practical fine-tuning strategies for GROOT N1.6 that can be trained on available hardware. The datasets can be cross-embodiment transferred - training on GR1 data and mapping to G1 via the `--embodiment-tag UNITREE_G1` flag during fine-tuning.

## Hardware Available

| Machine | GPU | VRAM | Best For |
|---------|-----|------|----------|
| Blackwell Workstation (192.168.1.205) | RTX PRO 6000 | 98GB | Large batch training |
| DGX Spark (192.168.1.237) | GB10 | 48GB | GROOT inference + small training |

---

## Available Datasets for Unitree G1

### Option 1: G1 Loco-Manipulation (Simulated) ✅ COMPLETED

**Status**: Fine-tuning completed on 2026-02-14

| Attribute | Value |
|-----------|-------|
| Dataset | [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) |
| Subset | `unitree_g1.LMPnPAppleToPlateDC` |
| Robot | **Unitree G1** (native) |
| Task | Pick and Place Apple to Plate with Locomotion |
| Trajectories | 103 episodes |
| Format | LeRobot v2 (Parquet + MP4) |
| Training Time | ~22 minutes (5000 steps) |

**Download Command**:
```bash
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --include "unitree_g1.LMPnPAppleToPlateDC/**" \
  --local-dir ~/datasets/gr00t_x_embodiment
```

**Checkpoint Location**: `/workspace/checkpoints/groot_g1_full/checkpoint-5000/`

---

### Option 2: G1 Real Robot Teleoperation ⭐ RECOMMENDED NEXT

**Status**: Ready to train

| Attribute | Value |
|-----------|-------|
| Dataset | [`nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) |
| Robot | **Unitree G1** (real robot data) |
| Task | Fruit picking and placement (Apple, Pear, Starfruit, Grape) |
| Trajectories | 1,000 episodes (~124k frames) |
| Format | Parquet (auto-converted) |
| State Dim | 43 (upper body control with tri-finger hands) |
| Training Time | ~30-45 minutes (estimated) |

**Download Command**:
```python
from datasets import load_dataset
dataset = load_dataset("nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1")
```

Or via CLI:
```bash
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 \
  --repo-type dataset \
  --local-dir ~/datasets/gr00t_teleop_g1
```

---

### Option 3: Cross-Embodiment Transfer (GR1 → G1)

**Use Case**: When native G1 datasets are insufficient, train on GR1 data and transfer to G1

| Attribute | Value |
|-----------|-------|
| Dataset | [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) |
| Source Robot | GR1 (NVIDIA humanoid) |
| Target Robot | Unitree G1 (via embodiment mapping) |

#### Available GR1 Subsets:

| Subset | Trajectories | Tasks |
|--------|--------------|-------|
| `gr1_arms_only.CanSort` | 1,000 | Can sorting |
| `gr1_full_upper_body.Coffee` | 1,000 | Coffee making |
| `gr1_full_upper_body.Pouring` | 1,000 | Liquid pouring |
| `gr1_arms_waist.*` (24 tasks) | 240,000 | Tabletop manipulation |
| `gr1_unified.*` (24 tasks) | 24,000 | Downsampled unified |

**Cross-embodiment Training Command**:
```bash
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./datasets/gr00t_x_embodiment/gr1_arms_only.CanSort \
    --embodiment-tag UNITREE_G1 \
    --output-dir /workspace/checkpoints/groot_g1_cansort \
    --max-steps 5000
```

**Note**: The `--embodiment-tag UNITREE_G1` maps GR1 actions to G1's joint configuration.

---

### Option 4: Bimanual Manipulation (Panda Arms)

| Subset | Trajectories | Tasks |
|--------|--------------|-------|
| `bimanual_panda_gripper.Threading` | 1,000 | Threading tasks |
| `bimanual_panda_hand.LiftTray` | 1,000 | Tray lifting |
| `bimanual_panda_gripper.ThreePieceAssembly` | 1,000 | Assembly |
| `bimanual_panda_gripper.Transport` | 1,000 | Object transport |
| `bimanual_panda_hand.BoxCleanup` | 1,000 | Box cleanup |
| `bimanual_panda_hand.DrawerCleanup` | 1,000 | Drawer cleanup |

---

## Training Configuration

### Fine-Tuning Command (Verified Working)

```bash
# Activate environment
source /opt/conda/etc/profile.d/conda.sh
conda activate grootenv
cd /workspace/Isaac-GR00T

# Run fine-tuning
python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path ./datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment-tag UNITREE_G1 \
    --output-dir /workspace/checkpoints/groot_g1_full \
    --max-steps 5000 \
    --save-steps 1000 \
    --global-batch-size 8 \
    --learning-rate 1e-4 \
    --dataloader-num-workers 4
```

### Key Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--max-steps` | 5000 | ~20 min on RTX PRO 6000 |
| `--global-batch-size` | 8 | Adjust based on VRAM |
| `--learning-rate` | 1e-4 | Standard for fine-tuning |
| `--save-steps` | 1000 | Checkpoint frequency |

---

## Completed Training Sessions

### Session 1: G1 Loco-Manipulation (2026-02-14)

| Metric | Value |
|--------|-------|
| Dataset | `unitree_g1.LMPnPAppleToPlateDC` |
| Steps | 5000 |
| Duration | 22 minutes |
| Final Loss | 0.04-0.06 |
| Speed | 4.3 it/s |
| GPU Memory | 42GB |
| Checkpoint | `/workspace/checkpoints/groot_g1_full/checkpoint-5000/` |

---

## Deployment Plan

### Multi-Model Setup on Spark Servers

| Model | Port | Server | Task |
|-------|------|--------|------|
| G1 Loco-Manipulation | 5555 | 192.168.1.237 | Pick & Place with locomotion |
| G1 Teleop (pending) | 5556 | 192.168.1.237 | Fruit manipulation |

### Deploy Commands

```bash
# Copy checkpoint to Spark
scp -r /workspace/checkpoints/groot_g1_full/checkpoint-5000 \
    nvidia@192.168.1.237:/workspace/checkpoints/

# Start server on Spark
ssh nvidia@192.168.1.237
cd /workspace/gr00t
python gr00t/eval/run_gr00t_server.py \
    --model-path /workspace/checkpoints/checkpoint-5000 \
    --embodiment-tag UNITREE_G1 \
    --port 5555
```

---

## Physics Engine Compatibility

| Training Source | Inference Target | Status |
|-----------------|------------------|--------|
| MuJoCo (sim data) | MuJoCo | ✅ Works |
| MuJoCo (sim data) | Isaac Sim | ⚠️ Physics mismatch |
| Isaac Gym | Isaac Sim | ✅ Works |
| Real Robot | MuJoCo/Isaac | ✅ Works (with tuning) |

**Note**: Models trained on simulated data work best in matching physics engines. See [PHYSICS_ENGINE_LEARNINGS.md](PHYSICS_ENGINE_LEARNINGS.md).

---

## Quick Reference Links

| Resource | URL |
|----------|-----|
| X-Embodiment Sim Dataset | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim |
| G1 Teleop Dataset | https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1 |
| GR00T N1.6 Model | https://huggingface.co/nvidia/GR00T-N1.6-3B |
| Isaac-GR00T Repo | https://github.com/NVIDIA/Isaac-GR00T |

---

## Success Criteria

After training, verify:

- [x] Final loss < 0.1
- [x] Checkpoints saved correctly
- [ ] Model loads on GROOT server
- [ ] Inference produces reasonable actions
- [ ] Robot performs task in simulation
