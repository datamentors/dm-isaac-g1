# DM-ISAAC-G1

G1 Robot Training Suite for the **Unitree G1 EDU 2** with **UNITREE_G1 Gripper Hands** (31 DOF state, 23 DOF action).

## Features

- **Fine-tuning**: GROOT N1.6 model training with UNITREE_G1 embodiment (pre-registered in Isaac-GR00T)
- **Data Pipeline**: Convert Unitree hospitality datasets to GR00T format
- **Inference**: Deploy models to Spark server for real-time control
- **Imitation Learning**: Collect and train from demonstrations
- **Reinforcement Learning**: Isaac Lab integration for RL training

## Quick Start

```bash
# Install with uv
uv pip install -e .

# Or with pip
pip install -e .

# Check CLI
dm-g1 --help
```

## Trained Models

| Model | HuggingFace | Datasets | Steps | Final Loss |
|-------|-------------|----------|-------|------------|
| **G1 Gripper Hospitality 7-Dataset** | [datamentorshf/groot-g1-gripper-hospitality-7ds](https://huggingface.co/datamentorshf/groot-g1-gripper-hospitality-7ds) | All 7 hospitality (1400 eps) | 10,000 | 0.055 |
| G1 Gripper Fold Towel | [datamentorshf/groot-g1-gripper-fold-towel](https://huggingface.co/datamentorshf/groot-g1-gripper-fold-towel) | G1_Fold_Towel (200 eps) | 6,000* | 0.029 |
| G1 Gripper Fold Towel (Full) | datamentorshf/groot-g1-gripper-fold-towel-full | G1_Fold_Towel (200 eps) | 10,000** | (in progress) |

\* Training interrupted at step 6123 due to disk pressure; checkpoint-6000 saved.
\*\* Resumed from step 6000 checkpoint for 4000 additional steps.

### Legacy Models (Inspire/Dex3 — deprecated)

| Model | HuggingFace | Datasets | Steps |
|-------|-------------|----------|-------|
| G1 Loco-Manipulation | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | LMPnPAppleToPlateDC | 5,000 |
| G1 Teleop | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | g1-pick-apple | 4,000 |

## Datasets

### UNITREE_G1 Gripper Datasets (Current)

All 7 hospitality datasets from [unitreerobotics](https://huggingface.co/unitreerobotics) on HuggingFace. Each has 200 episodes with 1 DOF gripper hands.

| Dataset | HF Repo | Episodes | Frames | Task |
|---------|---------|----------|--------|------|
| G1_Fold_Towel | `unitreerobotics/G1_Fold_Towel` | 200 | 310,000 | Fold a towel |
| G1_Clean_Table | `unitreerobotics/G1_Clean_Table` | 200 | 196,000 | Clean table surface |
| G1_Wipe_Table | `unitreerobotics/G1_Wipe_Table` | 200 | 264,000 | Wipe table |
| G1_Prepare_Fruit | `unitreerobotics/G1_Prepare_Fruit` | 200 | 123,000 | Prepare fruit |
| G1_Pour_Medicine | `unitreerobotics/G1_Pour_Medicine` | 200 | 158,000 | Pour medicine |
| G1_Organize_Tools | `unitreerobotics/G1_Organize_Tools` | 200 | 182,000 | Organize tools |
| G1_Pack_PingPong | `unitreerobotics/G1_Pack_PingPong` | 200 | 160,000 | Pack ping pong balls |
| **Merged (all 7)** | — | **1,400** | **1,280,000** | All 7 tasks |

### Data Pipeline

```bash
# 1. Download from HuggingFace
huggingface-cli download unitreerobotics/G1_Fold_Towel --repo-type dataset \
    --local-dir /workspace/datasets/hospitality/G1_Fold_Towel

# 2. Convert to GR00T UNITREE_G1 format (flat state/action vectors + ego_view video)
python -m dm_isaac_g1.data.convert_to_groot \
    --input /workspace/datasets/hospitality/G1_Fold_Towel \
    --output /workspace/datasets/groot/G1_Fold_Towel

# 3. Generate normalization stats
conda run -n unitree_sim_env python -c "
from gr00t.utils.generate_rel_stats import generate_rel_stats
from gr00t.data.embodiment_tags import EmbodimentTag
generate_rel_stats('/workspace/datasets/groot/G1_Fold_Towel', EmbodimentTag.UNITREE_G1)
"

# 4. (Optional) Merge multiple datasets for multi-task training
python scripts/training/merge_datasets.py \
    --input-dir /workspace/datasets/groot \
    --output /workspace/datasets/groot_merged
```

## Robot Configuration

### G1 + Gripper (UNITREE_G1 — 31 DOF state / 23 DOF action)

| Component | State DOF | Action DOF | Representation |
|-----------|-----------|------------|----------------|
| Left Leg | 6 | — | — |
| Right Leg | 6 | — | — |
| Waist | 3 | 3 | ABSOLUTE |
| Left Arm | 7 | 7 | RELATIVE |
| Right Arm | 7 | 7 | RELATIVE |
| Left Hand (Gripper) | 1 | 1 | ABSOLUTE |
| Right Hand (Gripper) | 1 | 1 | ABSOLUTE |
| Base Height | — | 1 | ABSOLUTE |
| Navigate (VX, VY, AngZ) | — | 3 | ABSOLUTE |
| **Total** | **31** | **23** | |

Camera: 1 ego-view (`observation.images.ego_view`, mapped from `cam_left_high`)
Action horizon: 30 steps (official UNITREE_G1 default)

## Fine-tuning

```bash
# Single dataset training (inside dm-workstation container)
bash /workspace/dm-isaac-g1/scripts/training/finetune_fold_towel.sh

# All 7 datasets training
bash /workspace/dm-isaac-g1/scripts/training/finetune_hospitality_7ds.sh

# Resume from checkpoint
bash /workspace/dm-isaac-g1/scripts/training/finetune_resume.sh \
    /workspace/checkpoints/groot-g1-gripper-fold-towel \
    /workspace/datasets/groot/G1_Fold_Towel \
    /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
    4000

# Or use the Python launcher
python -m dm_isaac_g1.finetuning.launcher \
    --datasets /workspace/datasets/groot/G1_Fold_Towel \
    --output /workspace/checkpoints/groot-g1-gripper-fold-towel \
    --embodiment-tag UNITREE_G1
```

See [FINETUNING_GUIDE.md](docs/FINETUNING_GUIDE.md) for full details.

## Package Structure

```
dm-isaac-g1/
├── src/dm_isaac_g1/           # Main package
│   ├── core/                  # Config, robot definitions, remote
│   ├── data/                  # Download, convert_to_groot, validate, stats
│   ├── finetuning/            # GROOT training launcher + configs
│   │   └── configs/           # g1_gripper_unitree.py, g1_dex3_28dof.py, ...
│   ├── inference/             # Client, server, Isaac runner
│   ├── imitation/             # Demonstration collection
│   └── rl/                    # Isaac Lab RL training
├── scripts/
│   ├── training/              # Training shell scripts + merge tool
│   └── *.py                   # Inference and utility scripts
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## Infrastructure

### Blackwell Workstation (192.168.1.205)
- GPU: NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM)
- Container: `dm-workstation` with Isaac-GR00T, conda env `unitree_sim_env`
- Disk: 1.4 TB (LVM)

### Spark Inference Server (192.168.1.237)
- GROOT inference server (`groot-server` container)
- Port: 5555 (ZMQ)
- Current model: `groot-g1-gripper-hospitality-7ds`

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format src/
ruff check src/ --fix
```

## Documentation

- [FINETUNING_GUIDE.md](docs/FINETUNING_GUIDE.md) — Step-by-step training guide
- [FINETUNING_LOG.md](docs/FINETUNING_LOG.md) — Training session logs
- [INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) — Model deployment
- [agent.md](agent.md) — AI agent workflow rules

## References

- [NVIDIA GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)

## License

MIT License
