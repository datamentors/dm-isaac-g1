# DM-ISAAC-G1

G1 Robot Training Suite for the **Unitree G1 EDU 2** with **Inspire Robotics Dexterous Hands** (53 DOF total).

## Features

- **Fine-tuning**: GROOT N1.6 model training with multi-dataset support
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

## CLI Commands

```bash
# Data management
dm-g1 data download unitreerobotics/G1_Fold_Towel
dm-g1 data convert ./G1_Fold_Towel ./G1_Fold_Towel_Inspire
dm-g1 data validate ./G1_Fold_Towel_Inspire
dm-g1 data stats ./G1_Fold_Towel_Inspire

# Inference
dm-g1 infer status                    # Check GROOT server
dm-g1 infer serve --model datamentorshf/groot-g1-inspire-9datasets
dm-g1 infer benchmark --env Isaac-PickPlace-RedBlock-G129-Inspire-Joint

# Remote workstation
dm-g1 remote connect                  # SSH to workstation
dm-g1 remote gpu                      # Check GPU status
dm-g1 remote sync                     # Git pull on workstation
```

## Trained Models

| Model | HuggingFace | Datasets | Steps |
|-------|-------------|----------|-------|
| **G1 Inspire 9-Dataset** | [datamentorshf/groot-g1-inspire-9datasets](https://huggingface.co/datamentorshf/groot-g1-inspire-9datasets) | 9 datasets, 2,230 episodes | 10,000 |
| G1 Loco-Manipulation | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | LMPnPAppleToPlateDC | 5,000 |
| G1 Teleop | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | g1-pick-apple | 4,000 |

## Robot Configuration

### G1 EDU 2 + Inspire Hands (53 DOF)

| Component | DOF | Joints |
|-----------|-----|--------|
| Legs | 12 | Hip (yaw/roll/pitch), Knee, Ankle (pitch/roll) × 2 |
| Waist | 3 | Yaw, Roll, Pitch |
| Arms | 14 | Shoulder (pitch/roll/yaw), Elbow, Wrist (roll/pitch/yaw) × 2 |
| **Inspire Hands** | 24 | 5 fingers × 2-4 DOF each, per hand |
| **Total** | **53** | |

### Inspire Hand Joints (12 per hand)

```
Index:  proximal, intermediate
Middle: proximal, intermediate
Ring:   proximal, intermediate
Pinky:  proximal, intermediate
Thumb:  proximal_yaw, proximal_pitch, intermediate, distal
```

## Package Structure

```
dm-isaac-g1/
├── src/dm_isaac_g1/           # Main package
│   ├── core/                  # Config, robot definitions, remote
│   ├── data/                  # Download, convert, validate, stats
│   ├── finetuning/            # GROOT training
│   ├── inference/             # Client, server, Isaac runner
│   ├── imitation/             # Demonstration collection
│   └── rl/                    # Isaac Lab RL training
├── configs/                   # YAML configurations
├── scripts/                   # Bash scripts
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## Datasets

### Training Data (9 Datasets, 2,230 Episodes)

**Hospitality Tasks** (Gripper → Inspire conversion):
- G1_Fold_Towel (714 episodes)
- G1_Clean_Table (775 episodes)
- G1_Wipe_Table (526 episodes)
- G1_Prepare_Fruit (427 episodes)
- G1_Pour_Medicine (596 episodes)
- G1_Organize_Tools (407 episodes)
- G1_Pack_PingPong (506 episodes)

**Dex3 Tasks** (Dex3 → Inspire conversion):
- G1_Dex3_ToastedBread (418 episodes)
- G1_Dex3_BlockStacking (301 episodes)

## Infrastructure

### Blackwell Workstation (192.168.1.205)
- GPU: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM)
- Container: isaac-sim with grootenv

### Spark Inference Server (192.168.1.237)
- GROOT inference server
- Port: 5555

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

- [RESTRUCTURING_PLAN.md](docs/RESTRUCTURING_PLAN.md) - Package architecture plan
- [FINETUNING_LOG.md](docs/FINETUNING_LOG.md) - Training session logs
- [G1_INSPIRE_TRAINING_PLAN.md](docs/G1_INSPIRE_TRAINING_PLAN.md) - Dataset preparation
- [agent.md](agent.md) - AI agent workflow rules

## References

- [NVIDIA GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- [Inspire Robotics](https://www.inspire-robots.com/)

## License

MIT License
