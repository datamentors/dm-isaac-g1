# Cleanup Reference Files

This folder contains reference files and environment specifications for the GROOT fine-tuning setup. These files document what was installed and configured on the workstation before cleanup.

## Conda Environments

### grootenv_main.yml
The active conda environment used for GROOT fine-tuning. Located at `/opt/conda/envs/grootenv` on the workstation's isaac-sim container.

Key packages:
- Python 3.10.19
- PyTorch 2.10.0+cu128
- gr00t 0.1.0
- transformers 4.51.3
- flash-attn 2.8.3
- CUDA 12.8

## Cleaned Up Items (2026-02-14)

The following items were cleaned to free disk space:

| Item | Size | Location | Reason |
|------|------|----------|--------|
| BEHAVIOR-1K | 47GB | ~/Isaac-GR00T/BEHAVIOR-1K | Not needed for G1 training |
| .uv-cache | 6.5GB | ~/Isaac-GR00T/.uv-cache | Package cache |
| Duplicate miniconda | 17GB | /workspace/Isaac-GR00T/.miniconda | Duplicate of /opt/conda |
| Duplicate IsaacLab | 902MB | ~/IsaacLab | Duplicate of /workspace/IsaacLab |
| Intermediate checkpoints | 66GB | /workspace/checkpoints/groot_g1_teleop/checkpoint-1000,2000,3000 | Only final checkpoint needed |

**Total freed: ~137GB**

## Models Uploaded to HuggingFace

Models are stored privately at `datamentorshf` organization:

| Model | HuggingFace Repo | Training Data | Steps |
|-------|-----------------|---------------|-------|
| G1 Loco-Manipulation | [datamentorshf/groot-g1-loco-manip](https://huggingface.co/datamentorshf/groot-g1-loco-manip) | unitree_g1.LMPnPAppleToPlateDC (sim) | 5000 |
| G1 Teleop | [datamentorshf/groot-g1-teleop](https://huggingface.co/datamentorshf/groot-g1-teleop) | g1-pick-apple (real) | 4000 |

## Workstation Structure

```
/workspace/                          # Docker container workspace
├── Isaac-GR00T/                     # GROOT fine-tuning framework
├── IsaacLab/                        # Isaac Lab (RL training)
├── checkpoints/                     # Model checkpoints
│   ├── groot_g1_full/              # Loco-manip model
│   │   └── checkpoint-5000/        # Final checkpoint
│   └── groot_g1_teleop/            # Teleop model
│       └── checkpoint-4000/        # Final checkpoint
├── grootenv_main.yml               # Exported conda env spec
└── grootenv_miniconda.yml          # Backup env spec (removed)

/home/datamentors/                   # User home
├── unitree_sim_isaaclab/           # Inspire hand simulation
├── datasets/                        # Downloaded datasets
│   ├── hospitality/
│   │   └── G1_Fold_Towel/          # 21GB - Complete
│   └── dex3/                       # Cleaned (partial downloads)
└── Isaac-GR00T/                    # Duplicate (can be cleaned)
```

## Recreating the Environment

To recreate the grootenv environment:

```bash
conda env create -f grootenv_main.yml
conda activate grootenv
```
