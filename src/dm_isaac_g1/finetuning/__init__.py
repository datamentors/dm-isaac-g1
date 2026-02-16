"""Fine-tuning module for GROOT N1.6 model.

This module provides utilities for fine-tuning GROOT on custom datasets.
The actual training uses NVIDIA's Isaac-GR00T framework.

Usage:
    # From workstation with Isaac-GR00T installed
    python -m dm_isaac_g1.finetuning.launcher \\
        --dataset /workspace/datasets/my_dataset \\
        --config configs/g1_inspire_53dof.py \\
        --output /workspace/checkpoints/my_model
"""

from .launcher import FinetuneArgs, launch_finetune, build_finetune_command

__all__ = ["FinetuneArgs", "launch_finetune", "build_finetune_command"]
