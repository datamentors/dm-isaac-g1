# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""
GROOT N1.6 Fine-tuning Launcher

This module provides utilities for launching fine-tuning jobs on the GROOT model.
The actual training is done using NVIDIA's Isaac-GR00T framework.

Usage (single dataset):
    On the workstation with Isaac-GR00T installed:

    python -m dm_isaac_g1.finetuning.launcher \\
        --base-model nvidia/GR00T-N1.6-3B \\
        --datasets /workspace/datasets/my_dataset \\
        --config /workspace/Isaac-GR00T/my_config.py \\
        --output /workspace/checkpoints/my_model

Usage (multiple datasets — uses launch_multi_finetune.py):
    python -m dm_isaac_g1.finetuning.launcher \\
        --datasets /workspace/datasets/ds1 /workspace/datasets/ds2 \\
        --config /workspace/Isaac-GR00T/my_config.py \\
        --output /workspace/checkpoints/my_model

Note:
    This launcher is a thin wrapper around the Isaac-GR00T fine-tuning system.
    The Isaac-GR00T repository must be installed at /workspace/Isaac-GR00T/
    Single dataset → launch_finetune.py
    Multiple datasets → launch_multi_finetune.py (accepts --dataset-paths nargs+)
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FinetuneArgs:
    """Arguments for fine-tuning GROOT model."""

    # Required paths
    base_model: str = "nvidia/GR00T-N1.6-3B"
    """Path or HuggingFace ID of base model."""

    datasets: List[str] = field(default_factory=list)
    """One or more dataset paths in LeRobot format."""

    config: str = ""
    """Path to modality config file (copied to /workspace/Isaac-GR00T/ first)."""

    output: str = "./checkpoints/groot_finetuned"
    """Output directory for checkpoints."""

    # Training settings
    max_steps: int = 10000
    """Maximum training steps."""

    save_steps: int = 2000
    """Save checkpoint every N steps."""

    save_total_limit: int = 2
    """Keep only the last N checkpoints (auto-deletes older ones to save disk)."""

    batch_size: int = 32
    """Global batch size (across all GPUs)."""

    learning_rate: float = 1e-4
    """Learning rate."""

    num_gpus: int = 1
    """Number of GPUs to use."""

    num_workers: int = 4
    """Number of dataloader workers."""

    # Model tuning flags
    tune_llm: bool = False
    """Fine-tune the language model backbone."""

    tune_visual: bool = False
    """Fine-tune the visual encoder."""

    tune_projector: bool = True
    """Fine-tune the multimodal projector."""

    tune_diffusion: bool = True
    """Fine-tune the diffusion action head."""

    # Optional
    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag for the dataset."""

    use_wandb: bool = False
    """Log to Weights & Biases."""

    resume_from: Optional[str] = None
    """Resume from checkpoint path."""


def build_finetune_command(args: FinetuneArgs) -> list[str]:
    """Build the command to launch fine-tuning via Isaac-GR00T.

    Uses launch_multi_finetune.py when multiple datasets are given,
    launch_finetune.py for a single dataset.

    Args:
        args: Fine-tuning arguments

    Returns:
        Command as list of strings
    """
    if not args.datasets:
        raise ValueError("At least one dataset path must be provided in args.datasets")

    multi = len(args.datasets) > 1
    script = (
        "/workspace/Isaac-GR00T/gr00t/experiment/launch_multi_finetune.py"
        if multi
        else "/workspace/Isaac-GR00T/gr00t/experiment/launch_finetune.py"
    )
    dataset_flag = "--dataset-paths" if multi else "--dataset-path"

    cmd = [
        "python", script,
        "--base-model-path", args.base_model,
        dataset_flag, *args.datasets,
        "--embodiment-tag", args.embodiment_tag,
        "--output-dir", args.output,
        "--max-steps", str(args.max_steps),
        "--save-steps", str(args.save_steps),
        "--global-batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-gpus", str(args.num_gpus),
        "--dataloader-num-workers", str(args.num_workers),
        "--save-total-limit", str(args.save_total_limit),
    ]

    if args.config:
        cmd.extend(["--modality-config-path", args.config])

    if args.tune_llm:
        cmd.append("--tune-llm")
    if args.tune_visual:
        cmd.append("--tune-visual")
    if not args.tune_projector:
        cmd.append("--no-tune-projector")
    if not args.tune_diffusion:
        cmd.append("--no-tune-diffusion-model")
    if args.use_wandb:
        cmd.append("--use-wandb")

    return cmd


def launch_finetune(args: FinetuneArgs, dry_run: bool = False) -> int:
    """Launch fine-tuning job.

    Args:
        args: Fine-tuning arguments
        dry_run: If True, print command but don't execute

    Returns:
        Return code from subprocess (0 = success)
    """
    cmd = build_finetune_command(args)

    print("=" * 60)
    print("GROOT Fine-tuning Command:")
    print(" ".join(cmd))
    print("=" * 60)

    if dry_run:
        print("(dry run - not executing)")
        return 0

    # Check if Isaac-GR00T is available
    groot_path = Path("/workspace/Isaac-GR00T")
    if not groot_path.exists():
        print(f"ERROR: Isaac-GR00T not found at {groot_path}")
        print("This script must be run on the workstation with Isaac-GR00T installed.")
        return 1

    # Launch training
    result = subprocess.run(cmd, cwd=str(groot_path))
    return result.returncode


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch GROOT fine-tuning (single or multi-dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.6-3B",
                       help="Base model path or HuggingFace ID")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="One or more dataset paths (space-separated)")
    parser.add_argument("--config", required=True,
                       help="Modality config file path on the workstation")
    parser.add_argument("--output", default="./checkpoints/groot_finetuned",
                       help="Output directory")

    # Training
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2,
                       help="Keep only the last N checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model tuning
    parser.add_argument("--tune-llm", action="store_true")
    parser.add_argument("--tune-visual", action="store_true")
    parser.add_argument("--no-tune-projector", action="store_true")
    parser.add_argument("--no-tune-diffusion", action="store_true")

    # Other
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print command but don't execute")

    args = parser.parse_args()

    ft_args = FinetuneArgs(
        base_model=args.base_model,
        datasets=args.datasets,
        config=args.config,
        output=args.output,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=not args.no_tune_projector,
        tune_diffusion=not args.no_tune_diffusion,
        embodiment_tag=args.embodiment_tag,
        use_wandb=args.use_wandb,
    )

    return launch_finetune(ft_args, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
