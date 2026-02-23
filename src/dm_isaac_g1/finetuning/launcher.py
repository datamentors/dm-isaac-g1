# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""
GROOT N1.6 Fine-tuning Launcher

Thin wrapper around Isaac-GR00T's launch_finetune.py using torchrun.
Follows official NVIDIA fine-tuning conventions from:
  examples/GR00T-WholeBodyControl/finetune_g1.sh

Usage (UNITREE_G1, recommended):
    python -m dm_isaac_g1.finetuning.launcher \\
        --datasets /workspace/datasets/groot/G1_Fold_Towel \\
        --output /workspace/checkpoints/groot_g1_gripper

    Uses the pre-registered UNITREE_G1 embodiment config automatically.
    No --config needed.

Usage (custom embodiment):
    python -m dm_isaac_g1.finetuning.launcher \\
        --datasets /workspace/datasets/my_dataset \\
        --config /workspace/Isaac-GR00T/my_config.py \\
        --embodiment-tag NEW_EMBODIMENT \\
        --output /workspace/checkpoints/my_model

Note:
    Isaac-GR00T must be installed at /workspace/Isaac-GR00T/
    Uses torchrun for distributed training (matching official examples).
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
    """One or more dataset paths in LeRobot v2 format."""

    config: str = ""
    """Path to modality config file. Leave empty for pre-registered embodiments."""

    output: str = "./checkpoints/groot_finetuned"
    """Output directory for checkpoints."""

    # Training settings (matching official finetune_g1.sh)
    max_steps: int = 10000
    """Maximum training steps."""

    save_steps: int = 2000
    """Save checkpoint every N steps."""

    save_total_limit: int = 2
    """Keep only the last N checkpoints (~22 GB each with optimizer state)."""

    batch_size: int = 64
    """Global batch size (across all GPUs). Official uses 1024 with 8 GPUs."""

    learning_rate: float = 1e-4
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    warmup_ratio: float = 0.05
    """Warmup ratio."""

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
    embodiment_tag: str = "UNITREE_G1"
    """Embodiment tag for the dataset."""

    use_wandb: bool = False
    """Log to Weights & Biases."""

    resume_from: Optional[str] = None
    """Resume from checkpoint path."""

    # Data augmentation (matching official finetune_g1.sh)
    color_jitter: bool = True
    """Apply color jitter augmentation."""


def build_finetune_command(args: FinetuneArgs) -> list[str]:
    """Build the torchrun command to launch fine-tuning.

    Args:
        args: Fine-tuning arguments

    Returns:
        Command as list of strings
    """
    if not args.datasets:
        raise ValueError("At least one dataset path must be provided in args.datasets")

    script = "/workspace/Isaac-GR00T/gr00t/experiment/launch_finetune.py"

    # Use torchrun for distributed training (matching official approach)
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--master_port=29500",
        script,
        "--base_model_path", args.base_model,
        "--dataset_path", args.datasets[0],
        "--embodiment_tag", args.embodiment_tag,
        "--output_dir", args.output,
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--global_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_ratio", str(args.warmup_ratio),
        "--num_gpus", str(args.num_gpus),
        "--dataloader_num_workers", str(args.num_workers),
        "--save_total_limit", str(args.save_total_limit),
    ]

    if args.config:
        cmd.extend(["--modality_config_path", args.config])

    if args.tune_llm:
        cmd.append("--tune_llm")
    if args.tune_visual:
        cmd.append("--tune_visual")
    if not args.tune_projector:
        cmd.append("--no_tune_projector")
    if not args.tune_diffusion:
        cmd.append("--no_tune_diffusion_model")
    if args.use_wandb:
        cmd.append("--use_wandb")
    if args.color_jitter:
        cmd.extend([
            "--color_jitter_params",
            "brightness", "0.3",
            "contrast", "0.4",
            "saturation", "0.5",
            "hue", "0.08",
        ])

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
        description="Launch GROOT fine-tuning via torchrun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.6-3B",
                       help="Base model path or HuggingFace ID")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="Dataset path (first dataset used)")
    parser.add_argument("--config", default="",
                       help="Modality config file (empty for pre-registered embodiments)")
    parser.add_argument("--output", default="./checkpoints/groot_finetuned",
                       help="Output directory")

    # Training
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model tuning
    parser.add_argument("--tune-llm", action="store_true")
    parser.add_argument("--tune-visual", action="store_true")
    parser.add_argument("--no-tune-projector", action="store_true")
    parser.add_argument("--no-tune-diffusion", action="store_true")

    # Other
    parser.add_argument("--embodiment-tag", default="UNITREE_G1")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--no-color-jitter", action="store_true")
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        tune_llm=args.tune_llm,
        tune_visual=args.tune_visual,
        tune_projector=not args.no_tune_projector,
        tune_diffusion=not args.no_tune_diffusion,
        embodiment_tag=args.embodiment_tag,
        use_wandb=args.use_wandb,
        color_jitter=not args.no_color_jitter,
    )

    return launch_finetune(ft_args, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
