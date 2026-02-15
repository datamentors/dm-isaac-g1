#!/usr/bin/env python3
"""
Launch multi-dataset finetuning for GROOT N1.6.

This script extends the standard launch_finetune.py to support
multiple datasets with equal weighting.

Usage:
    python launch_multi_finetune.py \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-paths /path/to/dataset1 /path/to/dataset2 ... \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path /path/to/config.py \
        --output-dir /path/to/output \
        --max-steps 10000
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.experiment import run


class EmbodimentTagEnum(Enum):
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    GR1 = "gr1"
    UNITREE_G1 = "unitree_g1"
    LIBERO_PANDA = "libero_panda"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"
    OXE_DROID = "oxe_droid"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    NEW_EMBODIMENT = "new_embodiment"


@dataclass
class MultiDatasetFinetuneConfig:
    """Configuration for multi-dataset fine-tuning."""

    # Required
    base_model_path: str
    """Path or HuggingFace identifier for the pre-trained base model."""

    dataset_paths: List[str]
    """Paths to dataset directories. Multiple datasets can be specified."""

    embodiment_tag: EmbodimentTagEnum
    """Identifier specifying which embodiment (robot configuration) to use."""

    output_dir: str
    """Directory to save checkpoints and training outputs."""

    # Optional with defaults
    modality_config_path: Optional[str] = None
    """Path to custom modality config Python file."""

    max_steps: int = 10000
    save_steps: int = 1000
    save_total_limit: int = 5
    global_batch_size: int = 32
    dataloader_num_workers: int = 8
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    num_gpus: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    tune_llm: bool = False
    tune_visual: bool = True
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    shard_size: int = 1024
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = 100000

    use_wandb: bool = False


def load_modality_config(modality_config_path: str):
    """Load custom modality config from Python file."""
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(MultiDatasetFinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # Load modality config if provided
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # Build datasets list - one entry per dataset path
    datasets_config = []
    for dataset_path in ft_config.dataset_paths:
        datasets_config.append({
            "dataset_paths": [dataset_path],
            "mix_ratio": 1.0,  # Equal weighting
            "embodiment_tag": embodiment_tag,
        })

    print(f"Configuring {len(datasets_config)} datasets for training:")
    for i, ds in enumerate(datasets_config):
        print(f"  [{i+1}] {ds['dataset_paths'][0]}")

    # Build config
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": datasets_config,
            }
        }
    )
    config.load_config_path = None

    # Apply model tuning parameters
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Training parameters
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6-multi"

    # Data parameters
    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
