# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""
Launch PnP Apple fine-tuning v3 with early stopping and wandb.

Key changes from v2:
- gradient_accumulation_steps=1 (no accumulation — each update from 128 diverse samples)
- gradient_checkpointing=True (trades compute for VRAM to fit batch=128 without accum)
- Early stopping at smoothed loss < 0.01 to prevent overfitting
- wandb enabled for monitoring (project: dm-isaac-g1)

Run on workstation:
    cd /workspace/Isaac-GR00T
    WANDB_API_KEY=<key> python -u /workspace/scripts/launch_finetune_v3.py
"""

import logging
import os
import sys

# Ensure Isaac-GR00T is importable
sys.path.insert(0, "/workspace/Isaac-GR00T")

from gr00t.configs.base_config import get_default_config
from gr00t.experiment import experiment as exp_module


def main():
    os.environ.setdefault("LOGURU_LEVEL", "INFO")

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [
                            "/workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC"
                        ],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "unitree_g1",
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Model config (matching NVIDIA finetune_g1.sh exactly)
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.0
    config.model.color_jitter_params = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    }

    # Training config — KEY CHANGES for v3
    config.training.start_from_checkpoint = "nvidia/GR00T-N1.6-3B"
    config.training.optim = "adamw_torch"
    config.training.output_dir = "/workspace/checkpoints/groot-g1-pnp-apple-dex3-v3"
    config.training.num_gpus = 1
    config.training.global_batch_size = 128  # Same as NVIDIA per-GPU batch
    config.training.gradient_accumulation_steps = 1  # NO accumulation — key fix
    config.training.gradient_checkpointing = True  # Save memory (recompute activations)
    config.training.max_steps = 10000
    config.training.learning_rate = 1e-4
    config.training.lr_scheduler_type = "cosine"
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.save_steps = 1000
    config.training.save_total_limit = 5
    config.training.logging_steps = 10
    config.training.use_wandb = True
    config.training.wandb_project = "dm-isaac-g1"
    config.training.dataloader_num_workers = 6  # Match NVIDIA

    # Data config
    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 100000

    # Inject early stopping callback via monkey-patch
    OrigTrainer = exp_module.Gr00tTrainer
    _orig_init = OrigTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Import here to avoid import errors on machines without the callback
        sys.path.insert(0, "/workspace/scripts")
        try:
            from callbacks import EarlyStopOnLossCallback

            self.add_callback(EarlyStopOnLossCallback(loss_threshold=0.01, window_size=50))
            logging.info("[v3] Added EarlyStopOnLossCallback(threshold=0.01, window=50)")
        except ImportError:
            logging.warning("[v3] callbacks.py not found — running without early stopping")

    OrigTrainer.__init__ = _patched_init

    print("=" * 60)
    print("[v3] PnP Fine-tuning v3 — no gradient accumulation")
    print(f"  global_batch_size=128, grad_accum=1, grad_checkpointing=True")
    print(f"  max_steps=10000, lr=1e-4, cosine schedule")
    print(f"  Early stopping at smoothed loss < 0.01")
    print(f"  wandb project: dm-isaac-g1")
    print("=" * 60)

    exp_module.run(config)


if __name__ == "__main__":
    main()
