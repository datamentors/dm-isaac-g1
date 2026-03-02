# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
"""
Continue PnP Apple fine-tuning v2 for 30K more steps (total 40K).

v2 reached loss=0.006 at 10K steps with grad_accum=8 (effective batch 1024).
This continues from checkpoint-10000 to push training further.

Key settings (identical to v2):
- gradient_accumulation_steps=8 (effective batch 1024, same as NVIDIA's 8-GPU DDP)
- max_steps=40000 (30K additional from checkpoint-10000)
- save_steps=5000 (checkpoints at 15K, 20K, 25K, 30K, 35K, 40K)
- wandb enabled, early stopping at loss < 0.001

Run on workstation:
    cd /workspace/Isaac-GR00T
    WANDB_API_KEY=<key> python -u /workspace/scripts/launch_finetune_v2_continue.py
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

    # Training config — RESUME from v2 checkpoint-10000
    config.training.start_from_checkpoint = (
        "/workspace/checkpoints/groot-g1-pnp-apple-dex3-v2/checkpoint-10000"
    )
    config.training.optim = "adamw_torch"
    config.training.output_dir = "/workspace/checkpoints/groot-g1-pnp-apple-dex3-v2-40k"
    config.training.num_gpus = 1
    config.training.global_batch_size = 128  # Same as v2
    config.training.gradient_accumulation_steps = 8  # Same as v2 (eff. batch 1024)
    config.training.gradient_checkpointing = False  # v2 didn't use it
    config.training.max_steps = 40000  # 30K more from checkpoint-10000
    config.training.learning_rate = 1e-4
    config.training.lr_scheduler_type = "cosine"
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.save_steps = 5000
    config.training.save_total_limit = 5
    config.training.logging_steps = 10
    config.training.use_wandb = True
    config.training.wandb_project = "dm-isaac-g1"
    config.training.dataloader_num_workers = 6

    # Data config
    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 100000
    config.data.video_backend = "torchcodec"

    # Inject early stopping callback via monkey-patch
    OrigTrainer = exp_module.Gr00tTrainer
    _orig_init = OrigTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        sys.path.insert(0, "/workspace/scripts")
        try:
            from callbacks import EarlyStopOnLossCallback

            self.add_callback(EarlyStopOnLossCallback(loss_threshold=0.001, window_size=50))
            logging.info("[v2-40k] Added EarlyStopOnLossCallback(threshold=0.001, window=50)")
        except ImportError:
            logging.warning("[v2-40k] callbacks.py not found — running without early stopping")

    OrigTrainer.__init__ = _patched_init

    print("=" * 60)
    print("[v2-40k] Continuing v2 training for 30K more steps")
    print(f"  Resume from: checkpoint-10000 (loss=0.006)")
    print(f"  global_batch_size=128, grad_accum=8 (eff. 1024)")
    print(f"  max_steps=40000, lr=1e-4, cosine schedule")
    print(f"  Early stopping at smoothed loss < 0.001")
    print(f"  wandb project: dm-isaac-g1")
    print("=" * 60)

    exp_module.run(config)


if __name__ == "__main__":
    main()
