#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
#
# 8-GPU PnP Apple Fine-tuning — EXACT replica of NVIDIA's finetune_g1.sh
#
# This is a 1:1 copy of NVIDIA's official training command from:
# https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/GR00T-WholeBodyControl/finetune_g1.sh
#
# The ONLY additions are:
#   - WANDB_API_KEY for experiment tracking
#   - Output dir changed to our naming convention
#   - Dataset path adjusted for our directory structure
#
# Expected results:
#   - ~58% success rate (NVIDIA's benchmark, ±15% variance)
#   - Training time: ~35 min on 8x H100, ~55 min on 8x A100
#   - DeepSpeed ZeRO-2 auto-enables (num_gpus > 1)
#
# Usage:
#   export WANDB_API_KEY=<your-key>
#   bash train_8gpu.sh

set -x -e

export NUM_GPUS=8

cd /workspace/Isaac-GR00T

# NVIDIA's exact command — DO NOT modify parameters
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC \
    --embodiment_tag UNITREE_G1 \
    --num_gpus $NUM_GPUS \
    --output_dir /workspace/checkpoints/groot-g1-pnp-apple-8gpu \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 1024 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /workspace/logs/finetune_pnp_8gpu.log
