#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
#
# PnP Apple Fine-tuning v3 — no gradient accumulation
#
# Key differences from v2 (and NVIDIA's 8-GPU setup):
#   v2: global_batch=128, grad_accum=8 → 1024 effective, sequential batches
#   NVIDIA: global_batch=1024, 8 GPUs, grad_accum=1 → 128/GPU, parallel diverse batches
#   v3: global_batch=128, grad_accum=1, grad_checkpointing=True → 128/step, no accumulation
#
# v3 trades total sample volume for gradient diversity:
#   - Each optimizer step sees 128 independently sampled + augmented examples
#   - No sequential accumulation artifacts
#   - gradient_checkpointing saves memory (trades compute for VRAM)
#   - Early stopping at smoothed loss < 0.01 via EarlyStopOnLossCallback
#   - wandb logging enabled
#
# Run on workstation (inside dm-workstation container):
#   export WANDB_API_KEY=<your-key>
#   bash /workspace/scripts/finetune_pnp_v3.sh
#
# Or from Mac via SSH:
#   source .env
#   sshpass -p "$WORKSTATION_PASSWORD" ssh ... \
#     "docker exec -e WANDB_API_KEY=$WANDB_API_KEY dm-workstation \
#       bash /workspace/scripts/finetune_pnp_v3.sh"

set -x -e

cd /workspace/Isaac-GR00T

python -u /workspace/scripts/launch_finetune_v3.py 2>&1 | tee /workspace/logs/finetune_pnp_v3.log
