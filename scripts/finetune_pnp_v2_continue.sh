#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
#
# Continue PnP Apple v2 Fine-tuning — 30K more steps (total 40K)
#
# Resumes from v2 checkpoint-10000 (loss=0.006) with identical settings:
#   global_batch=128, grad_accum=8 → 1024 effective batch
#   max_steps=40000, save every 5K steps
#   Early stopping at smoothed loss < 0.001
#   wandb logging enabled
#
# Run on workstation (inside dm-workstation container):
#   export WANDB_API_KEY=<your-key>
#   bash /workspace/scripts/finetune_pnp_v2_continue.sh

set -x -e

cd /workspace/Isaac-GR00T

python -u /workspace/scripts/launch_finetune_v2_continue.py 2>&1 | tee /workspace/logs/finetune_pnp_v2_continue.log
