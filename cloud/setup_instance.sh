#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for AWS/cloud 8-GPU training instance.
#
# Prerequisites:
#   - Ubuntu 22.04+ with NVIDIA drivers + CUDA installed
#   - 8x A100 80GB or 8x H100 GPUs
#   - ~100GB disk space
#
# Usage:
#   ssh into instance, then:
#   bash setup_instance.sh

set -x -e

echo "=== Setting up 8-GPU training instance ==="

# 1. Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone Isaac-GR00T
if [ ! -d "/workspace/Isaac-GR00T" ]; then
    mkdir -p /workspace
    cd /workspace
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
    cd Isaac-GR00T
else
    cd /workspace/Isaac-GR00T
    git pull
fi

# 3. Install dependencies
uv pip install --system -e ".[finetune]"
uv pip install --system torchcodec wandb

# 4. Install FFmpeg (for torchcodec)
apt-get update -qq && apt-get install -y -qq ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswresample-dev

# 5. Download training dataset from HuggingFace
# The dataset is the NVIDIA-published PnP Apple sim data
mkdir -p /workspace/datasets/gr00t_x_embodiment
if [ ! -d "/workspace/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC" ]; then
    echo "Downloading PnP Apple dataset..."
    cd /workspace/Isaac-GR00T
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='PhysicalAI-Robotics/GR00T-X-Embodiment-Sim',
    repo_type='dataset',
    allow_patterns='unitree_g1.LMPnPAppleToPlateDC/**',
    local_dir='/workspace/datasets/gr00t_x_embodiment',
)
print('Dataset downloaded')
"
fi

# 6. Verify GPU setup
echo "=== GPU Status ==="
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Setup complete. Ready for training. ==="
echo "Run: bash /workspace/train_8gpu.sh"
