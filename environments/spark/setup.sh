#!/bin/bash
# Spark Inference Environment Setup Script
# Run this after cloning the repo on Spark (ARM64)
# This sets up UV, installs dependencies, and configures git

set -e

echo "=========================================="
echo "DM-ISAAC-G1 Spark Inference Setup (ARM64)"
echo "=========================================="

# Check if running in container or on host
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
    IN_CONTAINER=true
else
    echo "Running on host machine"
    IN_CONTAINER=false
fi

# Determine repo path
if [ -z "$DM_REPO_PATH" ]; then
    if [ "$IN_CONTAINER" = true ]; then
        DM_REPO_PATH="/workspace/dm-isaac-g1"
    else
        DM_REPO_PATH="/home/datamentors/dm-isaac-g1"
    fi
fi

cd "$DM_REPO_PATH"

# 1. Install UV if not present
echo ""
echo "1. Checking UV installation..."
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "UV already installed: $(uv --version)"
fi

# 2. Set up Python environment
echo ""
echo "2. Setting up Python environment..."
if [ "$IN_CONTAINER" = true ]; then
    # Inside container, use system Python (JetPack Python)
    export UV_SYSTEM_PYTHON=1
    echo "Using system Python (JetPack)"
else
    # On host, create venv
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv --python 3.10
    fi
    source .venv/bin/activate
fi

# 3. Install ARM64-compatible PyTorch (if not already installed)
echo ""
echo "3. Checking PyTorch installation..."
if ! python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "PyTorch not found. On Jetson, install from JetPack wheels:"
    echo "  pip install --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision"
fi

# 4. Install inference dependencies
echo ""
echo "4. Installing inference dependencies with UV..."
cd environments/spark
uv pip install --system $(grep -E "^\s+\"" pyproject.toml | grep -v "^#" | tr -d '", ' | head -20) 2>/dev/null || \
pip install numpy pyyaml python-dotenv tqdm transformers safetensors pillow huggingface-hub pyzmq msgpack httpx einops timm diffusers

cd "$DM_REPO_PATH"

# 5. Install dm-isaac-g1 package in editable mode
echo ""
echo "5. Installing dm-isaac-g1 package..."
pip install -e .

# 6. Set up git credentials (if GITHUB_TOKEN is set)
echo ""
echo "6. Configuring git..."
if [ -f ".env" ]; then
    source .env
fi

if [ -n "$GITHUB_TOKEN" ]; then
    echo "Configuring git credentials..."
    git config --global credential.helper store
    echo "https://oauth2:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    echo "Git credentials configured"
else
    echo "GITHUB_TOKEN not set - skipping git credential setup"
fi

# 7. Set up environment variables
echo ""
echo "7. Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example - please edit with your values"
    fi
fi

# 8. PYTHONPATH setup
echo ""
echo "8. Setting PYTHONPATH..."
export PYTHONPATH="${DM_REPO_PATH}/src:${PYTHONPATH}"
echo "PYTHONPATH set"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start GROOT inference server:"
echo "  python -m dm_isaac_g1.inference.server \\"
echo "    --model-path /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \\"
echo "    --port 5555"
echo ""
