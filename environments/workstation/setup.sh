#!/bin/bash
# Workstation Environment Setup Script
# Run this after cloning the repo on the workstation
# This sets up UV, installs dependencies, and configures git

set -e

echo "=========================================="
echo "DM-ISAAC-G1 Workstation Environment Setup"
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
    # Inside container, use system Python (Isaac Sim's Python)
    export UV_SYSTEM_PYTHON=1
    echo "Using system Python (Isaac Sim embedded)"
else
    # On host, create venv
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv --python 3.10
    fi
    source .venv/bin/activate
fi

# 3. Install dependencies
echo ""
echo "3. Installing dependencies with UV..."
uv sync

# 4. Install dm-isaac-g1 package in editable mode
echo ""
echo "4. Installing dm-isaac-g1 package..."
uv pip install -e .

# 5. Set up git credentials (if GITHUB_TOKEN is set)
echo ""
echo "5. Configuring git..."
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
    echo "To enable git push, add GITHUB_TOKEN to .env"
fi

# 6. Set up environment variables
echo ""
echo "6. Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example - please edit with your values"
    fi
fi

# 7. CycloneDDS setup (for unitree_sdk2py)
echo ""
echo "7. Checking CycloneDDS..."
if [ -d "/opt/cyclonedds/install" ]; then
    export CYCLONEDDS_HOME=/opt/cyclonedds/install
    export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}"
    echo "CycloneDDS configured: $CYCLONEDDS_HOME"
elif [ -d "/workspace/cyclonedds/install" ]; then
    export CYCLONEDDS_HOME=/workspace/cyclonedds/install
    export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}"
    echo "CycloneDDS configured: $CYCLONEDDS_HOME"
else
    echo "CycloneDDS not found - unitree_sdk2py may not work"
fi

# 8. PYTHONPATH setup
echo ""
echo "8. Setting PYTHONPATH..."
export PYTHONPATH="${DM_REPO_PATH}/src:${PYTHONPATH}"
if [ -d "/workspace/IsaacLab" ]; then
    export PYTHONPATH="/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:${PYTHONPATH}"
fi
echo "PYTHONPATH set"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Run 'source .env' to load environment"
echo "  3. Run 'uv run dm-g1 --help' to test CLI"
echo ""
