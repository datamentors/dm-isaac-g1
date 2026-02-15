#!/bin/bash
# ============================================
# Setup Blackwell Workstation for Isaac G1
# ============================================
# Run this script to set up the workstation environment
set -e

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

HOST="${WORKSTATION_HOST:-$WORKSTATION_IP}"
PORT="${WORKSTATION_SSH_PORT:-22}"
USER="$WORKSTATION_USER"

# Function to run remote command
run_remote() {
    local cmd="$1"
    if [[ -n "$WORKSTATION_PASSWORD" ]] && command -v sshpass &> /dev/null; then
        sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST" "$cmd"
    else
        ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST" "$cmd"
    fi
}

echo "============================================"
echo "Setting up Blackwell Workstation"
echo "============================================"

# Create workspace directory
echo "Creating workspace directories..."
run_remote "mkdir -p ~/dm-isaac-g1/{checkpoints,logs,data,configs}"

# Clone isaac-g1-ulc-vlm repository if not present
echo "Checking for isaac-g1-ulc-vlm repository..."
run_remote "
if [[ ! -d ~/dm-isaac-g1/isaac-g1-ulc-vlm ]]; then
    echo 'Cloning isaac-g1-ulc-vlm repository...'
    cd ~/dm-isaac-g1
    git clone https://github.com/mturan33/isaac-g1-ulc-vlm.git
else
    echo 'Repository already exists, pulling latest...'
    cd ~/dm-isaac-g1/isaac-g1-ulc-vlm
    git pull
fi
"

# Check for Isaac Sim installation
echo "Checking Isaac Sim installation..."
run_remote "
ISAAC_SIM_PATH=\$(find ~/.local/share/ov/pkg -maxdepth 1 -name 'isaac-sim*' -type d 2>/dev/null | head -1)
if [[ -n \"\$ISAAC_SIM_PATH\" ]]; then
    echo \"Found Isaac Sim at: \$ISAAC_SIM_PATH\"
    echo \"export ISAAC_SIM_PATH=\$ISAAC_SIM_PATH\" >> ~/.bashrc
else
    echo 'Isaac Sim not found. You may need to install it via Omniverse Launcher.'
fi
"

# Check for Isaac Lab
echo "Checking Isaac Lab installation..."
run_remote "
if [[ -d ~/IsaacLab ]]; then
    echo 'Isaac Lab found at ~/IsaacLab'
elif [[ -d /opt/isaac-lab ]]; then
    echo 'Isaac Lab found at /opt/isaac-lab'
else
    echo 'Isaac Lab not found. Consider cloning from: https://github.com/isaac-sim/IsaacLab'
    echo 'To install:'
    echo '  git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab'
    echo '  cd ~/IsaacLab && ./isaaclab.sh --install'
fi
"

# Setup Python environment
echo "Setting up Python environment..."
run_remote "
cd ~/dm-isaac-g1
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install numpy torch torchvision gymnasium tensorboard wandb httpx pyyaml
"

echo "============================================"
echo "Workstation setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Ensure Isaac Sim is installed via Omniverse Launcher"
echo "2. Install Isaac Lab: https://isaac-sim.github.io/IsaacLab/"
echo "3. Configure GROOT server connection in .env"
echo ""
