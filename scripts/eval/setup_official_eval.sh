#!/usr/bin/env bash
# =============================================================================
# Setup NVIDIA Official GR00T Evaluation Pipeline
# =============================================================================
#
# This script sets up the official NVIDIA GR00T evaluation pipeline using:
#   - GR00T-WholeBodyControl (gr00t_wbc) for WBC integration
#   - RoboCasa + Robosuite (modified fork) for MuJoCo scenes
#   - Official rollout_policy.py for evaluation
#
# This replaces our custom MuJoCo XML injection approach with NVIDIA's
# battle-tested pipeline that properly handles Dex1 grippers, WBC balance,
# camera setup, and scene physics.
#
# Prerequisites:
#   - NVIDIA GPU with EGL support
#   - Git LFS installed
#   - Running inside dm-workstation container on 192.168.1.205
#
# Usage:
#   bash scripts/eval/setup_official_eval.sh
#
# After setup, run evaluation with:
#   bash scripts/eval/run_official_eval.sh
# =============================================================================

set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="/workspace/Isaac-GR00T"
WBC_SETUP="$GROOT_DIR/gr00t/eval/sim/GR00T-WholeBodyControl"
WBC_VENV="$WBC_SETUP/GR00T-WholeBodyControl_uv/.venv/bin/python"

echo "================================================================="
echo "  Setting up NVIDIA Official GR00T Evaluation Pipeline"
echo "================================================================="

# Step 1: Ensure Isaac-GR00T repo is available
if [ ! -d "$GROOT_DIR" ]; then
    echo "ERROR: Isaac-GR00T not found at $GROOT_DIR"
    echo "Clone it first: git clone https://github.com/NVIDIA/Isaac-GR00T.git $GROOT_DIR"
    exit 1
fi

cd "$GROOT_DIR"

# Step 2: Install system dependencies (EGL for headless rendering)
echo ""
echo "[Step 2] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq libegl1-mesa-dev libglu1-mesa 2>/dev/null || true

# Step 3: Initialize GR00T-WholeBodyControl submodule
echo ""
echo "[Step 3] Initializing GR00T-WholeBodyControl submodule..."
git submodule update --init external_dependencies/GR00T-WholeBodyControl
git -C external_dependencies/GR00T-WholeBodyControl lfs pull

# Step 4: Run NVIDIA's setup script
echo ""
echo "[Step 4] Running NVIDIA WBC setup script..."
echo "  This creates a separate Python 3.10 venv with robosuite + robocasa + gr00t_wbc"
bash "$WBC_SETUP/setup_GR00T_WholeBodyControl.sh"

# Step 5: Verify the environment works
echo ""
echo "[Step 5] Verifying evaluation environment..."
$WBC_VENV - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym
import robocasa, robosuite
from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv
print(f"robosuite version: {robosuite.__version__}")
env = gym.make(
    "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
    onscreen=False,
    offscreen=True,
    enable_waist=True,
)
print(f"Environment created: {type(env)}")
obs, info = env.reset()
print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
env.close()
print("Environment verification PASSED")
PY

echo ""
echo "================================================================="
echo "  Setup Complete!"
echo ""
echo "  To run evaluation:"
echo "    bash $SCRIPT_DIR/run_official_eval.sh"
echo "================================================================="
