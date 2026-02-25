#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Workstation Evaluation Environment Setup
# ============================================================================
# Run INSIDE the dm-workstation container on 192.168.1.205
#
# Usage:
#   docker exec -it dm-workstation bash
#   cd /workspace/dm-isaac-g1
#   bash scripts/eval/setup_eval_workstation.sh
#
# What this does:
#   1. Sets up GR00T-WholeBodyControl (MuJoCo eval) in a separate venv
#   2. Downloads the G1_Fold_Towel dataset for open-loop eval
#   3. Downloads the hospitality-7ds checkpoint if missing
#   4. Installs mujoco 3.2.6 in the WBC venv
#   5. Runs sanity checks
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GROOT_REPO="/workspace/Isaac-GR00T"
WBC_SETUP="$GROOT_REPO/gr00t/eval/sim/GR00T-WholeBodyControl"
DATASETS_DIR="/workspace/datasets/groot"
CHECKPOINTS_DIR="/workspace/checkpoints"

echo "============================================"
echo "  DM-ISAAC-G1 Eval Setup"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Setup GR00T-WholeBodyControl (MuJoCo eval environment)
# ------------------------------------------------------------------
echo ""
echo "[1/5] Setting up GR00T-WholeBodyControl (MuJoCo) ..."

if [ ! -f "$WBC_SETUP/setup_GR00T_WholeBodyControl.sh" ]; then
    echo "ERROR: WBC setup script not found at $WBC_SETUP"
    echo "Ensure Isaac-GR00T is mounted at /workspace/Isaac-GR00T"
    exit 1
fi

WBC_VENV="$WBC_SETUP/GR00T-WholeBodyControl_uv"
if [ -d "$WBC_VENV/.venv" ]; then
    echo "  WBC venv already exists at $WBC_VENV/.venv — skipping full setup"
    echo "  (delete $WBC_VENV to force reinstall)"
else
    echo "  Running WBC setup script (this takes ~5-10 minutes) ..."
    cd "$GROOT_REPO"
    bash "$WBC_SETUP/setup_GR00T_WholeBodyControl.sh"
    echo "  WBC setup complete."
fi

# ------------------------------------------------------------------
# Step 2: Download G1_Fold_Towel dataset for open-loop eval
# ------------------------------------------------------------------
echo ""
echo "[2/5] Checking datasets ..."

mkdir -p "$DATASETS_DIR"

download_dataset() {
    local name="$1"
    local hf_repo="$2"
    local target="$DATASETS_DIR/$name"

    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  $name: already present"
    else
        echo "  $name: downloading from $hf_repo ..."
        cd "$GROOT_REPO"
        python3 download_g1_dataset.py \
            --dataset "$hf_repo" \
            --output "$target" 2>/dev/null || \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$hf_repo', local_dir='$target', repo_type='dataset')
" 2>/dev/null || \
        huggingface-cli download --repo-type dataset "$hf_repo" --local-dir "$target"
        echo "  $name: downloaded"
    fi
}

# Download key datasets for eval
download_dataset "G1_Fold_Towel" "unitreerobotics/G1_Fold_Towel"

# The X-Embodiment sim dataset should already be at:
XE_DATASET="$GROOT_REPO/datasets/gr00t_x_embodiment/unitree_g1.LMPnPAppleToPlateDC"
if [ -d "$XE_DATASET" ]; then
    echo "  X-Embodiment PnP dataset: present at $XE_DATASET"
else
    echo "  X-Embodiment PnP dataset: NOT found (run download_dataset.py from Isaac-GR00T)"
fi

# ------------------------------------------------------------------
# Step 3: Ensure checkpoint is available
# ------------------------------------------------------------------
echo ""
echo "[3/5] Checking model checkpoints ..."

HOSPITALITY_CKPT="$CHECKPOINTS_DIR/groot-g1-gripper-hospitality-7ds"
FOLD_TOWEL_CKPT="$CHECKPOINTS_DIR/groot-g1-gripper-fold-towel-full"

if [ -d "$FOLD_TOWEL_CKPT" ]; then
    echo "  groot-g1-gripper-fold-towel-full: present"
else
    echo "  groot-g1-gripper-fold-towel-full: NOT found"
    echo "  Download with: huggingface-cli download datamentors/groot-g1-gripper-fold-towel-full --local-dir $FOLD_TOWEL_CKPT"
fi

if [ -d "$HOSPITALITY_CKPT" ]; then
    echo "  groot-g1-gripper-hospitality-7ds: present"
else
    echo "  groot-g1-gripper-hospitality-7ds: NOT found"
    echo "  Download with: huggingface-cli download datamentors/groot-g1-gripper-hospitality-7ds --local-dir $HOSPITALITY_CKPT"
fi

# ------------------------------------------------------------------
# Step 4: Install MuJoCo in the main container Python (for custom scenes)
# ------------------------------------------------------------------
echo ""
echo "[4/5] Installing MuJoCo in container Python ..."

if python3 -c "import mujoco; print(f'mujoco {mujoco.__version__}')" 2>/dev/null; then
    echo "  MuJoCo already installed in system Python"
else
    echo "  Installing mujoco 3.2.6 ..."
    pip install mujoco==3.2.6
    echo "  Done."
fi

# ------------------------------------------------------------------
# Step 5: Sanity checks
# ------------------------------------------------------------------
echo ""
echo "[5/5] Running sanity checks ..."

echo ""
echo "  --- System Python imports ---"
python3 -c "
import mujoco
print(f'  mujoco:       {mujoco.__version__}')
" 2>/dev/null || echo "  FAIL: mujoco not importable"

python3 -c "
import gr00t
print(f'  gr00t:        OK')
" 2>/dev/null || echo "  FAIL: gr00t not importable"

python3 -c "
import transformers
print(f'  transformers: {transformers.__version__}')
" 2>/dev/null || echo "  FAIL: transformers not importable"

python3 -c "
import pyzmq
print(f'  pyzmq:        OK')
" 2>/dev/null || echo "  FAIL: pyzmq not importable"

echo ""
echo "  --- WBC venv check ---"
if [ -d "$WBC_VENV/.venv" ]; then
    "$WBC_VENV/.venv/bin/python" -c "
import mujoco
print(f'  WBC mujoco:   {mujoco.__version__}')
" 2>/dev/null || echo "  FAIL: mujoco not in WBC venv"
else
    echo "  WBC venv not found"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  # 1. Open-loop eval (offline, no server needed):"
echo "  cd /workspace/Isaac-GR00T"
echo "  python3 gr00t/eval/open_loop_eval.py \\"
echo "      --model-path $FOLD_TOWEL_CKPT \\"
echo "      --dataset-path $DATASETS_DIR/G1_Fold_Towel \\"
echo "      --embodiment-tag UNITREE_G1 \\"
echo "      --steps 300 --traj_ids 0 1 2"
echo ""
echo "  # 2. MuJoCo closed-loop eval (needs GROOT server):"
echo "  # Terminal 1 — start server:"
echo "  cd /workspace/Isaac-GR00T"
echo "  source $WBC_VENV/.venv/bin/activate"
echo "  python3 gr00t/eval/run_gr00t_server.py \\"
echo "      --model-path $FOLD_TOWEL_CKPT \\"
echo "      --embodiment-tag UNITREE_G1 \\"
echo "      --port 5555 --use-sim-policy-wrapper"
echo ""
echo "  # Terminal 2 — run rollout:"
echo "  source $WBC_VENV/.venv/bin/activate"
echo "  python3 gr00t/eval/rollout_policy.py \\"
echo "      --n_episodes 10 --max_episode_steps 1440 \\"
echo "      --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \\"
echo "      --n_action_steps 20 --n_envs 5"
echo ""
echo "  # 3. Custom MuJoCo towel scene (see scripts/eval/mujoco_towel_scene/):"
echo "  python3 /workspace/dm-isaac-g1/scripts/eval/run_mujoco_towel_eval.py \\"
echo "      --model-path $FOLD_TOWEL_CKPT \\"
echo "      --scene /workspace/dm-isaac-g1/scripts/eval/mujoco_towel_scene/g1_towel_folding.xml"
echo ""
