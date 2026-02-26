#!/usr/bin/env bash
# =============================================================================
# Run NVIDIA Official GR00T Evaluation
# =============================================================================
#
# Uses NVIDIA's rollout_policy.py with GR00T-WholeBodyControl + RoboCasa
# for proper Dex1 gripper handling, WBC balance, and scene physics.
#
# Architecture (two-process):
#   Terminal 1 (this script): Launches GROOT policy server
#   Terminal 2 (launched by this script): Runs simulation client
#
# Prerequisites:
#   - Run setup_official_eval.sh first
#   - GROOT checkpoint available (local or HuggingFace)
#
# Usage:
#   # With NVIDIA's pre-trained PnPAppleToPlate checkpoint:
#   bash scripts/eval/run_official_eval.sh
#
#   # With our finetuned checkpoint:
#   bash scripts/eval/run_official_eval.sh --model /workspace/checkpoints/groot_g1_gripper
#
#   # With custom task language:
#   bash scripts/eval/run_official_eval.sh --model /workspace/checkpoints/groot_g1_gripper \
#       --episodes 20 --env gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc
# =============================================================================

set -euo pipefail

# Defaults
MODEL_PATH="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG="UNITREE_G1"
ENV_NAME="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
N_EPISODES=10
N_ACTION_STEPS=20
MAX_EPISODE_STEPS=1440
N_ENVS=1  # Use 1 for single GPU; increase if you have more VRAM
SERVER_PORT=5555

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --tag)        EMBODIMENT_TAG="$2"; shift 2 ;;
        --env)        ENV_NAME="$2"; shift 2 ;;
        --episodes)   N_EPISODES="$2"; shift 2 ;;
        --steps)      MAX_EPISODE_STEPS="$2"; shift 2 ;;
        --n-envs)     N_ENVS="$2"; shift 2 ;;
        --port)       SERVER_PORT="$2"; shift 2 ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

GROOT_DIR="/workspace/Isaac-GR00T"
WBC_VENV="$GROOT_DIR/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
LOG_DIR="/tmp/groot_official_eval_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "================================================================="
echo "  NVIDIA Official GR00T Evaluation"
echo "================================================================="
echo "  Model:     $MODEL_PATH"
echo "  Embodiment: $EMBODIMENT_TAG"
echo "  Env:       $ENV_NAME"
echo "  Episodes:  $N_EPISODES"
echo "  Action steps: $N_ACTION_STEPS"
echo "  Max steps: $MAX_EPISODE_STEPS"
echo "  N envs:    $N_ENVS"
echo "  Log dir:   $LOG_DIR"
echo "================================================================="

cd "$GROOT_DIR"

# Check WBC venv exists
if [ ! -f "$WBC_VENV" ]; then
    echo "ERROR: WBC virtual environment not found at $WBC_VENV"
    echo "Run setup_official_eval.sh first."
    exit 1
fi

# Launch GROOT policy server in background
echo ""
echo "[1/2] Launching GROOT policy server..."
python -u gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --port "$SERVER_PORT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID (log: $LOG_DIR/server.log)"

# Wait for server to be ready
echo "  Waiting for server to start..."
for i in $(seq 1 60); do
    if grep -q "Server started\|Serving\|Listening\|ready" "$LOG_DIR/server.log" 2>/dev/null; then
        echo "  Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Check $LOG_DIR/server.log"
        cat "$LOG_DIR/server.log"
        exit 1
    fi
    sleep 1
done

# Launch simulation client
echo ""
echo "[2/2] Launching simulation client..."
$WBC_VENV gr00t/eval/rollout_policy.py \
    --n_episodes "$N_EPISODES" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --env_name "$ENV_NAME" \
    --policy_client_host localhost \
    --policy_client_port "$SERVER_PORT" \
    --n_action_steps "$N_ACTION_STEPS" \
    --n_envs "$N_ENVS" \
    2>&1 | tee "$LOG_DIR/client.log"

# Capture exit code
CLIENT_EXIT=$?

# Cleanup
echo ""
echo "Stopping server (PID $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "================================================================="
echo "  Evaluation Complete"
echo "  Logs: $LOG_DIR/"
echo "  Videos: check /tmp/sim_eval_videos_*/"
echo "================================================================="

exit $CLIENT_EXIT
