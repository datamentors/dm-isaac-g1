#!/bin/bash
# ============================================
# Play/Evaluate Trained Policy
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
fi

HOST="${WORKSTATION_HOST:-192.168.1.205}"
USER="${WORKSTATION_USER:-datamentors}"
PASSWORD="${WORKSTATION_PASSWORD:-datamentors}"
CONTAINER="isaac-sim"

# Options
TRAINING_TYPE="${1:-g1_reach}"
CHECKPOINT="${2:-best}"  # best, latest, or path
NUM_ENVS="${3:-16}"
VIDEO="${4:-false}"

run_remote() {
    if command -v sshpass &> /dev/null; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    fi
}

echo "============================================"
echo "Playing Trained Policy"
echo "============================================"
echo "Training Type: $TRAINING_TYPE"
echo "Checkpoint: $CHECKPOINT"
echo "Num Envs: $NUM_ENVS"
echo "Record Video: $VIDEO"
echo "============================================"

VIDEO_FLAG=""
if [[ "$VIDEO" == "true" ]]; then
    VIDEO_FLAG="--video --video_length 500"
fi

case "$TRAINING_TYPE" in
    g1_reach)
        run_remote "docker exec $CONTAINER bash -c '
cd /workspace/g1_reach

# Find checkpoint
if [[ \"$CHECKPOINT\" == \"best\" ]]; then
    CKPT=\$(find logs/skrl -name \"best_agent.pt\" -type f | sort | tail -1)
elif [[ \"$CHECKPOINT\" == \"latest\" ]]; then
    CKPT=\$(find logs/skrl -name \"agent_*.pt\" -type f | sort | tail -1)
else
    CKPT=\"$CHECKPOINT\"
fi

echo \"Using checkpoint: \$CKPT\"

/isaac-sim/python.sh scripts/skrl/play.py \
    --task G1-Reach-Rl-Direct-v0 \
    --num_envs $NUM_ENVS \
    --checkpoint \"\$CKPT\" \
    --headless \
    $VIDEO_FLAG
'"
        ;;

    ulc_vlm)
        echo "ULC-VLM playback - select stage with third argument"
        STAGE="${CHECKPOINT}"  # Reuse checkpoint arg for stage
        run_remote "docker exec $CONTAINER bash -c '
cd /home/datamentors/dm-isaac-g1/isaac-g1-ulc-vlm
export PYTHONPATH=\$PYTHONPATH:/workspace/IsaacLab/source
/isaac-sim/python.sh g1/isaac_g1_ulc/play/play_ulc_stage_$STAGE.py \
    --num_envs $NUM_ENVS \
    --headless
'"
        ;;

    *)
        echo "Error: Unknown training type"
        exit 1
        ;;
esac
