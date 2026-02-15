#!/bin/bash
# ============================================
# Phase 2: Reinforcement Learning Training
# ============================================
# Supports both g1_reach and isaac-g1-ulc-vlm training
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
fi

# Default values
HOST="${WORKSTATION_HOST:-192.168.1.205}"
USER="${WORKSTATION_USER:-datamentors}"
PASSWORD="${WORKSTATION_PASSWORD:-datamentors}"
CONTAINER="isaac-sim"

# Training options
TRAINING_TYPE="${1:-g1_reach}"  # g1_reach or ulc_vlm
STAGE="${2:-1}"                  # Stage number for ulc_vlm
NUM_ENVS="${3:-4096}"
HEADLESS="${4:---headless}"

usage() {
    echo "Usage: $0 [training_type] [stage] [num_envs] [--headless|--no-headless]"
    echo ""
    echo "Training types:"
    echo "  g1_reach    - Train G1 reaching task (simple, recommended first)"
    echo "  ulc_vlm     - Train with isaac-g1-ulc-vlm curriculum (stages 1-8)"
    echo ""
    echo "Examples:"
    echo "  $0 g1_reach              # Train G1 reach"
    echo "  $0 ulc_vlm 1             # Train ULC stage 1 (standing)"
    echo "  $0 ulc_vlm 7 4096        # Train ULC stage 7 with 4096 envs"
    exit 1
}

run_remote() {
    if command -v sshpass &> /dev/null; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    fi
}

echo "============================================"
echo "Phase 2: RL Training"
echo "============================================"
echo "Host: $HOST"
echo "Training Type: $TRAINING_TYPE"
echo "Stage: $STAGE"
echo "Num Envs: $NUM_ENVS"
echo "============================================"

case "$TRAINING_TYPE" in
    g1_reach)
        echo "Starting G1 Reach training with skrl PPO..."
        run_remote "docker exec $CONTAINER bash -c '
cd /workspace/g1_reach
/isaac-sim/python.sh scripts/skrl/train.py \
    --task G1-Reach-Rl-Direct-v0 \
    --num_envs $NUM_ENVS \
    $HEADLESS
'"
        ;;

    ulc_vlm)
        echo "Starting ULC-VLM Stage $STAGE training..."

        # Map stage to training script
        case "$STAGE" in
            1) SCRIPT="train_ulc_stage_1.py" ;;
            2) SCRIPT="train_ulc_stage_2.py" ;;
            3) SCRIPT="train_ulc_stage_3.py" ;;
            4) SCRIPT="train_ulc_stage_4_arm.py" ;;
            5) SCRIPT="train_ulc_stage_5_arm.py" ;;
            6) SCRIPT="train_ulc_stage_6_unified.py" ;;
            7) SCRIPT="train_ulc_stage_7.py" ;;
            8) SCRIPT="train_ulc_stage_8.py" ;;
            *)
                echo "Error: Invalid stage $STAGE. Use 1-8."
                exit 1
                ;;
        esac

        run_remote "docker exec $CONTAINER bash -c '
cd /home/datamentors/dm-isaac-g1/isaac-g1-ulc-vlm
export PYTHONPATH=\$PYTHONPATH:/workspace/IsaacLab/source
/isaac-sim/python.sh g1/isaac_g1_ulc/train/$SCRIPT \
    --num_envs $NUM_ENVS \
    $HEADLESS
'"
        ;;

    *)
        echo "Error: Unknown training type '$TRAINING_TYPE'"
        usage
        ;;
esac

echo ""
echo "Training started! Monitor with:"
echo "  tensorboard --logdir logs/"
