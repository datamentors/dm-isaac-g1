#!/bin/bash
# ============================================
# Phase 3: GROOT Fine-tuning with RL
# ============================================
# Fine-tunes GROOT model using demonstrations from RL policies
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
fi

HOST="${WORKSTATION_HOST:-192.168.1.205}"
USER="${WORKSTATION_USER:-datamentors}"
PASSWORD="${WORKSTATION_PASSWORD:-datamentors}"
CONTAINER="isaac-sim"

# Options
GROOT_CHECKPOINT="${1:-nvidia/GR00T-N1.6-3B}"
RL_CHECKPOINT="${2:-best}"
NUM_DEMOS="${3:-1000}"
TASK="${4:-manipulation}"

run_remote() {
    if command -v sshpass &> /dev/null; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    fi
}

echo "============================================"
echo "Phase 3: GROOT Fine-tuning + RL"
echo "============================================"
echo "GROOT Model: $GROOT_CHECKPOINT"
echo "RL Policy: $RL_CHECKPOINT"
echo "Num Demonstrations: $NUM_DEMOS"
echo "Task: $TASK"
echo "============================================"

echo ""
echo "Phase 3 Fine-tuning Process:"
echo "1. Load trained RL policy from Phase 2"
echo "2. Collect demonstrations using RL policy in Isaac Sim"
echo "3. Fine-tune GROOT model on demonstrations"
echo "4. Evaluate fine-tuned model"
echo ""

# Check for GROOT workspace
run_remote "
if [ -d /workspace/Isaac-GR00T ]; then
    echo 'Isaac-GR00T workspace found'
    ls /workspace/Isaac-GR00T/ | head -10
else
    echo 'Warning: Isaac-GR00T not found in /workspace'
fi
"

echo ""
echo "To run fine-tuning manually:"
echo ""
echo "1. Collect demonstrations:"
echo "   docker exec isaac-sim python /home/datamentors/dm-isaac-g1/phases/phase3_finetuning/collect_demos.py \\"
echo "     --rl-checkpoint \$CHECKPOINT \\"
echo "     --num-episodes $NUM_DEMOS"
echo ""
echo "2. Fine-tune GROOT:"
echo "   docker exec isaac-sim python /workspace/Isaac-GR00T/examples/finetune.py \\"
echo "     --model $GROOT_CHECKPOINT \\"
echo "     --data demos/"
echo ""
echo "Note: Full fine-tuning requires significant GPU memory (~40GB+)"
