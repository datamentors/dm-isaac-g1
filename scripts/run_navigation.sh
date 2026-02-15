#!/bin/bash
# ============================================
# Phase 4: Navigation + Inference + RL
# ============================================
# Full autonomous G1 with navigation, manipulation, and learned policies
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

GROOT_HOST="${GROOT_SERVER_HOST:-10.20.0.248}"
GROOT_PORT="${GROOT_SERVER_PORT:-5555}"

# Options
MAP_SIZE="${1:-20}"
NUM_WAYPOINTS="${2:-5}"
RL_CHECKPOINT="${3:-best}"

run_remote() {
    if command -v sshpass &> /dev/null; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    fi
}

echo "============================================"
echo "Phase 4: Navigation + Inference + RL"
echo "============================================"
echo "Map Size: ${MAP_SIZE}m x ${MAP_SIZE}m"
echo "Waypoints: $NUM_WAYPOINTS"
echo "RL Policy: $RL_CHECKPOINT"
echo "GROOT Server: $GROOT_HOST:$GROOT_PORT"
echo "============================================"

echo ""
echo "Phase 4 combines:"
echo "  - Navigation: High-level waypoint following"
echo "  - Locomotion RL: Trained from Phase 2"
echo "  - Manipulation: GROOT inference for object interaction"
echo ""

echo "This phase requires:"
echo "1. Trained locomotion policy (from Phase 2 ULC stages 1-3)"
echo "2. Trained arm reaching policy (from Phase 2 ULC stages 4-5)"
echo "3. GROOT server running for manipulation inference"
echo "4. Navigation environment with obstacles"
echo ""

# Check available policies
run_remote "
echo '=== Available RL Checkpoints ==='
find /workspace -name 'best_agent.pt' -o -name 'best*.pt' 2>/dev/null | head -10

echo ''
echo '=== Isaac Lab Navigation Environments ==='
ls /workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/ 2>/dev/null || echo 'Navigation tasks not found in standard location'
"

echo ""
echo "To run navigation demo:"
echo "  ./scripts/play_policy.sh ulc_vlm 6  # Stage 6 = loco-manipulation"
echo ""
echo "For full Phase 4 with custom navigation:"
echo "  docker exec isaac-sim python /home/datamentors/dm-isaac-g1/phases/phase4_navigation/navigation.py \\"
echo "    --map-size $MAP_SIZE $MAP_SIZE \\"
echo "    --num-waypoints $NUM_WAYPOINTS"
