#!/bin/bash
# ============================================
# Run G1 Demo in Isaac Sim
# Phase 1: G1 reaching task demonstration
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

HOST="${WORKSTATION_HOST:-$WORKSTATION_IP}"
USER="$WORKSTATION_USER"
CONTAINER="isaac-sim"

echo "============================================"
echo "Phase 1: G1 Reach Demo"
echo "============================================"
echo "Host: $HOST"
echo "Container: $CONTAINER"
echo ""

# Function to run remote command
run_remote() {
    local cmd="$1"
    if [[ -n "$WORKSTATION_PASSWORD" ]] && command -v sshpass &> /dev/null; then
        sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$cmd"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$cmd"
    fi
}

# Check if container is running
echo "Checking Isaac Sim container..."
if ! run_remote "docker ps --format '{{.Names}}' | grep -q $CONTAINER"; then
    echo "Error: Isaac Sim container not running"
    exit 1
fi
echo "Container is running!"

# Run the G1 reach demo using existing environment
echo ""
echo "Starting G1 reach demonstration..."
echo "This will run the existing g1_reach environment with a random agent."
echo ""

run_remote "docker exec $CONTAINER bash -c '
cd /workspace/g1_reach
echo \"Running G1 reach with random agent...\"
python scripts/random_agent.py --task G1Reach-Direct-v0 --num_envs 16 --headless
'"

echo ""
echo "Demo complete!"
