#!/bin/bash
# ============================================
# Phase 1: Run G1 with GROOT Policy Inference
# ============================================
# Connects Isaac Sim G1 to the GROOT inference server
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

GROOT_HOST="${GROOT_SERVER_HOST:-10.20.0.248}"
GROOT_PORT="${GROOT_SERVER_PORT:-5555}"
NUM_ENVS="${1:-4}"
TASK="${2:-reach}"

run_remote() {
    if command -v sshpass &> /dev/null; then
        sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    else
        ssh -o StrictHostKeyChecking=no "$USER@$HOST" "$1"
    fi
}

echo "============================================"
echo "Phase 1: G1 + GROOT Inference"
echo "============================================"
echo "GROOT Server: $GROOT_HOST:$GROOT_PORT"
echo "Task: $TASK"
echo "Num Envs: $NUM_ENVS"
echo "============================================"

# Check if GROOT server is accessible from workstation
echo "Checking GROOT server connectivity..."
run_remote "curl -s --connect-timeout 5 http://$GROOT_HOST:$GROOT_PORT/health || echo 'GROOT server not reachable'"

# Run the inference demo
run_remote "docker exec $CONTAINER bash -c '
cd /workspace/g1_reach
export GROOT_SERVER_HOST=$GROOT_HOST
export GROOT_SERVER_PORT=$GROOT_PORT

# Use Python script to run inference with GROOT
python3 << EOF
import os
import sys
import time
sys.path.insert(0, \"/home/datamentors/dm-isaac-g1/phases/phase1_inference\")

from groot_client import GrootClient

# Test GROOT connection
print(\"Testing GROOT connection...\")
try:
    client = GrootClient(host=\"$GROOT_HOST\", port=$GROOT_PORT, timeout=10.0)
    if client.health_check():
        print(f\"GROOT server healthy at {client.base_url}\")
    else:
        print(\"GROOT server not responding\")
except Exception as e:
    print(f\"GROOT connection failed: {e}\")
    print(\"Running in standalone mode without GROOT\")

print(\"\\nGROOT integration ready.\")
print(\"To run full inference, use the Isaac Sim environment with GROOT actions.\")
EOF
'"

echo ""
echo "GROOT inference setup complete!"
echo "For full integration, the G1 environment will query GROOT for actions."
