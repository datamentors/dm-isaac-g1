#!/bin/bash
# ============================================
# Run command inside Isaac Sim container
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

# Command to run inside container (passed as argument)
CMD="${1:-bash}"

echo "============================================"
echo "Running in Isaac Sim container: $CMD"
echo "============================================"

# Use sshpass if available
if [[ -n "$WORKSTATION_PASSWORD" ]] && command -v sshpass &> /dev/null; then
    sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no "$USER@$HOST" \
        "docker exec -it $CONTAINER bash -c '$CMD'"
else
    ssh -o StrictHostKeyChecking=no "$USER@$HOST" \
        "docker exec -it $CONTAINER bash -c '$CMD'"
fi
