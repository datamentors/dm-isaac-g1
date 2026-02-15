#!/bin/bash
# ============================================
# Connect to Blackwell Workstation via SSH
# ============================================
set -e

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Validate required variables
if [[ -z "$WORKSTATION_HOST" ]] && [[ -z "$WORKSTATION_IP" ]]; then
    echo "Error: WORKSTATION_HOST or WORKSTATION_IP must be set in .env"
    exit 1
fi

if [[ -z "$WORKSTATION_USER" ]]; then
    echo "Error: WORKSTATION_USER must be set in .env"
    exit 1
fi

# Use IP if HOST is not reachable
HOST="${WORKSTATION_HOST:-$WORKSTATION_IP}"
PORT="${WORKSTATION_SSH_PORT:-22}"
USER="$WORKSTATION_USER"

echo "============================================"
echo "Connecting to Blackwell Workstation"
echo "============================================"
echo "Host: $HOST"
echo "User: $USER"
echo "Port: $PORT"
echo "============================================"

# SSH connection
if [[ -n "$WORKSTATION_PASSWORD" ]]; then
    # Use sshpass if password is provided
    if command -v sshpass &> /dev/null; then
        sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST"
    else
        echo "Note: sshpass not installed. You'll be prompted for password."
        echo "To auto-login, install sshpass: brew install hudochenkov/sshpass/sshpass"
        ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST"
    fi
else
    ssh -o StrictHostKeyChecking=no -p "$PORT" "$USER@$HOST"
fi
