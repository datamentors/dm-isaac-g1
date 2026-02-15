#!/bin/bash
# ============================================
# Check Blackwell Workstation Capabilities
# ============================================
# This script connects to the workstation and gathers system information
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

HOST="${WORKSTATION_HOST:-$WORKSTATION_IP}"
PORT="${WORKSTATION_SSH_PORT:-22}"
USER="$WORKSTATION_USER"

echo "============================================"
echo "Checking Blackwell Workstation Capabilities"
echo "============================================"
echo "Host: $HOST"
echo ""

# Function to run remote command
run_remote() {
    local cmd="$1"
    if [[ -n "$WORKSTATION_PASSWORD" ]] && command -v sshpass &> /dev/null; then
        sshpass -p "$WORKSTATION_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" "$USER@$HOST" "$cmd" 2>/dev/null
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" "$USER@$HOST" "$cmd" 2>/dev/null
    fi
}

# Check connectivity first
echo "Testing SSH connectivity..."
if ! run_remote "echo 'Connected!'" > /dev/null 2>&1; then
    echo "Error: Cannot connect to workstation. Check credentials and network."
    exit 1
fi
echo "SSH connection successful!"
echo ""

echo "============================================"
echo "SYSTEM INFORMATION"
echo "============================================"
run_remote "uname -a"
echo ""

echo "============================================"
echo "OS RELEASE"
echo "============================================"
run_remote "cat /etc/os-release | head -5"
echo ""

echo "============================================"
echo "CPU INFORMATION"
echo "============================================"
run_remote "lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread|Core|Socket'"
echo ""

echo "============================================"
echo "MEMORY INFORMATION"
echo "============================================"
run_remote "free -h"
echo ""

echo "============================================"
echo "GPU INFORMATION (nvidia-smi)"
echo "============================================"
run_remote "nvidia-smi"
echo ""

echo "============================================"
echo "GPU DETAILS"
echo "============================================"
run_remote "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv"
echo ""

echo "============================================"
echo "CUDA VERSION"
echo "============================================"
run_remote "nvcc --version 2>/dev/null || echo 'NVCC not in PATH'"
echo ""

echo "============================================"
echo "DOCKER STATUS"
echo "============================================"
run_remote "docker --version 2>/dev/null || echo 'Docker not installed'"
run_remote "docker info 2>/dev/null | grep -E 'Server Version|Runtimes|Default Runtime' || echo 'Cannot get docker info'"
echo ""

echo "============================================"
echo "NVIDIA CONTAINER TOOLKIT"
echo "============================================"
run_remote "nvidia-container-cli --version 2>/dev/null || echo 'NVIDIA Container Toolkit not installed'"
echo ""

echo "============================================"
echo "ISAAC SIM CHECK"
echo "============================================"
run_remote "ls -la /isaac-sim 2>/dev/null || echo 'Isaac Sim not found at /isaac-sim'"
run_remote "ls -la ~/.local/share/ov/pkg/ 2>/dev/null | head -10 || echo 'No Omniverse packages found'"
echo ""

echo "============================================"
echo "PYTHON ENVIRONMENT"
echo "============================================"
run_remote "python3 --version 2>/dev/null || echo 'Python3 not found'"
run_remote "pip3 --version 2>/dev/null || echo 'pip3 not found'"
echo ""

echo "============================================"
echo "DISK SPACE"
echo "============================================"
run_remote "df -h | grep -E '^/dev|Filesystem'"
echo ""

echo "============================================"
echo "NETWORK INTERFACES"
echo "============================================"
run_remote "ip addr show | grep -E 'inet |^[0-9]+:' | head -20"
echo ""

echo "============================================"
echo "Check Complete!"
echo "============================================"
