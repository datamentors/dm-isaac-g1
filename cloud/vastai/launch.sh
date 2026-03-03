#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 Datamentors
# SPDX-License-Identifier: Apache-2.0
#
# Vast.ai 8-GPU Training — Full Lifecycle
#
# Provisions an 8-GPU instance on Vast.ai, runs setup + training,
# downloads the checkpoint, and destroys the instance.
#
# Prerequisites:
#   pip install vastai
#   Fill VASTAI_API_KEY in .env
#   SSH key at cloud/vastai/vastai_ssh_key (auto-generated if missing)
#
# Usage:
#   cd dm-isaac-g1
#   bash cloud/vastai/launch.sh              # Full lifecycle
#   bash cloud/vastai/launch.sh provision     # Only provision instance
#   bash cloud/vastai/launch.sh setup         # Only run setup on existing instance
#   bash cloud/vastai/launch.sh train         # Only run training
#   bash cloud/vastai/launch.sh download      # Only download checkpoint
#   bash cloud/vastai/launch.sh destroy       # Only destroy instance
#   bash cloud/vastai/launch.sh status        # Check instance status
#   bash cloud/vastai/launch.sh ssh           # SSH into instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Validate API key
if [ -z "$VASTAI_API_KEY" ]; then
    echo "ERROR: VASTAI_API_KEY not set. Add it to .env"
    exit 1
fi

vastai set api-key "$VASTAI_API_KEY" > /dev/null 2>&1

# SSH key setup
SSH_KEY="$SCRIPT_DIR/vastai_ssh_key"
if [ ! -f "$SSH_KEY" ]; then
    echo "Generating SSH key for Vast.ai..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "vastai-dm-isaac-g1"
    echo "SSH key generated at $SSH_KEY"
fi
SSH_PUB_KEY=$(cat "$SSH_KEY.pub")

# Instance state file (tracks current instance ID)
INSTANCE_FILE="$SCRIPT_DIR/.instance_id"

# ============================================
# Configuration
# ============================================
GPU_NAME="${VASTAI_GPU_NAME:-A100_SXM4}"
GPU_RAM_MIN="${VASTAI_GPU_RAM_MIN:-80}"
NUM_GPUS="${VASTAI_NUM_GPUS:-8}"
DISK_GB="${VASTAI_DISK_GB:-100}"
IMAGE="vastai/pytorch"
LOCAL_CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# ============================================
# Helper functions
# ============================================

get_instance_id() {
    if [ -f "$INSTANCE_FILE" ]; then
        cat "$INSTANCE_FILE"
    else
        echo ""
    fi
}

get_ssh_info() {
    local instance_id="$1"
    vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
host = data.get('ssh_host', '')
port = data.get('ssh_port', '')
if host and port:
    print(f'{host} {port}')
else:
    print('')
"
}

wait_for_instance() {
    local instance_id="$1"
    echo "Waiting for instance $instance_id to be ready..."
    for i in $(seq 1 60); do
        status=$(vastai show instance "$instance_id" --raw | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))")
        if [ "$status" = "running" ]; then
            echo "Instance is running!"
            return 0
        fi
        echo "  Status: $status (attempt $i/60)"
        sleep 10
    done
    echo "ERROR: Instance did not start within 10 minutes"
    return 1
}

ssh_cmd() {
    local instance_id="$1"
    shift
    local ssh_info
    ssh_info=$(get_ssh_info "$instance_id")
    if [ -z "$ssh_info" ]; then
        echo "ERROR: Could not get SSH info for instance $instance_id"
        return 1
    fi
    local host=$(echo "$ssh_info" | cut -d' ' -f1)
    local port=$(echo "$ssh_info" | cut -d' ' -f2)
    ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "root@$host" "$@"
}

scp_to() {
    local instance_id="$1"
    local src="$2"
    local dst="$3"
    local ssh_info
    ssh_info=$(get_ssh_info "$instance_id")
    local host=$(echo "$ssh_info" | cut -d' ' -f1)
    local port=$(echo "$ssh_info" | cut -d' ' -f2)
    scp -i "$SSH_KEY" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$src" "root@$host:$dst"
}

scp_from() {
    local instance_id="$1"
    local src="$2"
    local dst="$3"
    local ssh_info
    ssh_info=$(get_ssh_info "$instance_id")
    local host=$(echo "$ssh_info" | cut -d' ' -f1)
    local port=$(echo "$ssh_info" | cut -d' ' -f2)
    scp -i "$SSH_KEY" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -r "root@$host:$src" "$dst"
}

# ============================================
# Commands
# ============================================

cmd_provision() {
    echo "=== Searching for ${NUM_GPUS}x ${GPU_NAME} (${GPU_RAM_MIN}GB+) ==="

    # Search for offers sorted by price
    echo "Available offers:"
    vastai search offers "num_gpus=${NUM_GPUS} gpu_name=${GPU_NAME} gpu_ram>=${GPU_RAM_MIN} rentable=true inet_down>200 disk_space>=${DISK_GB}" \
        -o 'dph' --limit 5

    echo ""
    echo "Selecting cheapest offer..."

    # Get cheapest offer ID
    OFFER_ID=$(vastai search offers "num_gpus=${NUM_GPUS} gpu_name=${GPU_NAME} gpu_ram>=${GPU_RAM_MIN} rentable=true inet_down>200 disk_space>=${DISK_GB}" \
        -o 'dph' --limit 1 --raw | python3 -c "import sys,json; data=json.load(sys.stdin); print(data[0]['id'] if data else '')")

    if [ -z "$OFFER_ID" ]; then
        echo "ERROR: No suitable offers found. Try adjusting GPU_NAME or GPU_RAM_MIN."
        echo "  Current search: ${NUM_GPUS}x ${GPU_NAME} ${GPU_RAM_MIN}GB+"
        echo "  Tip: export VASTAI_GPU_NAME=A100 for any A100 variant"
        exit 1
    fi

    echo "Creating instance from offer $OFFER_ID..."
    INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK_GB" \
        --ssh \
        --direct \
        --env "-e WANDB_API_KEY=$WANDB_API_KEY -e HF_TOKEN=$HF_TOKEN" \
        --raw | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_contract',''))")

    if [ -z "$INSTANCE_ID" ]; then
        echo "ERROR: Failed to create instance"
        exit 1
    fi

    echo "$INSTANCE_ID" > "$INSTANCE_FILE"
    echo "Instance created: $INSTANCE_ID"

    wait_for_instance "$INSTANCE_ID"

    # Upload SSH key
    echo "Uploading SSH public key..."
    ssh_cmd "$INSTANCE_ID" "mkdir -p ~/.ssh && echo '$SSH_PUB_KEY' >> ~/.ssh/authorized_keys"

    echo ""
    echo "=== Instance $INSTANCE_ID is ready ==="
    echo "SSH: bash cloud/vastai/launch.sh ssh"
}

cmd_setup() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "ERROR: No instance found. Run: bash cloud/vastai/launch.sh provision"
        exit 1
    fi

    echo "=== Setting up instance $instance_id ==="

    # Upload setup and training scripts
    scp_to "$instance_id" "$SCRIPT_DIR/../setup_instance.sh" "/tmp/setup_instance.sh"
    scp_to "$instance_id" "$SCRIPT_DIR/../train_8gpu.sh" "/tmp/train_8gpu.sh"

    # Run setup
    ssh_cmd "$instance_id" "bash /tmp/setup_instance.sh"

    # Copy training script to workspace
    ssh_cmd "$instance_id" "cp /tmp/train_8gpu.sh /workspace/train_8gpu.sh && chmod +x /workspace/train_8gpu.sh"

    echo "=== Setup complete ==="
}

cmd_train() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "ERROR: No instance found. Run: bash cloud/vastai/launch.sh provision"
        exit 1
    fi

    echo "=== Starting 8-GPU training on instance $instance_id ==="

    # Set env vars and run training
    ssh_cmd "$instance_id" "export WANDB_API_KEY='$WANDB_API_KEY' && \
        export HF_TOKEN='$HF_TOKEN' && \
        mkdir -p /workspace/logs && \
        cd /workspace/Isaac-GR00T && \
        bash /workspace/train_8gpu.sh"

    echo "=== Training complete ==="
}

cmd_download() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "ERROR: No instance found."
        exit 1
    fi

    echo "=== Downloading checkpoint from instance $instance_id ==="

    mkdir -p "$LOCAL_CHECKPOINT_DIR"

    # Find the latest checkpoint
    CHECKPOINT_PATH=$(ssh_cmd "$instance_id" "ls -td /workspace/checkpoints/groot-g1-pnp-apple-8gpu/checkpoint-* 2>/dev/null | head -1")

    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "ERROR: No checkpoint found on instance"
        exit 1
    fi

    echo "Downloading: $CHECKPOINT_PATH"
    scp_from "$instance_id" "$CHECKPOINT_PATH" "$LOCAL_CHECKPOINT_DIR/"

    echo "Checkpoint saved to: $LOCAL_CHECKPOINT_DIR/$(basename $CHECKPOINT_PATH)"
    echo "=== Download complete ==="
}

cmd_destroy() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "No instance to destroy."
        return 0
    fi

    echo "Destroying instance $instance_id..."
    vastai destroy instance "$instance_id"
    rm -f "$INSTANCE_FILE"
    echo "Instance $instance_id destroyed."
}

cmd_status() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "No active instance."
        return 0
    fi

    echo "Instance: $instance_id"
    vastai show instance "$instance_id"
}

cmd_ssh() {
    local instance_id=$(get_instance_id)
    if [ -z "$instance_id" ]; then
        echo "ERROR: No instance found. Run: bash cloud/vastai/launch.sh provision"
        exit 1
    fi

    local ssh_info
    ssh_info=$(get_ssh_info "$instance_id")
    if [ -z "$ssh_info" ]; then
        echo "ERROR: Could not get SSH info"
        exit 1
    fi
    local host=$(echo "$ssh_info" | cut -d' ' -f1)
    local port=$(echo "$ssh_info" | cut -d' ' -f2)

    echo "Connecting to instance $instance_id..."
    ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$host"
}

cmd_full() {
    echo "============================================"
    echo "  Vast.ai 8-GPU Training — Full Lifecycle"
    echo "============================================"
    echo ""

    cmd_provision
    echo ""
    cmd_setup
    echo ""
    cmd_train
    echo ""
    cmd_download
    echo ""

    echo "Training complete. Destroy instance? (y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        cmd_destroy
    else
        echo "Instance kept alive. Don't forget to destroy it when done!"
        echo "  bash cloud/vastai/launch.sh destroy"
    fi
}

# ============================================
# Entry point
# ============================================

case "${1:-full}" in
    provision)  cmd_provision ;;
    setup)      cmd_setup ;;
    train)      cmd_train ;;
    download)   cmd_download ;;
    destroy)    cmd_destroy ;;
    status)     cmd_status ;;
    ssh)        cmd_ssh ;;
    full)       cmd_full ;;
    *)
        echo "Usage: bash cloud/vastai/launch.sh [command]"
        echo ""
        echo "Commands:"
        echo "  full        Full lifecycle: provision → setup → train → download → destroy"
        echo "  provision   Search and create an 8-GPU instance"
        echo "  setup       Install deps and download dataset on instance"
        echo "  train       Run 8-GPU training"
        echo "  download    Download checkpoint to local machine"
        echo "  destroy     Terminate and delete the instance"
        echo "  status      Show instance status"
        echo "  ssh         SSH into the instance"
        echo ""
        echo "Environment variables (set in .env or export):"
        echo "  VASTAI_API_KEY      Required. Your Vast.ai API key"
        echo "  VASTAI_GPU_NAME     GPU model to search for (default: A100_SXM4)"
        echo "  VASTAI_GPU_RAM_MIN  Minimum GPU RAM in GB (default: 80)"
        echo "  VASTAI_NUM_GPUS     Number of GPUs (default: 8)"
        echo "  VASTAI_DISK_GB      Disk space in GB (default: 100)"
        ;;
esac
