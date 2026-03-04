#!/usr/bin/env bash
# =============================================================================
# ECS Training Job Runner
# =============================================================================
# Submits GPU training jobs to the ECS cluster. The cluster auto-scales
# GPU instances on demand and scales back to 0 when idle (zero cost).
#
# Usage:
#   ./run.sh <command> [options]
#
# Commands:
#   submit    Upload data + register task + run on ECS
#   replay    Run replay/export on a trained model (export ONNX/JIT + record video)
#   sim2sim   Validate exported policy in MuJoCo (Isaac Lab → MuJoCo transfer)
#   shell     Launch a long-running GPU container for interactive work
#   exec      Get a bash shell into a running container (ECS Exec)
#   status    Check running/pending tasks
#   logs      Stream CloudWatch logs for a task
#   list      List recent tasks
#   stop      Stop a running task
#   download  Download checkpoints from S3
#
# Examples:
#   ./run.sh submit --task mimic --motion cr7_06_tiktok_uefa
#   ./run.sh submit --task mimic --motion cr7_06_tiktok_uefa --max-iterations 50000
#   ./run.sh submit --task rl --task-id DM-G1-29dof-FALCON
#   ./run.sh submit --task rl --task-id DM-G1-29dof-SoFTA --max-iterations 30000
#   ./run.sh replay --task mimic --motion cr7_06_tiktok_uefa
#   ./run.sh replay --task rl --task-id DM-G1-29dof-FALCON
#   ./run.sh sim2sim --task rl --task-id DM-G1-29dof-FALCON
#   ./run.sh shell                          # launch interactive GPU container
#   ./run.sh exec --task-arn <arn>           # get bash shell into running container
#   ./run.sh status
#   ./run.sh logs --task-arn <arn>
#   ./run.sh download --motion cr7_06_tiktok_uefa
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env and cluster config
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
if [[ -f "$SCRIPT_DIR/.cluster-config" ]]; then
    source "$SCRIPT_DIR/.cluster-config"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
AWS_PROFILE="${AWS_PROFILE:-elianomarques-dm}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
ECR_REGISTRY="${AWS_ECR_REGISTRY:-260464233120.dkr.ecr.eu-west-1.amazonaws.com}"
ECR_REPO="${AWS_ECR_REPO:-isaac-g1-sim-ft-rl}"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:latest"
CLUSTER_NAME="${CLUSTER_NAME:-dm-isaac-g1-gpu}"
CAPACITY_PROVIDER_NAME="${CAPACITY_PROVIDER_NAME:-dm-isaac-g1-gpu-cp}"
S3_BUCKET="${S3_BUCKET:-dm-isaac-g1-training-${AWS_REGION}}"
ACCOUNT_ID="${ACCOUNT_ID:-260464233120}"
TASK_ROLE_ARN="${TASK_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/dm-isaac-g1-ecs-task}"
EXEC_ROLE_ARN="${EXEC_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/dm-isaac-g1-ecs-exec}"

TASK_TYPE=""
MOTION_NAME=""
TASK_ID=""
MAX_ITERATIONS=""
TASK_ARN=""
CHECKPOINT_FILE=""
VIDEO_LENGTH=""
HF_REPO=""
HEADLESS="true"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

aws_cmd() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# Vulkan diagnostic command (shared across all containers)
# The host UserData upgrades nvidia-container-toolkit to >= 1.12.0 which automatically
# mounts Vulkan/graphics libs (libnvidia-vulkan-producer.so, nvidia_icd.json, etc.)
# into containers when NVIDIA_DRIVER_CAPABILITIES=all. No manual bind-mounts needed.
# This command just creates required dirs and logs Vulkan state for debugging.
VULKAN_SETUP_CMD='mkdir -p /tmp/xdg 2>/dev/null; echo "=== Vulkan diagnostics ==="; ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || true; ls -la /etc/vulkan/icd.d/ 2>/dev/null || true; ldconfig -p 2>/dev/null | grep -i vulkan || true; ldconfig -p 2>/dev/null | grep -i nvidia | head -5 || true'

# VNC + XFCE4 desktop startup command (shared across all containers)
# Starts TurboVNC on :1 (port 5901), then launches XFCE4 desktop in background.
# Uses -noxstartup to bypass TurboVNC's session detection (which fails to find xfce.desktop).
VNC_STARTUP_CMD='rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null; /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 -noxstartup 2>/dev/null; export DISPLAY=:1; (nohup dbus-launch startxfce4 &>/dev/null &)'

# ── Upload Training Data ─────────────────────────────────────────────────────
upload_data() {
    local task="$1"
    local motion="$2"

    log "Uploading training data to S3..."

    if [[ "$task" == "mimic" ]]; then
        local task_dir="$REPO_ROOT/src/dm_isaac_g1/mimic/tasks/${motion}"

        # If files not local, pull from workstation
        if [[ ! -f "$task_dir/${motion}.npz" ]]; then
            warn "NPZ not found locally at $task_dir/${motion}.npz"
            warn "Pulling from workstation..."
            mkdir -p "$task_dir"
            sshpass -p "${WORKSTATION_PASSWORD}" ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
                "${WORKSTATION_USER}@${WORKSTATION_HOST}" \
                "docker cp dm-workstation:/workspace/dm-isaac-g1/src/dm_isaac_g1/mimic/tasks/${motion}/. /tmp/ecs_${motion}/"
            sshpass -p "${WORKSTATION_PASSWORD}" scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
                -r "${WORKSTATION_USER}@${WORKSTATION_HOST}:/tmp/ecs_${motion}/*" "$task_dir/"
        fi

        # Upload task data to S3
        aws_cmd s3 sync "$task_dir/" "s3://${S3_BUCKET}/tasks/mimic/${motion}/" \
            --exclude "*.pyc" --exclude "__pycache__/*"
        log "Uploaded: s3://${S3_BUCKET}/tasks/mimic/${motion}/"
    fi

    # Upload training/replay/sim2sim scripts
    aws_cmd s3 cp "$SCRIPT_DIR/train_mimic.sh" "s3://${S3_BUCKET}/scripts/train_mimic.sh"
    aws_cmd s3 cp "$SCRIPT_DIR/train_rl.sh" "s3://${S3_BUCKET}/scripts/train_rl.sh"
    aws_cmd s3 cp "$SCRIPT_DIR/replay.sh" "s3://${S3_BUCKET}/scripts/replay.sh"
    aws_cmd s3 cp "$SCRIPT_DIR/sim2sim.sh" "s3://${S3_BUCKET}/scripts/sim2sim.sh"
    log "Uploaded training/replay/sim2sim scripts"
}

# ── Register Task Definition ─────────────────────────────────────────────────
register_task_def() {
    local task="$1"
    local motion="$2"
    local max_iter="$3"
    local task_id="$4"
    local family="dm-${task}-${motion}"

    # Select training script based on task type
    local train_script="train_mimic.sh"
    local wandb_project="dm-isaac-g1"
    if [[ "$task" == "rl" ]]; then
        train_script="train_rl.sh"
        wandb_project="dm-isaac-g1-rl"
    fi

    log "Registering task definition: $family (task_id=$task_id, script=$train_script)" >&2

    local tmpfile
    tmpfile=$(mktemp /tmp/ecs-taskdef-XXXXXX.json)
    cat > "$tmpfile" << EOF
{
    "family": "${family}",
    "taskRoleArn": "${TASK_ROLE_ARN}",
    "executionRoleArn": "${EXEC_ROLE_ARN}",
    "networkMode": "host",
    "requiresCompatibilities": ["EC2"],
    "containerDefinitions": [
        {
            "name": "training",
            "image": "${ECR_IMAGE}",
            "essential": true,
            "resourceRequirements": [
                {"type": "GPU", "value": "1"}
            ],
            "memory": 28000,
            "cpu": 6144,
            "environment": [
                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"},
                {"name": "VK_ICD_FILENAMES", "value": "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"},
                {"name": "XDG_RUNTIME_DIR", "value": "/tmp/xdg"},
                {"name": "ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ALLOW_ROOT", "value": "1"},
                {"name": "MOTION_NAME", "value": "${motion}"},
                {"name": "TASK_ID", "value": "${task_id}"},
                {"name": "MAX_ITERATIONS", "value": "${max_iter}"},
                {"name": "S3_BUCKET", "value": "${S3_BUCKET}"},
                {"name": "AWS_REGION", "value": "${AWS_REGION}"},
                {"name": "HF_TOKEN", "value": "${HF_TOKEN:-}"},
                {"name": "WANDB_API_KEY", "value": "${WANDB_API_KEY:-}"},
                {"name": "WANDB_PROJECT", "value": "${WANDB_PROJECT:-${wandb_project}}"},
                {"name": "GITHUB_TOKEN", "value": "${GITHUB_TOKEN:-}"}
            ],
            "command": ["bash", "-c", "${VULKAN_SETUP_CMD}; ${VNC_STARTUP_CMD}; echo 'VNC+XFCE started on :5901'; bash /opt/training/scripts/${train_script}"],
            "dependsOn": [
                {"containerName": "data-sync", "condition": "SUCCESS"}
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "${task}-${motion}"
                }
            },
            "linuxParameters": {
                "initProcessEnabled": true,
                "sharedMemorySize": 8192,
                "devices": [
                    {"hostPath": "/dev/dri", "containerPath": "/dev/dri", "permissions": ["read", "write", "mknod"]}
                ]
            }
        },
        {
            "name": "data-sync",
            "image": "amazon/aws-cli:latest",
            "essential": false,
            "memory": 512,
            "cpu": 256,
            "command": [
                "s3", "sync",
                "s3://${S3_BUCKET}/scripts/", "/opt/training/scripts/",
                "--region", "${AWS_REGION}"
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "data-sync"
                }
            }
        }
    ],
    "volumes": [
        {"name": "training-data", "host": {"sourcePath": "/opt/training"}}
    ]
}
EOF

    local revision
    revision=$(aws_cmd ecs register-task-definition \
        --cli-input-json "file://${tmpfile}" \
        --query "taskDefinition.revision" --output text)
    rm -f "$tmpfile"

    log "Registered: ${family}:${revision}" >&2
    echo "${family}:${revision}"
}

# ── Submit Task ───────────────────────────────────────────────────────────────
cmd_submit() {
    [[ -z "$TASK_TYPE" ]] && { err "Must specify --task (e.g., mimic, rl)"; exit 1; }

    if [[ "$TASK_TYPE" == "rl" ]]; then
        # RL tasks: require --task-id, derive motion name from it
        [[ -z "$TASK_ID" ]] && { err "RL tasks require --task-id (e.g., DM-G1-29dof-FALCON)"; exit 1; }
        # Use task ID as the "motion" identifier for ECS family naming
        if [[ -z "$MOTION_NAME" ]]; then
            MOTION_NAME=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//' | tr '[:upper:]' '[:lower:]')
        fi
    else
        # Mimic tasks: require --motion
        [[ -z "$MOTION_NAME" ]] && { err "Must specify --motion (e.g., cr7_06_tiktok_uefa)"; exit 1; }

        # Auto-detect task ID from the task's __init__.py (gym.register id)
        if [[ -z "$TASK_ID" ]]; then
            local init_py="$REPO_ROOT/src/dm_isaac_g1/mimic/tasks/${MOTION_NAME}/__init__.py"
            if [[ -f "$init_py" ]]; then
                TASK_ID=$(sed -n 's/.*id="\([^"]*\)".*/\1/p' "$init_py" | head -1)
            fi
        fi
        if [[ -z "$TASK_ID" ]]; then
            TASK_ID="DM-G1-29dof-Mimic-${MOTION_NAME}"
            warn "Could not auto-detect task ID, using: $TASK_ID"
        fi
    fi

    # Default iterations: 50000 for RL, 30000 for mimic
    if [[ -z "$MAX_ITERATIONS" ]]; then
        if [[ "$TASK_TYPE" == "rl" ]]; then
            MAX_ITERATIONS=50000
        else
            MAX_ITERATIONS=30000
        fi
    fi

    log "=== Submitting ECS Training Job ==="
    log "Task: $TASK_TYPE, Name: $MOTION_NAME, Task ID: $TASK_ID, Iterations: $MAX_ITERATIONS"

    # Upload data
    upload_data "$TASK_TYPE" "$MOTION_NAME"

    # Register task definition
    local task_def_rev
    task_def_rev=$(register_task_def "$TASK_TYPE" "$MOTION_NAME" "$MAX_ITERATIONS" "$TASK_ID")

    # Run the task
    log "Submitting task to ECS cluster..."
    local task_arn
    task_arn=$(aws_cmd ecs run-task \
        --cluster "$CLUSTER_NAME" \
        --task-definition "$task_def_rev" \
        --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1" \
        --count 1 \
        --enable-execute-command \
        --started-by "run.sh" \
        --tags "key=TaskId,value=$TASK_ID" "key=Task,value=$TASK_TYPE" \
        --query "tasks[0].taskArn" --output text)

    log ""
    log "=== Task Submitted ==="
    log "Task ARN:    $task_arn"
    log "Cluster:     $CLUSTER_NAME"
    log "Definition:  $task_def_rev"
    log ""
    log "ECS will now auto-scale a GPU instance (~3-5 min)."
    log "Monitor with:"
    log "  $0 status"
    log "  $0 logs --task-arn $task_arn"
    log ""
    if [[ "$TASK_TYPE" == "rl" ]]; then
        local task_clean=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
        log "When training completes, checkpoints will be in:"
        log "  s3://${S3_BUCKET}/Models/RL/${task_clean}/"
    else
        log "When training completes, checkpoints will be in:"
        log "  s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/"
        log "Download with:"
        log "  $0 download --motion $MOTION_NAME"
    fi

    # Save for convenience
    echo "$task_arn" > "$SCRIPT_DIR/.last-task-arn"
}

# ── Replay (export + video) ────────────────────────────────────────────────────
cmd_replay() {
    [[ -z "$TASK_TYPE" ]] && { err "Must specify --task (mimic or rl)"; exit 1; }

    if [[ "$TASK_TYPE" == "rl" ]]; then
        [[ -z "$TASK_ID" ]] && { err "RL replay requires --task-id (e.g., DM-G1-29dof-FALCON)"; exit 1; }
        if [[ -z "$MOTION_NAME" ]]; then
            MOTION_NAME=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//' | tr '[:upper:]' '[:lower:]')
        fi
    else
        [[ -z "$MOTION_NAME" ]] && { err "Mimic replay requires --motion"; exit 1; }
        if [[ -z "$TASK_ID" ]]; then
            local init_py="$REPO_ROOT/src/dm_isaac_g1/mimic/tasks/${MOTION_NAME}/__init__.py"
            if [[ -f "$init_py" ]]; then
                TASK_ID=$(sed -n 's/.*id="\([^"]*\)".*/\1/p' "$init_py" | head -1)
            fi
        fi
        [[ -z "$TASK_ID" ]] && { err "Could not determine task ID for motion: $MOTION_NAME"; exit 1; }
    fi

    log "=== Submitting ECS Replay Job ==="
    log "Task: $TASK_TYPE, Motion: $MOTION_NAME, Task ID: $TASK_ID"

    # Upload scripts (including replay.sh)
    upload_data "$TASK_TYPE" "$MOTION_NAME"

    # Register a task definition for replay
    local family="dm-replay-${MOTION_NAME}"

    local tmpfile
    tmpfile=$(mktemp /tmp/ecs-taskdef-XXXXXX.json)
    cat > "$tmpfile" << EOF
{
    "family": "${family}",
    "taskRoleArn": "${TASK_ROLE_ARN}",
    "executionRoleArn": "${EXEC_ROLE_ARN}",
    "networkMode": "host",
    "requiresCompatibilities": ["EC2"],
    "containerDefinitions": [
        {
            "name": "replay",
            "image": "${ECR_IMAGE}",
            "essential": true,
            "resourceRequirements": [
                {"type": "GPU", "value": "1"}
            ],
            "memory": 28000,
            "cpu": 6144,
            "environment": [
                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"},
                {"name": "VK_ICD_FILENAMES", "value": "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"},
                {"name": "XDG_RUNTIME_DIR", "value": "/tmp/xdg"},
                {"name": "ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ALLOW_ROOT", "value": "1"},
                {"name": "TASK_TYPE", "value": "${TASK_TYPE}"},
                {"name": "TASK_ID", "value": "${TASK_ID}"},
                {"name": "MOTION_NAME", "value": "${MOTION_NAME}"},
                {"name": "S3_BUCKET", "value": "${S3_BUCKET}"},
                {"name": "AWS_REGION", "value": "${AWS_REGION}"},
                {"name": "CHECKPOINT_FILE", "value": "${CHECKPOINT_FILE:-}"},
                {"name": "VIDEO_LENGTH", "value": "${VIDEO_LENGTH:-300}"},
                {"name": "HF_TOKEN", "value": "${HF_TOKEN:-}"},
                {"name": "HF_REPO", "value": "${HF_REPO:-}"},
                {"name": "GITHUB_TOKEN", "value": "${GITHUB_TOKEN:-}"}
            ],
            "command": ["bash", "-c", "${VULKAN_SETUP_CMD}; ${VNC_STARTUP_CMD}; bash /opt/training/scripts/replay.sh"],
            "dependsOn": [
                {"containerName": "data-sync", "condition": "SUCCESS"}
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "replay-${MOTION_NAME}"
                }
            },
            "linuxParameters": {
                "initProcessEnabled": true,
                "sharedMemorySize": 8192,
                "devices": [
                    {"hostPath": "/dev/dri", "containerPath": "/dev/dri", "permissions": ["read", "write", "mknod"]}
                ]
            }
        },
        {
            "name": "data-sync",
            "image": "amazon/aws-cli:latest",
            "essential": false,
            "memory": 512,
            "cpu": 256,
            "command": [
                "s3", "sync",
                "s3://${S3_BUCKET}/scripts/", "/opt/training/scripts/",
                "--region", "${AWS_REGION}"
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "data-sync"
                }
            }
        }
    ],
    "volumes": [
        {"name": "training-data", "host": {"sourcePath": "/opt/training"}}
    ]
}
EOF

    local revision
    revision=$(aws_cmd ecs register-task-definition \
        --cli-input-json "file://${tmpfile}" \
        --query "taskDefinition.revision" --output text)
    rm -f "$tmpfile"
    log "Registered: ${family}:${revision}"

    # Run the task
    local task_arn
    task_arn=$(aws_cmd ecs run-task \
        --cluster "$CLUSTER_NAME" \
        --task-definition "${family}:${revision}" \
        --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1" \
        --count 1 \
        --enable-execute-command \
        --started-by "run.sh-replay" \
        --tags "key=TaskId,value=$TASK_ID" "key=Task,value=replay" \
        --query "tasks[0].taskArn" --output text)

    echo "$task_arn" > "$SCRIPT_DIR/.last-task-arn"

    log ""
    log "=== Replay Job Submitted ==="
    log "Task ARN:    $task_arn"
    log "Cluster:     $CLUSTER_NAME"
    log "Definition:  ${family}:${revision}"
    log ""
    log "Monitor with:"
    log "  $0 status"
    log "  $0 logs --task-arn $task_arn"
    log ""
    if [[ "$TASK_TYPE" == "rl" ]]; then
        local task_clean=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
        log "Results will be in: s3://${S3_BUCKET}/Models/RL/${task_clean}/exported/"
    else
        log "Results will be in: s3://${S3_BUCKET}/Models/IL/Mimic-${MOTION_NAME}/exported/"
    fi
}

# ── Sim2Sim (Isaac Lab → MuJoCo) ─────────────────────────────────────────────
cmd_sim2sim() {
    [[ -z "$TASK_TYPE" ]] && { err "Must specify --task (mimic or rl)"; exit 1; }

    if [[ "$TASK_TYPE" == "rl" ]]; then
        [[ -z "$TASK_ID" ]] && { err "RL sim2sim requires --task-id"; exit 1; }
        if [[ -z "$MOTION_NAME" ]]; then
            MOTION_NAME=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//' | tr '[:upper:]' '[:lower:]')
        fi
    else
        [[ -z "$MOTION_NAME" ]] && { err "Mimic sim2sim requires --motion"; exit 1; }
        if [[ -z "$TASK_ID" ]]; then
            local init_py="$REPO_ROOT/src/dm_isaac_g1/mimic/tasks/${MOTION_NAME}/__init__.py"
            if [[ -f "$init_py" ]]; then
                TASK_ID=$(sed -n 's/.*id="\([^"]*\)".*/\1/p' "$init_py" | head -1)
            fi
        fi
        [[ -z "$TASK_ID" ]] && { err "Could not determine task ID for motion: $MOTION_NAME"; exit 1; }
    fi

    log "=== Submitting ECS Sim2Sim Job ==="
    log "Task: $TASK_TYPE, Motion: $MOTION_NAME, Task ID: $TASK_ID"

    upload_data "$TASK_TYPE" "$MOTION_NAME"

    local family="dm-sim2sim-${MOTION_NAME}"
    local tmpfile
    tmpfile=$(mktemp /tmp/ecs-taskdef-XXXXXX.json)
    cat > "$tmpfile" << EOF
{
    "family": "${family}",
    "taskRoleArn": "${TASK_ROLE_ARN}",
    "executionRoleArn": "${EXEC_ROLE_ARN}",
    "networkMode": "host",
    "requiresCompatibilities": ["EC2"],
    "containerDefinitions": [
        {
            "name": "sim2sim",
            "image": "${ECR_IMAGE}",
            "essential": true,
            "resourceRequirements": [
                {"type": "GPU", "value": "1"}
            ],
            "memory": 28000,
            "cpu": 6144,
            "environment": [
                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"},
                {"name": "VK_ICD_FILENAMES", "value": "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"},
                {"name": "XDG_RUNTIME_DIR", "value": "/tmp/xdg"},
                {"name": "ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ALLOW_ROOT", "value": "1"},
                {"name": "TASK_TYPE", "value": "${TASK_TYPE}"},
                {"name": "TASK_ID", "value": "${TASK_ID}"},
                {"name": "MOTION_NAME", "value": "${MOTION_NAME}"},
                {"name": "S3_BUCKET", "value": "${S3_BUCKET}"},
                {"name": "AWS_REGION", "value": "${AWS_REGION}"},
                {"name": "VIDEO_LENGTH", "value": "${VIDEO_LENGTH:-10}"},
                {"name": "HEADLESS", "value": "${HEADLESS}"},
                {"name": "GITHUB_TOKEN", "value": "${GITHUB_TOKEN:-}"}
            ],
            "command": ["bash", "-c", "${VULKAN_SETUP_CMD}; ${VNC_STARTUP_CMD}; echo 'VNC+XFCE started on :5901'; bash /opt/training/scripts/sim2sim.sh"],
            "dependsOn": [
                {"containerName": "data-sync", "condition": "SUCCESS"}
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "sim2sim-${MOTION_NAME}"
                }
            },
            "linuxParameters": {
                "initProcessEnabled": true,
                "sharedMemorySize": 8192,
                "devices": [
                    {"hostPath": "/dev/dri", "containerPath": "/dev/dri", "permissions": ["read", "write", "mknod"]}
                ]
            }
        },
        {
            "name": "data-sync",
            "image": "amazon/aws-cli:latest",
            "essential": false,
            "memory": 512,
            "cpu": 256,
            "command": [
                "s3", "sync",
                "s3://${S3_BUCKET}/scripts/", "/opt/training/scripts/",
                "--region", "${AWS_REGION}"
            ],
            "mountPoints": [
                {"sourceVolume": "training-data", "containerPath": "/opt/training"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "data-sync"
                }
            }
        }
    ],
    "volumes": [
        {"name": "training-data", "host": {"sourcePath": "/opt/training"}}
    ]
}
EOF

    local revision
    revision=$(aws_cmd ecs register-task-definition \
        --cli-input-json "file://${tmpfile}" \
        --query "taskDefinition.revision" --output text)
    rm -f "$tmpfile"
    log "Registered: ${family}:${revision}"

    local task_arn
    task_arn=$(aws_cmd ecs run-task \
        --cluster "$CLUSTER_NAME" \
        --task-definition "${family}:${revision}" \
        --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1" \
        --count 1 \
        --enable-execute-command \
        --started-by "run.sh-sim2sim" \
        --tags "key=TaskId,value=$TASK_ID" "key=Task,value=sim2sim" \
        --query "tasks[0].taskArn" --output text)

    echo "$task_arn" > "$SCRIPT_DIR/.last-task-arn"

    log ""
    log "=== Sim2Sim Job Submitted ==="
    log "Task ARN:    $task_arn"
    log ""
    log "Monitor: $0 logs --task-arn $task_arn"
    if [[ "$TASK_TYPE" == "rl" ]]; then
        local task_clean=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
        log "Videos will be in: s3://${S3_BUCKET}/Models/RL/${task_clean}/sim2sim/"
    else
        log "Videos will be in: s3://${S3_BUCKET}/Models/IL/Mimic-${MOTION_NAME}/sim2sim/"
    fi
}

# ── Status ────────────────────────────────────────────────────────────────────
cmd_status() {
    log "=== ECS Cluster Status ==="

    # Cluster info
    local cluster_info
    cluster_info=$(aws_cmd ecs describe-clusters --clusters "$CLUSTER_NAME" \
        --query "clusters[0].{Running:runningTasksCount,Pending:pendingTasksCount,Instances:registeredContainerInstancesCount}" \
        --output json 2>/dev/null)
    echo -e "${BLUE}Cluster:${NC}    $CLUSTER_NAME"
    echo "$cluster_info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  Instances: {d[\"Instances\"]}, Running tasks: {d[\"Running\"]}, Pending: {d[\"Pending\"]}')"

    # List tasks
    echo ""
    echo -e "${BLUE}Running tasks:${NC}"
    aws_cmd ecs list-tasks --cluster "$CLUSTER_NAME" --desired-status RUNNING \
        --query "taskArns[]" --output text | tr '\t' '\n' | while read -r arn; do
        [[ -z "$arn" ]] && continue
        local info
        info=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
            --query "tasks[0].{Status:lastStatus,Def:taskDefinitionArn,Started:startedAt,Stopped:stoppedAt}" --output json 2>/dev/null)
        echo "  $arn"
        echo "$info" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'    Status: {d[\"Status\"]}, Def: {d[\"Def\"].split(\"/\")[-1]}')" 2>/dev/null
    done || echo "  (none)"

    echo ""
    echo -e "${BLUE}Pending tasks:${NC}"
    aws_cmd ecs list-tasks --cluster "$CLUSTER_NAME" --desired-status PENDING \
        --query "taskArns[]" --output text | tr '\t' '\n' | while read -r arn; do
        [[ -z "$arn" ]] && continue
        echo "  $arn"
    done || echo "  (none)"

    # ASG info
    echo ""
    echo -e "${BLUE}Auto Scaling Group:${NC}"
    aws_cmd autoscaling describe-auto-scaling-groups --auto-scaling-group-names "dm-isaac-g1-gpu-asg" \
        --query "AutoScalingGroups[0].{Desired:DesiredCapacity,Min:MinSize,Max:MaxSize,Instances:Instances[*].{Id:InstanceId,State:LifecycleState}}" \
        --output json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  Desired: {d[\"Desired\"]}, Min: {d[\"Min\"]}, Max: {d[\"Max\"]}')
for i in (d.get('Instances') or []):
    print(f'  Instance: {i[\"Id\"]} ({i[\"State\"]})')
if not d.get('Instances'):
    print('  (no instances - cluster is scaled to zero)')
" 2>/dev/null
}

# ── Logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
    local arn="${TASK_ARN}"
    if [[ -z "$arn" && -f "$SCRIPT_DIR/.last-task-arn" ]]; then
        arn=$(cat "$SCRIPT_DIR/.last-task-arn")
    fi
    [[ -z "$arn" ]] && { err "No task ARN. Use --task-arn <arn> or run submit first."; exit 1; }

    local task_id
    task_id=$(echo "$arn" | awk -F/ '{print $NF}')

    log "Streaming logs for task: $task_id"
    log "(Ctrl+C to stop)"

    # Find the log stream
    aws_cmd logs tail "/ecs/dm-isaac-g1" --follow --since 1h \
        --filter-pattern "$task_id" 2>/dev/null || {
        # Fallback: try with the motion name prefix
        if [[ -n "$MOTION_NAME" ]]; then
            aws_cmd logs tail "/ecs/dm-isaac-g1" --follow --since 1h 2>/dev/null
        fi
    }
}

# ── List ──────────────────────────────────────────────────────────────────────
cmd_list() {
    log "=== Recent Tasks ==="
    aws_cmd ecs list-tasks --cluster "$CLUSTER_NAME" --desired-status STOPPED \
        --query "taskArns[]" --output text 2>/dev/null | tr '\t' '\n' | head -10 | while read -r arn; do
        [[ -z "$arn" ]] && continue
        aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
            --query "tasks[0].{Arn:taskArn,Status:lastStatus,Def:taskDefinitionArn,Exit:containers[0].exitCode,Started:startedAt,Stopped:stoppedAt}" \
            --output table 2>/dev/null
    done
}

# ── Stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
    local arn="${TASK_ARN}"
    if [[ -z "$arn" && -f "$SCRIPT_DIR/.last-task-arn" ]]; then
        arn=$(cat "$SCRIPT_DIR/.last-task-arn")
    fi
    [[ -z "$arn" ]] && { err "No task ARN. Use --task-arn <arn>."; exit 1; }

    log "Stopping task: $arn"
    aws_cmd ecs stop-task --cluster "$CLUSTER_NAME" --task "$arn" --reason "Stopped by run.sh" >/dev/null
    log "Task stopped"
}

# ── Shell (launch interactive container) ──────────────────────────────────────
cmd_shell() {
    log "=== Launching Interactive GPU Container ==="

    local family="dm-interactive-shell"

    # Register a long-running task definition (sleep for 24h)
    local tmpfile
    tmpfile=$(mktemp /tmp/ecs-taskdef-XXXXXX.json)
    cat > "$tmpfile" << TASKEOF
{
    "family": "${family}",
    "taskRoleArn": "${TASK_ROLE_ARN}",
    "executionRoleArn": "${EXEC_ROLE_ARN}",
    "networkMode": "host",
    "requiresCompatibilities": ["EC2"],
    "containerDefinitions": [
        {
            "name": "workspace",
            "image": "${ECR_IMAGE}",
            "essential": true,
            "resourceRequirements": [
                {"type": "GPU", "value": "1"}
            ],
            "memory": 28000,
            "cpu": 6144,
            "command": ["bash", "-c", "echo '=== Interactive container ready ===' && nvidia-smi; ${VULKAN_SETUP_CMD}; ${VNC_STARTUP_CMD}; echo 'VNC+XFCE started on :5901'; echo '=== Pulling latest code ==='; if [ -d /workspace/dm-isaac-g1 ]; then cd /workspace/dm-isaac-g1 && git pull origin main 2>/dev/null; pip install -e . -q 2>/dev/null; fi; if [ -d /workspace/unitree_rl_lab ]; then cd /workspace/unitree_rl_lab && git pull origin main 2>/dev/null; cd source/unitree_rl_lab && pip install -e . -q 2>/dev/null; python -m dm_isaac_g1.rl.install_tasks 2>/dev/null; fi; echo '=== Container ready ==='; sleep 86400"],
            "environment": [
                {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"},
                {"name": "VK_ICD_FILENAMES", "value": "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"},
                {"name": "XDG_RUNTIME_DIR", "value": "/tmp/xdg"},
                {"name": "ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ACCEPT_EULA", "value": "Y"},
                {"name": "OMNI_KIT_ALLOW_ROOT", "value": "1"},
                {"name": "S3_BUCKET", "value": "${S3_BUCKET}"},
                {"name": "AWS_REGION", "value": "${AWS_REGION}"},
                {"name": "HF_TOKEN", "value": "${HF_TOKEN:-}"},
                {"name": "WANDB_API_KEY", "value": "${WANDB_API_KEY:-}"},
                {"name": "WANDB_PROJECT", "value": "${WANDB_PROJECT:-dm-isaac-g1}"},
                {"name": "GITHUB_TOKEN", "value": "${GITHUB_TOKEN:-}"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/dm-isaac-g1",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "interactive"
                }
            },
            "linuxParameters": {
                "initProcessEnabled": true,
                "sharedMemorySize": 8192,
                "devices": [
                    {"hostPath": "/dev/dri", "containerPath": "/dev/dri", "permissions": ["read", "write", "mknod"]}
                ]
            }
        }
    ]
}
TASKEOF

    local revision
    revision=$(aws_cmd ecs register-task-definition \
        --cli-input-json "file://${tmpfile}" \
        --query "taskDefinition.revision" --output text)
    rm -f "$tmpfile"
    log "Registered: ${family}:${revision}"

    # Launch the task
    local task_arn
    task_arn=$(aws_cmd ecs run-task \
        --cluster "$CLUSTER_NAME" \
        --task-definition "${family}:${revision}" \
        --capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1" \
        --count 1 \
        --enable-execute-command \
        --started-by "run.sh-shell" \
        --tags "key=Type,value=interactive" \
        --query "tasks[0].taskArn" --output text)

    echo "$task_arn" > "$SCRIPT_DIR/.last-task-arn"

    log ""
    log "=== Interactive Container Submitted ==="
    log "Task ARN: $task_arn"
    log ""
    log "ECS will auto-scale a GPU instance (~3-5 min)."
    log "Once running, connect with:"
    log ""
    log "  $0 exec --task-arn $task_arn"
    log ""
    log "This container stays alive for 24h. Stop it when done:"
    log "  $0 stop --task-arn $task_arn"
    log ""

    # Wait for task to reach RUNNING
    log "Waiting for task to start..."
    local status="PROVISIONING"
    while [[ "$status" != "RUNNING" && "$status" != "STOPPED" ]]; do
        sleep 15
        status=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$task_arn" \
            --query "tasks[0].lastStatus" --output text 2>/dev/null || echo "PENDING")
        log "  Status: $status"
    done

    if [[ "$status" == "RUNNING" ]]; then
        log ""
        log "Container is RUNNING! Connect now:"
        log "  $0 exec --task-arn $task_arn"
    else
        err "Task stopped unexpectedly. Check logs with: $0 logs --task-arn $task_arn"
    fi
}

# ── Exec (shell into running container) ──────────────────────────────────────
cmd_exec() {
    local arn="${TASK_ARN}"
    if [[ -z "$arn" && -f "$SCRIPT_DIR/.last-task-arn" ]]; then
        arn=$(cat "$SCRIPT_DIR/.last-task-arn")
    fi
    [[ -z "$arn" ]] && { err "No task ARN. Use --task-arn <arn> or launch a shell first."; exit 1; }

    log "Connecting to container..."
    log "Task: $arn"
    log ""

    # Detect the container name from the task
    local container
    container=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
        --query "tasks[0].containers[?lastStatus=='RUNNING'].name | [0]" --output text 2>/dev/null)

    if [[ -z "$container" || "$container" == "None" ]]; then
        container="workspace"  # fallback for interactive shells
        # Try training container for training tasks
        local all_containers
        all_containers=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
            --query "tasks[0].containers[*].name" --output text 2>/dev/null)
        if echo "$all_containers" | grep -q "training"; then
            container="training"
        fi
    fi

    log "Container: $container"
    aws_cmd ecs execute-command \
        --cluster "$CLUSTER_NAME" \
        --task "$arn" \
        --container "$container" \
        --interactive \
        --command "/bin/bash"
}

# ── SSH (directly into EC2 instance) ──────────────────────────────────────────
cmd_ssh() {
    log "Finding running EC2 instances in ASG..."

    local instances
    instances=$(aws_cmd autoscaling describe-auto-scaling-groups --auto-scaling-group-names "dm-isaac-g1-gpu-asg" \
        --query "AutoScalingGroups[0].Instances[?LifecycleState=='InService'].InstanceId" --output text 2>/dev/null)

    if [[ -z "$instances" || "$instances" == "None" ]]; then
        err "No running instances. Launch a shell or submit a task first:"
        err "  $0 shell"
        exit 1
    fi

    # Get public IPs
    echo ""
    log "=== Running GPU Instances ==="
    for instance_id in $instances; do
        local ip
        ip=$(aws_cmd ec2 describe-instances --instance-ids "$instance_id" \
            --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2>/dev/null)
        log "  Instance: $instance_id  IP: $ip"
        log "  SSH:  ssh -i ${SCRIPT_DIR}/dm-isaac-g1-training.pem ec2-user@${ip}"
        log "  VNC:  ${ip}:5901  (password: datament) — start with: $0 vnc"
        echo ""
    done

    # If only one instance, SSH directly
    local count
    count=$(echo "$instances" | wc -w | tr -d ' ')
    if [[ "$count" == "1" ]]; then
        local ip
        ip=$(aws_cmd ec2 describe-instances --instance-ids "$instances" \
            --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
        log "Connecting to $ip..."
        ssh -i "${SCRIPT_DIR}/dm-isaac-g1-training.pem" -o StrictHostKeyChecking=no "ec2-user@${ip}"
    else
        log "Multiple instances running. Use the SSH command above for the one you want."
    fi
}

# ── VNC (start TurboVNC inside container via ECS Exec) ───────────────────────
cmd_vnc() {
    local arn="${TASK_ARN}"
    if [[ -z "$arn" && -f "$SCRIPT_DIR/.last-task-arn" ]]; then
        arn=$(cat "$SCRIPT_DIR/.last-task-arn")
    fi
    [[ -z "$arn" ]] && { err "No task ARN. Launch a shell first: $0 shell"; exit 1; }

    # Get the instance IP for this task
    local container_instance
    container_instance=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
        --query "tasks[0].containerInstanceArn" --output text 2>/dev/null)
    local instance_id
    instance_id=$(aws_cmd ecs describe-container-instances --cluster "$CLUSTER_NAME" \
        --container-instances "$container_instance" \
        --query "containerInstances[0].ec2InstanceId" --output text 2>/dev/null)
    local ip
    ip=$(aws_cmd ec2 describe-instances --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].PublicIpAddress" --output text 2>/dev/null)

    log "Starting TurboVNC inside container (task: $arn)..."

    # Detect container name
    local container
    container=$(aws_cmd ecs describe-tasks --cluster "$CLUSTER_NAME" --tasks "$arn" \
        --query "tasks[0].containers[?lastStatus=='RUNNING'].name | [0]" --output text 2>/dev/null)
    [[ -z "$container" || "$container" == "None" ]] && container="workspace"

    # Start VNC + XFCE desktop inside the container via ECS Exec
    aws_cmd ecs execute-command \
        --cluster "$CLUSTER_NAME" \
        --task "$arn" \
        --container "$container" \
        --interactive \
        --command "bash -c '${VULKAN_SETUP_CMD}; ${VNC_STARTUP_CMD}; sleep 3; echo VNC+XFCE started on display :1'"

    log ""
    log "=== VNC Ready ==="
    log "Connect with any VNC client (RealVNC, TigerVNC Viewer, etc.):"
    log "  Address:  ${ip}:5901"
    log "  Password: datament"
    log ""
    log "TurboVNC, Chrome, and XFCE4 desktop are all inside the container."
}

# ── Download ──────────────────────────────────────────────────────────────────
cmd_download() {
    [[ -z "$MOTION_NAME" ]] && { err "Must specify --motion"; exit 1; }

    local local_dir="$REPO_ROOT/checkpoints/mimic/${MOTION_NAME}"
    mkdir -p "$local_dir"

    log "Downloading checkpoints from s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/..."
    aws_cmd s3 sync "s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/" "$local_dir/"
    log "Saved to: $local_dir"
    ls -la "$local_dir"
}

# ── Parse Arguments ───────────────────────────────────────────────────────────
COMMAND="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)           TASK_TYPE="$2"; shift 2 ;;
        --motion)         MOTION_NAME="$2"; shift 2 ;;
        --task-id)        TASK_ID="$2"; shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --task-arn)       TASK_ARN="$2"; shift 2 ;;
        --checkpoint)     CHECKPOINT_FILE="$2"; shift 2 ;;
        --video-length)   VIDEO_LENGTH="$2"; shift 2 ;;
        --hf-repo)        HF_REPO="$2"; shift 2 ;;
        --gui)            HEADLESS="false"; shift ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

case "$COMMAND" in
    submit)   cmd_submit ;;
    replay)   cmd_replay ;;
    sim2sim)  cmd_sim2sim ;;
    shell)    cmd_shell ;;
    exec)     cmd_exec ;;
    ssh)      cmd_ssh ;;
    vnc)      cmd_vnc ;;
    status)   cmd_status ;;
    logs)     cmd_logs ;;
    list)     cmd_list ;;
    stop)     cmd_stop ;;
    download) cmd_download ;;
    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands (training):"
        echo "  submit    Upload data and run training on ECS"
        echo "  replay    Export policy + record replay video on ECS"
        echo "  sim2sim   Validate exported policy in MuJoCo (Isaac Lab -> MuJoCo)"
        echo "  status    Show cluster and task status"
        echo "  logs      Stream training logs (CloudWatch)"
        echo "  list      List recent tasks"
        echo "  stop      Stop a running task"
        echo "  download  Download checkpoints from S3"
        echo ""
        echo "Commands (interactive access):"
        echo "  shell     Launch interactive GPU container (24h lifetime)"
        echo "  exec      Get bash shell into a running container (ECS Exec)"
        echo "  ssh       SSH directly into the EC2 GPU instance"
        echo "  vnc       Start VNC/NoVNC server for GUI access (Isaac Sim, MuJoCo)"
        echo ""
        echo "Options:"
        echo "  --task <type>           Task type: mimic, rl (required for submit)"
        echo "  --motion <name>         Motion name, e.g. cr7_06_tiktok_uefa (required for mimic)"
        echo "  --task-id <id>          Gymnasium task ID (required for rl, e.g. DM-G1-29dof-FALCON)"
        echo "  --max-iterations <n>    Max training iterations (default: 30000 mimic, 50000 rl)"
        echo "  --task-arn <arn>        Task ARN for logs/stop/exec"
        echo "  --checkpoint <file>     Specific checkpoint file for replay (default: latest)"
        echo "  --video-length <n>      Video length in steps for replay (default: 300)"
        echo "  --hf-repo <repo>        HuggingFace repo for replay upload (optional)"
        echo "  --gui                   Run sim2sim in GUI mode (MuJoCo viewer via VNC)"
        echo ""
        echo "Examples:"
        echo "  $0 submit --task mimic --motion cr7_06_tiktok_uefa"
        echo "  $0 submit --task rl --task-id DM-G1-29dof-FALCON"
        echo "  $0 submit --task rl --task-id DM-G1-29dof-SoFTA --max-iterations 30000"
        echo "  $0 submit --task rl --task-id DM-G1-29dof-TWIST"
        echo "  $0 replay --task mimic --motion cr7_06_tiktok_uefa"
        echo "  $0 replay --task rl --task-id DM-G1-29dof-FALCON"
        echo "  $0 replay --task mimic --motion video_007 --hf-repo datamentorshf/dm-g1-video007-mimic"
        echo "  $0 sim2sim --task rl --task-id DM-G1-29dof-FALCON"
        echo "  $0 sim2sim --task mimic --motion cr7_06_tiktok_uefa"
        echo "  $0 sim2sim --task rl --task-id DM-G1-29dof-FALCON --gui  # VNC interactive"
        echo "  $0 shell                                   # interactive GPU container"
        echo "  $0 exec --task-arn <arn>                   # bash into container"
        echo "  $0 ssh                                     # SSH into EC2 instance"
        echo "  $0 vnc                                     # start GUI access"
        echo "  $0 status"
        echo "  $0 logs"
        echo "  $0 download --motion cr7_06_tiktok_uefa"
        echo "  $0 stop"
        ;;
esac
