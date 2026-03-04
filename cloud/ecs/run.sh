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
MAX_ITERATIONS=30000
TASK_ARN=""

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

aws_cmd() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

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

    # Upload the training script
    aws_cmd s3 cp "$SCRIPT_DIR/train_mimic.sh" "s3://${S3_BUCKET}/scripts/train_mimic.sh"
    log "Uploaded training script"
}

# ── Register Task Definition ─────────────────────────────────────────────────
register_task_def() {
    local task="$1"
    local motion="$2"
    local max_iter="$3"
    local task_id="$4"
    local family="dm-${task}-${motion}"

    log "Registering task definition: $family (task_id=$task_id)" >&2

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
                {"name": "MOTION_NAME", "value": "${motion}"},
                {"name": "TASK_ID", "value": "${task_id}"},
                {"name": "MAX_ITERATIONS", "value": "${max_iter}"},
                {"name": "S3_BUCKET", "value": "${S3_BUCKET}"},
                {"name": "AWS_REGION", "value": "${AWS_REGION}"},
                {"name": "HF_TOKEN", "value": "${HF_TOKEN:-}"},
                {"name": "WANDB_API_KEY", "value": "${WANDB_API_KEY:-}"},
                {"name": "WANDB_PROJECT", "value": "${WANDB_PROJECT:-dm-isaac-g1}"},
                {"name": "GITHUB_TOKEN", "value": "${GITHUB_TOKEN:-}"}
            ],
            "command": ["bash", "/opt/training/scripts/train_mimic.sh"],
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
                "sharedMemorySize": 8192
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
    [[ -z "$TASK_TYPE" ]] && { err "Must specify --task (e.g., mimic)"; exit 1; }
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

    log "=== Submitting ECS Training Job ==="
    log "Task: $TASK_TYPE, Motion: $MOTION_NAME, Task ID: $TASK_ID, Iterations: $MAX_ITERATIONS"

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
        --tags "key=Motion,value=$MOTION_NAME" "key=Task,value=$TASK_TYPE" \
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
    log "When training completes, checkpoints will be in:"
    log "  s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/"
    log "Download with:"
    log "  $0 download --motion $MOTION_NAME"

    # Save for convenience
    echo "$task_arn" > "$SCRIPT_DIR/.last-task-arn"
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
            "command": ["bash", "-c", "echo '=== Interactive container ready ===' && nvidia-smi && /opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 2>/dev/null && echo 'VNC started on :5901' && sleep 86400"],
            "environment": [
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
                "sharedMemorySize": 8192
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

    # Start VNC server inside the container via ECS Exec
    # TurboVNC is at /opt/TurboVNC/bin/vncserver, password pre-set to "datament"
    aws_cmd ecs execute-command \
        --cluster "$CLUSTER_NAME" \
        --task "$arn" \
        --container "$container" \
        --interactive \
        --command "bash -c '/opt/TurboVNC/bin/vncserver :1 -geometry 1920x1080 -depth 24 2>/dev/null; echo VNC started on display :1'"

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
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

case "$COMMAND" in
    submit)   cmd_submit ;;
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
        echo "  --task <type>           Task type: mimic (required for submit)"
        echo "  --motion <name>         Motion name, e.g. cr7_06_tiktok_uefa (required)"
        echo "  --max-iterations <n>    Max training iterations (default: 30000)"
        echo "  --task-arn <arn>        Task ARN for logs/stop/exec"
        echo ""
        echo "Examples:"
        echo "  $0 submit --task mimic --motion cr7_06_tiktok_uefa"
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
