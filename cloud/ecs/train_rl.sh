#!/usr/bin/env bash
# =============================================================================
# RL Training Script (runs inside the ECS container)
# =============================================================================
# This script runs inside the ECR container on an ECS GPU instance.
# It pulls the latest dm-isaac-g1 code and runs RL locomotion training.
#
# Expected environment variables:
#   TASK_ID           - Gymnasium task ID (e.g., DM-G1-29dof-FALCON)
#   MAX_ITERATIONS    - Training iterations (default: 50000)
#   S3_BUCKET         - S3 bucket for checkpoints
#   AWS_REGION        - AWS region
#   WANDB_API_KEY     - WandB API key
#   WANDB_PROJECT     - WandB project name (default: dm-isaac-g1-rl)
#   GITHUB_TOKEN      - GitHub token for private repo access
# =============================================================================
set -euo pipefail

echo "=== DM Isaac G1 — RL Training ==="
echo "Started at: $(date -u)"
echo "Task: ${TASK_ID}"
echo "Max iterations: ${MAX_ITERATIONS:-50000}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK_ID="${TASK_ID:?TASK_ID is required}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dm-isaac-g1-rl}"
WORKSPACE="/workspace"
mkdir -p "$WORKSPACE"

# ── Install AWS CLI if missing ────────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
    echo "Installing AWS CLI..."
    pip install --quiet awscli 2>/dev/null || {
        curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
        unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install
        rm -rf /tmp/awscliv2.zip /tmp/aws
    }
fi

# ── GPU Check ─────────────────────────────────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# ── Configure Git Auth ────────────────────────────────────────────────────────
REPO_URL="https://github.com/datamentors/dm-isaac-g1.git"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/datamentors/dm-isaac-g1.git"
    git config --global credential.helper store
    echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
    echo "Git auth configured via GITHUB_TOKEN"
else
    echo "WARNING: GITHUB_TOKEN not set — git operations may fail for private repos"
fi

# ── Pull Latest Code ──────────────────────────────────────────────────────────
echo "=== Updating dm-isaac-g1 ==="
cd "$WORKSPACE"

if [[ -d "dm-isaac-g1/.git" ]]; then
    cd dm-isaac-g1
    git fetch origin main 2>/dev/null || true
    git reset --hard origin/main 2>/dev/null || git pull --ff-only 2>/dev/null || {
        echo "Git pull failed, using existing code"
    }
    echo "dm-isaac-g1 at commit: $(git rev-parse --short HEAD)"
    cd "$WORKSPACE"
elif [[ -d "dm-isaac-g1" ]]; then
    echo "dm-isaac-g1 exists but not a git repo — removing and cloning fresh"
    rm -rf dm-isaac-g1
    git clone "$REPO_URL" dm-isaac-g1
    echo "dm-isaac-g1 at commit: $(cd dm-isaac-g1 && git rev-parse --short HEAD)"
else
    echo "Cloning dm-isaac-g1..."
    git clone "$REPO_URL" dm-isaac-g1 || {
        echo "ERROR: Could not clone repo."
        exit 1
    }
    echo "dm-isaac-g1 at commit: $(cd dm-isaac-g1 && git rev-parse --short HEAD)"
fi

# ── Activate conda env ────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate unitree_sim_env 2>/dev/null || true
    echo "Conda env: $(conda info --envs | grep '*' || echo 'base')"
fi

# Install dm-isaac-g1 as editable package
if [[ -f "$WORKSPACE/dm-isaac-g1/pyproject.toml" ]]; then
    pip install -e "$WORKSPACE/dm-isaac-g1" --quiet 2>/dev/null || true
fi

# ── Install unitree_rl_lab ────────────────────────────────────────────────────
if ! python -c "import unitree_rl_lab" 2>/dev/null; then
    echo "Installing unitree_rl_lab..."
    if [[ ! -d "$WORKSPACE/unitree_rl_lab" ]]; then
        git clone https://github.com/unitreerobotics/unitree_rl_lab.git "$WORKSPACE/unitree_rl_lab"
    fi
    sed -i 's|UNITREE_MODEL_DIR = "path/to/unitree_model"|UNITREE_MODEL_DIR = "/workspace/unitree_model"|' \
        "$WORKSPACE/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
    pip install -e "$WORKSPACE/unitree_rl_lab/source/unitree_rl_lab" --quiet || {
        echo "ERROR: Failed to install unitree_rl_lab"
        pip install -e "$WORKSPACE/unitree_rl_lab/source/unitree_rl_lab" 2>&1 | tail -20
    }
fi

# ── Download robot model assets from S3 ──────────────────────────────────────
if [[ ! -d "$WORKSPACE/unitree_model/G1" ]]; then
    echo "Downloading Unitree G1 model assets..."
    mkdir -p "$WORKSPACE"
    aws s3 cp "s3://${S3_BUCKET}/assets/unitree_model_g1.tar.gz" /tmp/unitree_model_g1.tar.gz --region "$AWS_REGION"
    tar xzf /tmp/unitree_model_g1.tar.gz -C "$WORKSPACE/"
    rm -f /tmp/unitree_model_g1.tar.gz
    echo "Model assets extracted to $WORKSPACE/unitree_model/"
fi

# ── Periodic Checkpoint Upload ─────────────────────────────────────────────────
# Sync checkpoints to S3 every 10 minutes so we never lose progress if task is stopped
TASK_CLEAN=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
SYNC_INTERVAL=600  # 10 minutes

checkpoint_sync_loop() {
    while true; do
        sleep "$SYNC_INTERVAL"
        local ckpt_dir
        ckpt_dir=$(find "$WORKSPACE/dm-isaac-g1/logs" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
        if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
            aws s3 sync "$ckpt_dir/" "s3://${S3_BUCKET}/Models/RL/${TASK_CLEAN}/" \
                --region "$AWS_REGION" \
                --exclude "optimizer_*" --quiet 2>/dev/null
            echo "[checkpoint-sync] $(date -u) — synced to s3://${S3_BUCKET}/Models/RL/${TASK_CLEAN}/"
        fi
    done
}

# Start background sync
checkpoint_sync_loop &
SYNC_PID=$!
echo "Started periodic checkpoint sync (PID=$SYNC_PID, interval=${SYNC_INTERVAL}s)"

# ── Run Training ──────────────────────────────────────────────────────────────
echo "=== Starting RL Training ==="
echo "Task: $TASK_ID"
echo "Iterations: $MAX_ITERATIONS"
echo "WandB Project: $WANDB_PROJECT"
echo "Time: $(date -u)"

cd "$WORKSPACE/dm-isaac-g1"

python -u src/dm_isaac_g1/rl/scripts/train.py \
    --task "$TASK_ID" \
    --num_envs 4096 \
    --max_iterations "$MAX_ITERATIONS" \
    --headless \
    --logger wandb \
    --log_project_name "$WANDB_PROJECT" 2>&1

TRAIN_EXIT=$?

# Stop background sync
kill "$SYNC_PID" 2>/dev/null || true

# ── Final Upload ──────────────────────────────────────────────────────────────
echo "=== Final checkpoint upload to S3 ==="

CHECKPOINT_DIR=$(find "$WORKSPACE/dm-isaac-g1/logs" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)

if [[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]]; then
    echo "Uploading from: $CHECKPOINT_DIR"
    aws s3 sync "$CHECKPOINT_DIR/" "s3://${S3_BUCKET}/Models/RL/${TASK_CLEAN}/" \
        --region "$AWS_REGION" \
        --exclude "optimizer_*"
    echo "Checkpoints uploaded to s3://${S3_BUCKET}/Models/RL/${TASK_CLEAN}/"
else
    echo "WARNING: No checkpoint directory found"
fi

# Upload wandb logs
if [[ -d "$WORKSPACE/dm-isaac-g1/wandb" ]]; then
    aws s3 sync "$WORKSPACE/dm-isaac-g1/wandb/" "s3://${S3_BUCKET}/logs/wandb/${TASK_CLEAN}/" \
        --region "$AWS_REGION" --quiet
fi

echo "=== RL Training Complete ==="
echo "Exit code: $TRAIN_EXIT"
echo "Finished at: $(date -u)"
exit $TRAIN_EXIT
