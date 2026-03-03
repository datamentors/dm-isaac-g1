#!/usr/bin/env bash
# =============================================================================
# Mimic Training Script (runs inside the ECS container)
# =============================================================================
# This script runs inside the ECR container on an ECS GPU instance.
# It pulls the latest dm-isaac-g1 code, copies training data, and runs
# the mimic policy training.
#
# Expected environment variables:
#   MOTION_NAME       - e.g., cr7_06_tiktok_uefa
#   TASK_ID           - Gymnasium task ID (auto-derived if not set)
#   MAX_ITERATIONS    - Training iterations (default: 30000)
#   S3_BUCKET         - S3 bucket for data and checkpoints
#   AWS_REGION        - AWS region
#   HF_TOKEN          - HuggingFace token
#   WANDB_API_KEY     - WandB API key
#   WANDB_PROJECT     - WandB project name
# =============================================================================
set -euo pipefail

echo "=== DM Isaac G1 — Mimic Training ==="
echo "Started at: $(date -u)"
echo "Motion: ${MOTION_NAME}"
echo "Max iterations: ${MAX_ITERATIONS:-30000}"

# ── Defaults ──────────────────────────────────────────────────────────────────
MOTION_NAME="${MOTION_NAME:?MOTION_NAME is required}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dm-isaac-g1}"
WORKSPACE="/workspace"

# Derive task ID from motion name if not provided
# Convention: cr7_06_tiktok_uefa -> DM-G1-29dof-Mimic-CR7-06-TikTokUEFA
if [[ -z "${TASK_ID:-}" ]]; then
    TASK_ID="DM-G1-29dof-Mimic-${MOTION_NAME}"
fi

# ── GPU Check ─────────────────────────────────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# ── Pull Latest Code ──────────────────────────────────────────────────────────
echo "=== Updating dm-isaac-g1 ==="
cd "$WORKSPACE"

if [[ -d "dm-isaac-g1/.git" ]]; then
    cd dm-isaac-g1
    git pull --ff-only origin main 2>/dev/null || git pull --ff-only 2>/dev/null || {
        echo "Git pull failed, using existing code"
    }
    cd "$WORKSPACE"
elif [[ -d "dm-isaac-g1" ]]; then
    echo "dm-isaac-g1 exists but not a git repo, using as-is"
else
    echo "Cloning dm-isaac-g1..."
    git clone https://github.com/datamentors/dm-isaac-g1.git 2>/dev/null || {
        echo "WARNING: Could not clone repo. Will use data from S3."
    }
fi

# Install dm-isaac-g1 as editable package if setup.py/pyproject.toml exists
if [[ -f "$WORKSPACE/dm-isaac-g1/pyproject.toml" ]]; then
    pip install -e "$WORKSPACE/dm-isaac-g1" --quiet 2>/dev/null || true
fi

# ── Download Training Data from S3 ───────────────────────────────────────────
echo "=== Downloading training data from S3 ==="
TASK_DIR="$WORKSPACE/dm-isaac-g1/src/dm_isaac_g1/mimic/tasks/${MOTION_NAME}"
mkdir -p "$TASK_DIR"

aws s3 sync "s3://${S3_BUCKET}/tasks/mimic/${MOTION_NAME}/" "$TASK_DIR/" --region "$AWS_REGION"

echo "Task data:"
ls -la "$TASK_DIR/"

# Verify NPZ exists
if [[ ! -f "$TASK_DIR/${MOTION_NAME}.npz" ]]; then
    echo "ERROR: NPZ file not found: $TASK_DIR/${MOTION_NAME}.npz"
    exit 1
fi

# ── Run Training ──────────────────────────────────────────────────────────────
echo "=== Starting Training ==="
echo "Task: $TASK_ID"
echo "Iterations: $MAX_ITERATIONS"
echo "Time: $(date -u)"

cd "$WORKSPACE/dm-isaac-g1"

# Activate the conda environment if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate unitree_sim_env 2>/dev/null || true
fi

python -u src/dm_isaac_g1/mimic/scripts/train.py \
    --task "$TASK_ID" \
    --num_envs 4096 \
    --max_iterations "$MAX_ITERATIONS" \
    --headless \
    --logger wandb 2>&1

TRAIN_EXIT=$?

# ── Upload Results ────────────────────────────────────────────────────────────
echo "=== Uploading checkpoints to S3 ==="

# Find the latest checkpoint directory
CHECKPOINT_DIR=$(find "$WORKSPACE/dm-isaac-g1/logs" -path "*${MOTION_NAME}*" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
if [[ -z "$CHECKPOINT_DIR" ]]; then
    # Fallback: search by task ID pattern
    CHECKPOINT_DIR=$(find "$WORKSPACE/dm-isaac-g1/logs" -name "model_*.pt" -printf '%h\n' 2>/dev/null | sort -u | tail -1)
fi

if [[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]]; then
    echo "Uploading from: $CHECKPOINT_DIR"
    aws s3 sync "$CHECKPOINT_DIR/" "s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/" \
        --region "$AWS_REGION" \
        --exclude "optimizer_*"
    echo "Checkpoints uploaded to s3://${S3_BUCKET}/checkpoints/mimic/${MOTION_NAME}/"
else
    echo "WARNING: No checkpoint directory found"
fi

# Upload any wandb logs
if [[ -d "$WORKSPACE/dm-isaac-g1/wandb" ]]; then
    aws s3 sync "$WORKSPACE/dm-isaac-g1/wandb/" "s3://${S3_BUCKET}/logs/wandb/${MOTION_NAME}/" \
        --region "$AWS_REGION" --quiet
fi

echo "=== Training Complete ==="
echo "Exit code: $TRAIN_EXIT"
echo "Finished at: $(date -u)"
exit $TRAIN_EXIT
