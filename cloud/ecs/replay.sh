#!/usr/bin/env bash
# =============================================================================
# Replay Script (runs inside the ECS container)
# =============================================================================
# Downloads a trained checkpoint from S3, runs play.py to:
#   1. Export ONNX + JIT policy
#   2. Record a replay video (headless)
#   3. Upload exports + video back to S3 and optionally to HuggingFace
#
# Supports both mimic and RL tasks.
#
# Expected environment variables:
#   TASK_TYPE         - "mimic" or "rl"
#   TASK_ID           - Gymnasium task ID (e.g., DM-G1-29dof-Mimic-CR7TiktokUEFA)
#   MOTION_NAME       - Motion/task name for S3 paths (e.g., cr7_06_tiktok_uefa)
#   S3_BUCKET         - S3 bucket
#   AWS_REGION        - AWS region
#   CHECKPOINT_FILE   - Specific checkpoint file (e.g., model_14500.pt), optional
#   VIDEO_LENGTH      - Video length in steps (default: 300)
#   GITHUB_TOKEN      - GitHub token for private repo access
#   HF_TOKEN          - HuggingFace token (optional, for HF upload)
#   HF_REPO           - HuggingFace repo (optional, e.g., datamentorshf/dm-g1-cr7-tiktok-mimic)
# =============================================================================
set -euo pipefail

echo "=== DM Isaac G1 — Replay & Export ==="
echo "Started at: $(date -u)"
echo "Task type: ${TASK_TYPE}"
echo "Task ID: ${TASK_ID}"
echo "Motion: ${MOTION_NAME}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TASK_TYPE="${TASK_TYPE:?TASK_TYPE is required (mimic or rl)}"
TASK_ID="${TASK_ID:?TASK_ID is required}"
MOTION_NAME="${MOTION_NAME:?MOTION_NAME is required}"
S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
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

# ── Setup Vulkan ─────────────────────────────────────────────────────────────
# Isaac Sim's PhysX discovers GPUs through Vulkan (not CUDA directly).
# nvidia-container-toolkit mounts libGLX_nvidia.so.0 into the container.
# We ensure the Vulkan ICD manifest points to this lib.
export XDG_RUNTIME_DIR=/tmp/xdg
mkdir -p "$XDG_RUNTIME_DIR"

# Ensure NVIDIA Vulkan ICD manifest exists (may be missing in older images)
mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/icd.d
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json
cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

# Diagnostic: show Vulkan state
echo "=== Vulkan Diagnostics ==="
echo "Vulkan ICD files:"
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  (none)"
cat /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
echo "NVIDIA libs in container:"
ldconfig -p 2>/dev/null | grep -i "vulkan\|nvidia" | head -10 || true
vulkaninfo --summary 2>&1 | head -20 || true

# ── Isaac Sim EULA acceptance (prevents zenity dialog from blocking headless) ─
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export OMNI_KIT_ALLOW_ROOT=1

# Unset DISPLAY to prevent zenity GUI dialogs from blocking headless operations.
# When VNC is started before this script, DISPLAY=:1 causes Isaac Sim's GPU error
# dialog (zenity) to display and wait for user interaction, hanging the process.
unset DISPLAY

# ── Configure Git Auth ────────────────────────────────────────────────────────
REPO_URL="https://github.com/datamentors/dm-isaac-g1.git"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/datamentors/dm-isaac-g1.git"
    git config --global credential.helper store
    echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
fi

# ── Pull Latest Code ──────────────────────────────────────────────────────────
echo "=== Updating dm-isaac-g1 ==="
cd "$WORKSPACE"

if [[ -d "dm-isaac-g1/.git" ]]; then
    cd dm-isaac-g1
    git fetch origin main 2>/dev/null || true
    git reset --hard origin/main 2>/dev/null || git pull --ff-only 2>/dev/null || true
    echo "dm-isaac-g1 at commit: $(git rev-parse --short HEAD)"
    cd "$WORKSPACE"
elif [[ -d "dm-isaac-g1" ]]; then
    rm -rf dm-isaac-g1
    git clone "$REPO_URL" dm-isaac-g1
else
    git clone "$REPO_URL" dm-isaac-g1 || { echo "ERROR: Could not clone repo."; exit 1; }
fi

# ── Activate conda env ────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate unitree_sim_env 2>/dev/null || true
fi

# Install dm-isaac-g1
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
    pip install -e "$WORKSPACE/unitree_rl_lab/source/unitree_rl_lab" --quiet || true
fi

# ── Download robot model assets from S3 ──────────────────────────────────────
if [[ ! -d "$WORKSPACE/unitree_model/G1" ]]; then
    echo "Downloading Unitree G1 model assets..."
    aws s3 cp "s3://${S3_BUCKET}/assets/unitree_model_g1.tar.gz" /tmp/unitree_model_g1.tar.gz --region "$AWS_REGION"
    tar xzf /tmp/unitree_model_g1.tar.gz -C "$WORKSPACE/"
    rm -f /tmp/unitree_model_g1.tar.gz
fi

# ── Download Checkpoint from S3 ──────────────────────────────────────────────
echo "=== Downloading checkpoint from S3 ==="

if [[ "$TASK_TYPE" == "rl" ]]; then
    TASK_CLEAN=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
    S3_MODEL_PREFIX="Models/RL/${TASK_CLEAN}"
else
    S3_MODEL_PREFIX="Models/IL/Mimic-${MOTION_NAME}"
fi

# Determine which checkpoint to use
if [[ -z "${CHECKPOINT_FILE:-}" ]]; then
    # Find the latest checkpoint on S3
    CHECKPOINT_FILE=$(aws s3 ls "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/" --region "$AWS_REGION" \
        | grep "model_.*\.pt" | awk '{print $4}' \
        | python3 -c "
import sys
files = [l.strip() for l in sys.stdin if l.strip()]
files.sort(key=lambda f: int(f.replace('model_','').replace('.pt','')))
print(files[-1] if files else '')
" 2>/dev/null)
    [[ -z "$CHECKPOINT_FILE" ]] && { echo "ERROR: No checkpoint found in s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/"; exit 1; }
fi

echo "Using checkpoint: $CHECKPOINT_FILE"

# Create a log directory structure matching what play.py expects
CKPT_DIR="$WORKSPACE/dm-isaac-g1/logs/rsl_rl/${TASK_ID}/seed_42"
mkdir -p "$CKPT_DIR"

aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/${CHECKPOINT_FILE}" "$CKPT_DIR/${CHECKPOINT_FILE}" --region "$AWS_REGION"
echo "Checkpoint downloaded to: $CKPT_DIR/${CHECKPOINT_FILE}"

# ── Download training data for mimic tasks ───────────────────────────────────
if [[ "$TASK_TYPE" == "mimic" ]]; then
    TASK_DATA_DIR="$WORKSPACE/dm-isaac-g1/src/dm_isaac_g1/mimic/tasks/${MOTION_NAME}"
    mkdir -p "$TASK_DATA_DIR"
    aws s3 sync "s3://${S3_BUCKET}/tasks/mimic/${MOTION_NAME}/" "$TASK_DATA_DIR/" --region "$AWS_REGION"
    echo "Task data synced: $TASK_DATA_DIR"
    ls -la "$TASK_DATA_DIR/"
fi

# ── Run Replay (Export + Video) ───────────────────────────────────────────────
echo "=== Running Replay ==="
echo "Task: $TASK_ID"
echo "Checkpoint: $CKPT_DIR/$CHECKPOINT_FILE"
echo "Video length: $VIDEO_LENGTH steps"

cd "$WORKSPACE/dm-isaac-g1"

# Determine which play.py script to use
if [[ "$TASK_TYPE" == "rl" ]]; then
    PLAY_SCRIPT="src/dm_isaac_g1/rl/scripts/play.py"
else
    PLAY_SCRIPT="src/dm_isaac_g1/mimic/scripts/play.py"
fi

echo "Using play script: $PLAY_SCRIPT"

# ── Isaac Sim replay: export ONNX/JIT + record video ─────────────────────────
# This is the primary path. Isaac Sim handles both export and video recording.
# If it fails (e.g., PhysX can't find GPU), we fall back to standalone PyTorch export.
echo "=== Running Isaac Sim replay (export + video) ==="
PLAY_EXIT=0
python -u "$PLAY_SCRIPT" \
    --task "$TASK_ID" \
    --checkpoint "$CKPT_DIR/$CHECKPOINT_FILE" \
    --video --video_length "$VIDEO_LENGTH" \
    --headless 2>&1 || PLAY_EXIT=$?

echo "Isaac Sim exit code: $PLAY_EXIT"

# If Isaac Sim failed, fall back to standalone PyTorch export (no video, but exports work)
if [[ $PLAY_EXIT -ne 0 ]]; then
    echo "=== Isaac Sim failed (exit $PLAY_EXIT) — falling back to standalone export ==="
    python -u "$WORKSPACE/dm-isaac-g1/src/dm_isaac_g1/mimic/scripts/export_policy.py" \
        --task "$TASK_ID" \
        --checkpoint "$CKPT_DIR/$CHECKPOINT_FILE" 2>&1 || {
        echo "WARNING: Standalone export also failed"
    }
    # Reset exit code — standalone export is sufficient for deployment
    PLAY_EXIT=0
fi

# ── Upload Results to S3 ─────────────────────────────────────────────────────
echo "=== Uploading results to S3 ==="

# Upload exported policy (ONNX, JIT, deploy.yaml)
if [[ -d "$CKPT_DIR/exported" ]]; then
    aws s3 sync "$CKPT_DIR/exported/" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/exported/" \
        --region "$AWS_REGION"
    echo "Exported policy uploaded to s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/exported/"
fi

# Upload params (deploy.yaml)
if [[ -d "$CKPT_DIR/params" ]]; then
    aws s3 sync "$CKPT_DIR/params/" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/params/" \
        --region "$AWS_REGION"
    echo "Deploy params uploaded to s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/params/"
fi

# Upload videos
if [[ -d "$CKPT_DIR/videos" ]]; then
    aws s3 sync "$CKPT_DIR/videos/" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/videos/" \
        --region "$AWS_REGION"
    echo "Videos uploaded to s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/videos/"
fi

# ── Upload to HuggingFace (optional) ─────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" && -n "${HF_REPO:-}" ]]; then
    echo "=== Uploading to HuggingFace ==="
    pip install --quiet huggingface_hub 2>/dev/null || true

    python3 -u << HFEOF
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO"]
ckpt_dir = "$CKPT_DIR"

# Upload exported policy
for subdir in ["exported", "params"]:
    path = os.path.join(ckpt_dir, subdir)
    if os.path.isdir(path):
        api.upload_folder(folder_path=path, repo_id=repo_id, path_in_repo=subdir)
        print(f"Uploaded {subdir}/ to {repo_id}")

# Upload videos
video_dir = os.path.join(ckpt_dir, "videos")
if os.path.isdir(video_dir):
    api.upload_folder(folder_path=video_dir, repo_id=repo_id, path_in_repo="videos")
    print(f"Uploaded videos/ to {repo_id}")

# Upload the checkpoint itself
ckpt_file = os.path.join(ckpt_dir, "$CHECKPOINT_FILE")
if os.path.isfile(ckpt_file):
    api.upload_file(path_or_fileobj=ckpt_file, repo_id=repo_id, path_in_repo="checkpoints/$CHECKPOINT_FILE")
    print(f"Uploaded checkpoint to {repo_id}")

print(f"HuggingFace upload complete: https://huggingface.co/{repo_id}")
HFEOF
fi

echo "=== Replay & Export Complete ==="
echo "Exit code: $PLAY_EXIT"
echo "Finished at: $(date -u)"
exit ${PLAY_EXIT:-0}
