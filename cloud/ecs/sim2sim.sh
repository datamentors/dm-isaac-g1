#!/usr/bin/env bash
# =============================================================================
# Sim2Sim Script (runs inside the ECS container)
# =============================================================================
# Validates a trained Isaac Lab RL/Mimic policy by deploying it in MuJoCo.
#
# Pipeline:
#   1. Downloads policy.onnx + deploy.yaml from S3 (previously exported by play.py)
#   2. Clones unitree_mujoco for G1 scene XML
#   3. Runs sim2sim.py (from src/) in MuJoCo — headless with video or GUI via VNC
#   4. Uploads videos to S3
#
# Modes:
#   - HEADLESS=true (default): EGL rendering, records video, auto-exits
#   - HEADLESS=false: Opens MuJoCo GUI viewer on VNC display :1 for interactive use
#
# Expected environment variables:
#   TASK_TYPE         - "mimic" or "rl"
#   TASK_ID           - Gymnasium task ID (e.g., DM-G1-29dof-FALCON)
#   MOTION_NAME       - Motion/task name for S3 paths
#   S3_BUCKET         - S3 bucket
#   AWS_REGION        - AWS region
#   VIDEO_LENGTH      - Video length in seconds (default: 10)
#   HEADLESS          - "true" (default) or "false" for VNC GUI mode
#   GITHUB_TOKEN      - GitHub token for private repo access
# =============================================================================
set -euo pipefail

echo "=== DM Isaac G1 — Sim2Sim (Isaac Lab → MuJoCo) ==="
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
VIDEO_LENGTH="${VIDEO_LENGTH:-10}"
HEADLESS="${HEADLESS:-true}"
WORKSPACE="/workspace"
mkdir -p "$WORKSPACE"

echo "Mode: $(if [[ "$HEADLESS" == "true" ]]; then echo 'headless (video recording)'; else echo 'GUI (VNC display :1)'; fi)"

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
nvidia-smi || echo "WARNING: No GPU available (MuJoCo can run on CPU)"

# ── Configure Git Auth ────────────────────────────────────────────────────────
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git config --global credential.helper store
    echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
fi

# ── Pull Latest Code ──────────────────────────────────────────────────────────
echo "=== Setting up repositories ==="
cd "$WORKSPACE"

# Clone dm-isaac-g1
REPO_URL="https://github.com/datamentors/dm-isaac-g1.git"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/datamentors/dm-isaac-g1.git"
fi

if [[ -d "dm-isaac-g1/.git" ]]; then
    cd dm-isaac-g1 && git fetch origin main 2>/dev/null && git reset --hard origin/main 2>/dev/null || true
    cd "$WORKSPACE"
elif [[ ! -d "dm-isaac-g1" ]]; then
    git clone "$REPO_URL" dm-isaac-g1
fi

# Clone unitree_mujoco (G1 MuJoCo scene XML)
if [[ ! -d "$WORKSPACE/unitree_mujoco" ]]; then
    echo "Cloning unitree_mujoco..."
    git clone https://github.com/unitreerobotics/unitree_mujoco.git "$WORKSPACE/unitree_mujoco"
fi

# ── Install Dependencies ─────────────────────────────────────────────────────
echo "=== Installing dependencies ==="

# Activate conda env if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate unitree_sim_env 2>/dev/null || true
fi

pip install --quiet mujoco onnxruntime numpy opencv-python pyyaml 2>/dev/null || true

# Install dm-isaac-g1 (for the sim2sim.py script)
if [[ -f "$WORKSPACE/dm-isaac-g1/pyproject.toml" ]]; then
    pip install -e "$WORKSPACE/dm-isaac-g1" --quiet 2>/dev/null || true
fi

# ── Download Exported Policy from S3 ─────────────────────────────────────────
echo "=== Downloading exported policy from S3 ==="

if [[ "$TASK_TYPE" == "rl" ]]; then
    TASK_CLEAN=$(echo "$TASK_ID" | sed 's/DM-G1-29dof-//')
    S3_MODEL_PREFIX="Models/RL/${TASK_CLEAN}"
else
    S3_MODEL_PREFIX="Models/IL/Mimic-${MOTION_NAME}"
fi

DEPLOY_DIR="$WORKSPACE/sim2sim_deploy/${MOTION_NAME}"
mkdir -p "$DEPLOY_DIR/exported" "$DEPLOY_DIR/params"

# Download policy.onnx
aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/exported/policy.onnx" "$DEPLOY_DIR/exported/policy.onnx" \
    --region "$AWS_REGION" 2>/dev/null || {
    echo "WARNING: policy.onnx not found on S3"
}

# Download policy.pt (JIT) as fallback
aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/exported/policy.pt" "$DEPLOY_DIR/exported/policy.pt" \
    --region "$AWS_REGION" 2>/dev/null || true

# Download deploy.yaml
aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/params/deploy.yaml" "$DEPLOY_DIR/params/deploy.yaml" \
    --region "$AWS_REGION" 2>/dev/null || {
    echo "WARNING: deploy.yaml not found on S3"
}

echo "Downloaded files:"
find "$DEPLOY_DIR" -type f | head -20

# ── Verify required files ────────────────────────────────────────────────────
POLICY_FILE=""
if [[ -f "$DEPLOY_DIR/exported/policy.onnx" ]]; then
    POLICY_FILE="$DEPLOY_DIR/exported/policy.onnx"
elif [[ -f "$DEPLOY_DIR/exported/policy.pt" ]]; then
    POLICY_FILE="$DEPLOY_DIR/exported/policy.pt"
else
    echo "ERROR: No policy file found (need policy.onnx or policy.pt)."
    echo "  Run export first: ./run.sh replay --task $TASK_TYPE --motion $MOTION_NAME"
    exit 1
fi

if [[ ! -f "$DEPLOY_DIR/params/deploy.yaml" ]]; then
    echo "ERROR: deploy.yaml not found. Run export first."
    exit 1
fi

# ── Detect MuJoCo scene ──────────────────────────────────────────────────────
SCENE_XML=""
for candidate in \
    "$WORKSPACE/unitree_mujoco/unitree_robots/g1/scene.xml" \
    "$WORKSPACE/unitree_model/G1/g1.xml"; do
    if [[ -f "$candidate" ]]; then
        SCENE_XML="$candidate"
        break
    fi
done

# ── Run Sim2Sim ──────────────────────────────────────────────────────────────
echo "=== Running Sim2Sim (MuJoCo) ==="
echo "Policy: $POLICY_FILE"
echo "Deploy: $DEPLOY_DIR/params/deploy.yaml"
echo "Scene: ${SCENE_XML:-auto-detect}"

OUTPUT_DIR="$WORKSPACE/sim2sim_output/${MOTION_NAME}"
mkdir -p "$OUTPUT_DIR"

cd "$WORKSPACE/dm-isaac-g1"

# Build sim2sim.py command
SIM2SIM_CMD=(
    python -u src/dm_isaac_g1/rl/scripts/sim2sim.py
    --policy "$POLICY_FILE"
    --deploy-yaml "$DEPLOY_DIR/params/deploy.yaml"
    --output-dir "$OUTPUT_DIR"
    --video-length "$VIDEO_LENGTH"
)

if [[ -n "$SCENE_XML" ]]; then
    SIM2SIM_CMD+=(--scene "$SCENE_XML")
fi

if [[ "$HEADLESS" == "true" ]]; then
    SIM2SIM_CMD+=(--headless --video)
else
    # GUI mode: ensure DISPLAY is set for VNC
    export DISPLAY="${DISPLAY:-:1}"
    echo "Using DISPLAY=$DISPLAY (connect via VNC to interact)"
fi

echo "Command: ${SIM2SIM_CMD[*]}"
"${SIM2SIM_CMD[@]}" 2>&1

SIM2SIM_EXIT=$?

# ── Upload Results to S3 ─────────────────────────────────────────────────────
echo "=== Uploading sim2sim results to S3 ==="

if [[ -d "$OUTPUT_DIR" ]] && ls "$OUTPUT_DIR"/*.mp4 &>/dev/null; then
    aws s3 sync "$OUTPUT_DIR/" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/sim2sim/" \
        --region "$AWS_REGION"
    echo "Sim2sim videos uploaded to s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/sim2sim/"
else
    echo "No video files to upload"
fi

echo "=== Sim2Sim Complete ==="
echo "Exit code: ${SIM2SIM_EXIT}"
echo "Finished at: $(date -u)"

# In GUI mode, keep container alive for VNC interaction
if [[ "$HEADLESS" != "true" ]]; then
    echo "Container staying alive for VNC interaction. Press Ctrl+C or stop the task to exit."
    sleep 86400
fi

exit ${SIM2SIM_EXIT}
