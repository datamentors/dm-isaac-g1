#!/usr/bin/env bash
# =============================================================================
# Sim2Sim Script (runs inside the ECS container)
# =============================================================================
# Validates a trained Isaac Lab RL/Mimic policy by deploying it in MuJoCo.
#
# Pipeline:
#   1. Downloads policy.onnx + deploy.yaml from S3 (previously exported by play.py)
#   2. Installs unitree_mujoco + unitree_rl_lab deploy infrastructure
#   3. Runs the policy in MuJoCo headless simulation
#   4. Records video of the MuJoCo rendering
#   5. Uploads videos to S3
#
# Expected environment variables:
#   TASK_TYPE         - "mimic" or "rl"
#   TASK_ID           - Gymnasium task ID (e.g., DM-G1-29dof-FALCON)
#   MOTION_NAME       - Motion/task name for S3 paths
#   S3_BUCKET         - S3 bucket
#   AWS_REGION        - AWS region
#   VIDEO_LENGTH      - Video length in seconds (default: 10)
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

# Clone unitree_rl_lab (has deploy_mujoco scripts)
if [[ ! -d "$WORKSPACE/unitree_rl_lab" ]]; then
    echo "Cloning unitree_rl_lab..."
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git "$WORKSPACE/unitree_rl_lab"
fi

# Clone unitree_mujoco (MuJoCo sim for Unitree robots)
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

pip install --quiet mujoco>=3.2.6 onnxruntime numpy opencv-python pyyaml 2>/dev/null || true

# Install dm-isaac-g1
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
    echo "WARNING: policy.onnx not found on S3. Will need to export first via: ./run.sh replay --task $TASK_TYPE ..."
}

# Download policy.pt (JIT)
aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/exported/policy.pt" "$DEPLOY_DIR/exported/policy.pt" \
    --region "$AWS_REGION" 2>/dev/null || true

# Download deploy.yaml
aws s3 cp "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/params/deploy.yaml" "$DEPLOY_DIR/params/deploy.yaml" \
    --region "$AWS_REGION" 2>/dev/null || {
    echo "WARNING: deploy.yaml not found on S3"
}

echo "Deployed files:"
find "$DEPLOY_DIR" -type f | head -20

# ── Verify required files ────────────────────────────────────────────────────
if [[ ! -f "$DEPLOY_DIR/exported/policy.onnx" ]]; then
    echo "ERROR: policy.onnx not found. Run replay/export first:"
    echo "  ./run.sh replay --task $TASK_TYPE --motion $MOTION_NAME"
    exit 1
fi

# ── Run Sim2Sim in MuJoCo ────────────────────────────────────────────────────
echo "=== Running Sim2Sim (MuJoCo) ==="

# Set MuJoCo to use EGL for headless rendering
export MUJOCO_GL=egl

OUTPUT_DIR="$WORKSPACE/sim2sim_output/${MOTION_NAME}"
mkdir -p "$OUTPUT_DIR"

cd "$WORKSPACE/dm-isaac-g1"

# Use the deploy_mujoco infrastructure from unitree_rl_lab if available
DEPLOY_SCRIPT="$WORKSPACE/unitree_rl_lab/deploy/deploy_mujoco/deploy_mujoco.py"

if [[ -f "$DEPLOY_SCRIPT" ]]; then
    echo "Using unitree_rl_lab deploy_mujoco..."

    # Create a YAML config for this specific task
    DEPLOY_CFG="$DEPLOY_DIR/sim2sim_config.yaml"
    cat > "$DEPLOY_CFG" << CFGEOF
# Auto-generated sim2sim config for ${TASK_ID}
robot: g1
dof: 29
policy_path: ${DEPLOY_DIR}/exported/policy.onnx
deploy_yaml_path: ${DEPLOY_DIR}/params/deploy.yaml
video_output: ${OUTPUT_DIR}/sim2sim_${MOTION_NAME}.mp4
video_length: ${VIDEO_LENGTH}
headless: true
CFGEOF

    python -u "$DEPLOY_SCRIPT" "$DEPLOY_CFG" 2>&1 || {
        echo "deploy_mujoco.py failed, falling back to standalone sim2sim..."
    }
else
    echo "unitree_rl_lab deploy_mujoco not found, using standalone sim2sim..."
fi

# Fallback: standalone sim2sim using our own script
if [[ ! -f "$OUTPUT_DIR/sim2sim_${MOTION_NAME}.mp4" ]]; then
    python3 -u << 'PYEOF'
import os
import sys
import numpy as np
import onnxruntime as ort
import yaml

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco not installed")
    sys.exit(1)

deploy_dir = os.environ.get("DEPLOY_DIR", "")
output_dir = os.environ.get("OUTPUT_DIR", "")
motion_name = os.environ.get("MOTION_NAME", "unknown")
video_length = int(os.environ.get("VIDEO_LENGTH", "10"))

# Load deploy config
deploy_yaml = os.path.join(deploy_dir, "params", "deploy.yaml")
if os.path.exists(deploy_yaml):
    with open(deploy_yaml) as f:
        deploy_cfg = yaml.safe_load(f)
    print(f"Deploy config loaded: {list(deploy_cfg.keys())}")
else:
    print("WARNING: No deploy.yaml found, using defaults")
    deploy_cfg = {}

# Load ONNX policy
policy_path = os.path.join(deploy_dir, "exported", "policy.onnx")
print(f"Loading policy: {policy_path}")
session = ort.InferenceSession(policy_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name
print(f"  Input: {input_name} shape={input_shape}")
print(f"  Output: {output_name}")

# Find G1 MuJoCo model
mujoco_model_paths = [
    "/workspace/unitree_mujoco/unitree_robots/g1/scene.xml",
    "/workspace/unitree_model/G1/g1.xml",
    "/workspace/unitree_rl_lab/deploy/deploy_mujoco/robots/g1/scene.xml",
]
model_path = None
for p in mujoco_model_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    print("ERROR: No G1 MuJoCo model found")
    print("Searched:", mujoco_model_paths)
    sys.exit(1)

print(f"MuJoCo model: {model_path}")

# Load model and create simulation
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Setup rendering
os.environ.setdefault("MUJOCO_GL", "egl")
renderer = mujoco.Renderer(model, height=720, width=1280)

# Get observation and action dimensions from model
obs_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]
n_joints = model.nu

print(f"Obs dim: {obs_dim}, Joints: {n_joints}")

# Simulation loop
dt = model.opt.timestep
fps = 30
steps_per_frame = max(1, int(1.0 / (dt * fps)))
total_frames = video_length * fps
frames = []

print(f"Running sim2sim for {video_length}s ({total_frames} frames)...")

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

for frame_idx in range(total_frames):
    # Build observation from joint state
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    # Construct observation matching training format
    # This is model-specific; adapt based on deploy.yaml
    obs = np.concatenate([qpos, qvel]).astype(np.float32)

    # Pad or trim to match expected input dimension
    if len(obs) < obs_dim:
        obs = np.pad(obs, (0, obs_dim - len(obs)))
    elif len(obs) > obs_dim:
        obs = obs[:obs_dim]

    obs = obs.reshape(1, -1)

    # Run policy inference
    action = session.run([output_name], {input_name: obs})[0]
    action = action.flatten()

    # Apply actions (clip to actuator limits)
    n_act = min(len(action), n_joints)
    data.ctrl[:n_act] = np.clip(action[:n_act],
                                 model.actuator_ctrlrange[:n_act, 0],
                                 model.actuator_ctrlrange[:n_act, 1])

    # Step physics
    for _ in range(steps_per_frame):
        mujoco.mj_step(model, data)

    # Render frame
    renderer.update_scene(data)
    frame = renderer.render()
    frames.append(frame)

    if (frame_idx + 1) % fps == 0:
        print(f"  {frame_idx + 1}/{total_frames} frames ({(frame_idx+1)/fps:.0f}s)")

# Save video
if frames:
    import cv2
    video_path = os.path.join(output_dir, f"sim2sim_{motion_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved: {video_path}")
else:
    print("WARNING: No frames rendered")

print("Sim2Sim complete!")
PYEOF
fi

SIM2SIM_EXIT=$?

# ── Upload Results to S3 ─────────────────────────────────────────────────────
echo "=== Uploading sim2sim results to S3 ==="

if [[ -d "$OUTPUT_DIR" ]]; then
    aws s3 sync "$OUTPUT_DIR/" "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/sim2sim/" \
        --region "$AWS_REGION"
    echo "Sim2sim videos uploaded to s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/sim2sim/"
fi

echo "=== Sim2Sim Complete ==="
echo "Exit code: ${SIM2SIM_EXIT:-0}"
echo "Finished at: $(date -u)"
exit ${SIM2SIM_EXIT:-0}
