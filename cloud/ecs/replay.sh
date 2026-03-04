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

# ── Setup Vulkan (lavapipe software renderer) ────────────────────────────────
# The AWS ECS GPU AMI NVIDIA driver is compute-only — the kernel module
# doesn't support Vulkan. We use Mesa's lavapipe (CPU Vulkan 1.3) for
# Isaac Sim headless rendering. CUDA/GPU compute still works normally.
if [[ ! -f /usr/share/vulkan/icd.d/lvp_icd.x86_64.json ]]; then
    echo "Installing Mesa Vulkan drivers (lavapipe)..."
    apt-get update -qq 2>/dev/null
    apt-get install -y -qq mesa-vulkan-drivers 2>/dev/null || true
fi
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export XDG_RUNTIME_DIR=/tmp/xdg
mkdir -p "$XDG_RUNTIME_DIR"
echo "Vulkan configured (lavapipe software renderer)"

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

# Step 1: Standalone PyTorch export (no Isaac Sim required)
# Isaac Sim initialization hangs on ECS because PhysX can't find CUDA GPU in the container runtime.
# Instead, we extract the policy directly from the RSL-RL checkpoint using pure PyTorch.
echo "=== Step 1: Exporting policy (standalone PyTorch) ==="
python -u - "$CKPT_DIR/$CHECKPOINT_FILE" "$CKPT_DIR/exported" << 'PYEOF'
import os, sys, torch

ckpt_path = sys.argv[1]
export_dir = sys.argv[2]
os.makedirs(export_dir, exist_ok=True)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")

model_state_dict = ckpt.get("model_state_dict", ckpt)

# Find actor network dimensions from weights
actor_keys = sorted([k for k in model_state_dict if "actor" in k and "weight" in k])
if not actor_keys:
    print("ERROR: No actor weights found in checkpoint")
    sys.exit(1)

obs_dim = model_state_dict[actor_keys[0]].shape[1]
action_dim = model_state_dict[actor_keys[-1]].shape[0]
print(f"Actor: obs_dim={obs_dim}, action_dim={action_dim}")
print(f"Actor layers: {actor_keys}")

# Determine hidden dims from intermediate layers
hidden_dims = []
for k in actor_keys:
    if k != actor_keys[0] and k != actor_keys[-1]:
        hidden_dims.append(model_state_dict[k].shape[0])
if not hidden_dims:
    # Only input -> output, infer from first layer output
    hidden_dims = [model_state_dict[actor_keys[0]].shape[0]]
print(f"Hidden dims: {hidden_dims}")

# Build a simple MLP matching the actor architecture
from collections import OrderedDict
import torch.nn as nn

# Extract just the actor weights
actor_state = OrderedDict()
for k, v in model_state_dict.items():
    if k.startswith("actor."):
        actor_state[k.replace("actor.", "")] = v

# Build matching MLP
layers = []
all_actor_weight_keys = sorted([k for k in actor_state if "weight" in k])
for i, wk in enumerate(all_actor_weight_keys):
    bk = wk.replace("weight", "bias")
    in_f = actor_state[wk].shape[1]
    out_f = actor_state[wk].shape[0]
    layers.append((wk.rsplit(".", 1)[0] + "." + wk.rsplit(".", 1)[0].split(".")[-1] if "." in wk else f"layer_{i}", nn.Linear(in_f, out_f)))
    if i < len(all_actor_weight_keys) - 1:  # Add activation except for last layer
        layers.append((f"activation_{i}", nn.ELU()))

# Simpler approach: just build the Sequential from the state dict structure
class ActorMLP(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        weight_keys = sorted([k for k in state_dict if "weight" in k])
        mlp_layers = []
        for i, wk in enumerate(weight_keys):
            in_f = state_dict[wk].shape[1]
            out_f = state_dict[wk].shape[0]
            mlp_layers.append(nn.Linear(in_f, out_f))
            if i < len(weight_keys) - 1:
                mlp_layers.append(nn.ELU())
        self.net = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.net(x)

actor = ActorMLP(actor_state)

# Load the weights by mapping to the sequential structure
new_state = OrderedDict()
weight_keys = sorted([k for k in actor_state if "weight" in k])
for i, wk in enumerate(weight_keys):
    bk = wk.replace("weight", "bias")
    # Map to sequential indices: each Linear is at index 2*i (ELU at 2*i+1)
    seq_idx = 2 * i if i < len(weight_keys) - 1 else 2 * i
    # Actually just 2*i for Linear with activation, last one has no activation
    if i < len(weight_keys) - 1:
        new_state[f"net.{2*i}.weight"] = actor_state[wk]
        if bk in actor_state:
            new_state[f"net.{2*i}.bias"] = actor_state[bk]
    else:
        # Last layer (no activation after)
        idx = 2 * i
        new_state[f"net.{idx}.weight"] = actor_state[wk]
        if bk in actor_state:
            new_state[f"net.{idx}.bias"] = actor_state[bk]

actor.load_state_dict(new_state)
actor.eval()
print(f"Actor network built and loaded ({sum(p.numel() for p in actor.parameters())} parameters)")

# Export JIT
dummy = torch.zeros(1, obs_dim)
try:
    traced = torch.jit.trace(actor, dummy)
    jit_path = os.path.join(export_dir, "policy.pt")
    torch.jit.save(traced, jit_path)
    print(f"Exported JIT: {jit_path} ({os.path.getsize(jit_path):,} bytes)")
except Exception as e:
    print(f"JIT export failed: {e}")

# Export ONNX
try:
    onnx_path = os.path.join(export_dir, "policy.onnx")
    torch.onnx.export(actor, dummy, onnx_path, input_names=["obs"], output_names=["actions"], opset_version=11)
    print(f"Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")
except Exception as e:
    print(f"ONNX export failed: {e}")

# Also save the raw state dict for reference
raw_path = os.path.join(export_dir, "policy_state_dict.pt")
torch.save({"model_state_dict": model_state_dict, "obs_dim": obs_dim, "action_dim": action_dim}, raw_path)
print(f"Saved state dict: {raw_path} ({os.path.getsize(raw_path):,} bytes)")

print("\nExport complete!")
for f in sorted(os.listdir(export_dir)):
    print(f"  {f}: {os.path.getsize(os.path.join(export_dir, f)):,} bytes")
PYEOF

PLAY_EXIT=$?
echo "Export exit code: $PLAY_EXIT"

# Step 2: Try Isaac Sim video recording with timeout (optional, often fails on ECS)
# Isaac Sim's renderer + PhysX need a working CUDA GPU device, which ECS containers
# may not expose correctly. We give it 5 minutes — if it doesn't finish, we skip video.
echo "=== Step 2: Attempting video recording (5 min timeout) ==="
timeout 300 python -u "$PLAY_SCRIPT" \
    --task "$TASK_ID" \
    --checkpoint "$CKPT_DIR/$CHECKPOINT_FILE" \
    --video --video_length "$VIDEO_LENGTH" \
    --headless 2>&1 || {
    VIDEO_EXIT=$?
    if [[ $VIDEO_EXIT -eq 124 ]]; then
        echo "WARNING: Video recording timed out (GPU rendering not available on ECS)"
    else
        echo "WARNING: Video recording failed with exit code $VIDEO_EXIT"
    fi
    echo "Export completed successfully — video skipped."
}

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
