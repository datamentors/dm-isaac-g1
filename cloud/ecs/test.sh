#!/usr/bin/env bash
# =============================================================================
# GPU Test Script (runs inside the ECS container)
# =============================================================================
# Validates the Docker image on a real GPU. Runs after ECR push.
#
# Tests:
#   1. test_environment.py — imports, versions, GPU memory, matmul benchmark
#   2. test_functional.py  — GROOT, MuJoCo sim+render, fine-tuning, Isaac Sim
#
# Results are uploaded to S3 for post-build review.
#
# Expected environment variables:
#   S3_BUCKET     - S3 bucket for results
#   AWS_REGION    - AWS region
#   GITHUB_TOKEN  - GitHub token for private repo access
# =============================================================================
set -euo pipefail

echo "=== DM Isaac G1 — GPU Build Validation ==="
echo "Started at: $(date -u)"

# ── Defaults ──────────────────────────────────────────────────────────────────
S3_BUCKET="${S3_BUCKET:?S3_BUCKET is required}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
WORKSPACE="/workspace"
TEST_DIR="$WORKSPACE/dm-isaac-g1/environments/tests"
RESULTS_DIR="/tmp/test-results"
mkdir -p "$RESULTS_DIR"

# ── GPU Check ─────────────────────────────────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# ── Vulkan Diagnostics ────────────────────────────────────────────────────────
echo "=== Vulkan Diagnostics ==="
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  (none)"
cat /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
vulkaninfo --summary 2>&1 | head -20 || true

# ── Setup ─────────────────────────────────────────────────────────────────────
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export OMNI_KIT_ALLOW_ROOT=1
export MUJOCO_GL=egl
unset DISPLAY

# ── Activate conda env ────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate unitree_sim_env 2>/dev/null || true
fi

# ── Pull Latest Code ──────────────────────────────────────────────────────────
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git config --global credential.helper store
    echo "https://x-access-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
fi

if [[ -d "$WORKSPACE/dm-isaac-g1/.git" ]]; then
    cd "$WORKSPACE/dm-isaac-g1"
    git fetch origin main 2>/dev/null || true
    git reset --hard origin/main 2>/dev/null || git pull --ff-only 2>/dev/null || true
    pip install -e . --quiet 2>/dev/null || true
    echo "dm-isaac-g1 at commit: $(git rev-parse --short HEAD)"
fi

# ── Run Environment Tests ─────────────────────────────────────────────────────
echo "=== Running Environment Tests ==="
ENV_EXIT=0
python -u "$TEST_DIR/test_environment.py" 2>&1 | tee "$RESULTS_DIR/test_environment.txt" || ENV_EXIT=$?
echo "Environment tests exit code: $ENV_EXIT"

# ── Run Functional Tests ──────────────────────────────────────────────────────
echo "=== Running Functional Tests ==="
FUNC_EXIT=0
python -u "$TEST_DIR/test_functional.py" 2>&1 | tee "$RESULTS_DIR/test_functional.txt" || FUNC_EXIT=$?
echo "Functional tests exit code: $FUNC_EXIT"

# ── Upload Results to S3 ──────────────────────────────────────────────────────
echo "=== Uploading test results to S3 ==="
S3_PREFIX="build-tests/gpu/$(date +%Y%m%d-%H%M%S)"
aws s3 sync "$RESULTS_DIR/" "s3://${S3_BUCKET}/${S3_PREFIX}/" --region "$AWS_REGION" 2>/dev/null || true
echo "Results uploaded to: s3://${S3_BUCKET}/${S3_PREFIX}/"

# ── Summary ───────────────────────────────────────────────────────────────────
OVERALL_EXIT=0
if [[ $ENV_EXIT -ne 0 || $FUNC_EXIT -ne 0 ]]; then
    OVERALL_EXIT=1
    echo ""
    echo "=========================================="
    echo "  GPU TESTS FAILED"
    echo "  Environment: exit $ENV_EXIT"
    echo "  Functional:  exit $FUNC_EXIT"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "  ALL GPU TESTS PASSED"
    echo "=========================================="
fi

echo "Finished at: $(date -u)"
exit $OVERALL_EXIT
