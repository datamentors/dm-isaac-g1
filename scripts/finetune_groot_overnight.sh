#!/bin/bash
# =============================================================================
# GROOT N1.6 Overnight Fine-Tuning Script
# =============================================================================
# This script sets up and runs GROOT fine-tuning for 8-12 hours overnight.
#
# Usage:
#   ./scripts/finetune_groot_overnight.sh [dataset] [gpu]
#
# Examples:
#   ./scripts/finetune_groot_overnight.sh gr1_arms_only 0
#   ./scripts/finetune_groot_overnight.sh droid 0
#
# =============================================================================

set -e

# Configuration
DATASET=${1:-"gr1_arms_only"}
GPU_ID=${2:-0}
WORKSPACE=/workspace
GROOT_DIR=${WORKSPACE}/Isaac-GR00T
DATASET_DIR=${WORKSPACE}/datasets
CHECKPOINT_DIR=${WORKSPACE}/checkpoints/groot_$(date +%Y%m%d_%H%M%S)

# Training hyperparameters (tuned for 8-10 hour training)
BATCH_SIZE=32
NUM_STEPS=5000
LEARNING_RATE=1e-5
SAVE_EVERY=1000
LOG_EVERY=100

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GROOT N1.6 Overnight Fine-Tuning     ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  GPU: ${GPU_ID}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Steps: ${NUM_STEPS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Output: ${CHECKPOINT_DIR}"
echo ""

# =============================================================================
# Step 1: Environment Setup
# =============================================================================

echo -e "${YELLOW}[1/5] Setting up environment...${NC}"

# Set GPU
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Check if we're in the right environment
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: 'uv' not found. Make sure you're in Isaac-GR00T environment${NC}"
    echo "Run: cd ${GROOT_DIR} && source .venv/bin/activate"
    exit 1
fi

# Clone/update GR00T if needed
if [ ! -d "${GROOT_DIR}" ]; then
    echo "Cloning Isaac-GR00T..."
    cd ${WORKSPACE}
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
    cd ${GROOT_DIR}
    uv sync
else
    echo "GR00T already present at ${GROOT_DIR}"
fi

cd ${GROOT_DIR}

# =============================================================================
# Step 2: Download Dataset
# =============================================================================

echo -e "${YELLOW}[2/5] Downloading dataset: ${DATASET}...${NC}"

mkdir -p ${DATASET_DIR}

case ${DATASET} in
    "gr1_arms_only")
        DATASET_PATH=${DATASET_DIR}/gr1_arms_only
        if [ ! -d "${DATASET_PATH}" ]; then
            huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
                --repo-type dataset \
                --include "gr1_arms_only/**" \
                --local-dir ${DATASET_PATH} \
                --resume-download
        else
            echo "Dataset already exists at ${DATASET_PATH}"
        fi
        ;;

    "gr1_arms_waist")
        DATASET_PATH=${DATASET_DIR}/gr1_arms_waist
        if [ ! -d "${DATASET_PATH}" ]; then
            echo -e "${YELLOW}Warning: This dataset is large (~240k trajectories)${NC}"
            echo "Training may take 24-48 hours"
            read -p "Continue? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
                    --repo-type dataset \
                    --include "gr1_arms_waist/**" \
                    --local-dir ${DATASET_PATH} \
                    --resume-download
            else
                echo "Aborting."
                exit 1
            fi
        fi
        ;;

    "droid")
        DATASET_PATH=${DATASET_DIR}/droid_subset
        echo -e "${YELLOW}Note: DROID requires separate download and conversion${NC}"
        echo "See: https://droid-dataset.github.io/"
        if [ ! -d "${DATASET_PATH}" ]; then
            echo -e "${RED}Error: DROID dataset not found at ${DATASET_PATH}${NC}"
            echo "Download and convert DROID first, then re-run this script."
            exit 1
        fi
        ;;

    *)
        echo -e "${RED}Unknown dataset: ${DATASET}${NC}"
        echo "Available: gr1_arms_only, gr1_arms_waist, droid"
        exit 1
        ;;
esac

# =============================================================================
# Step 3: Create Embodiment Config
# =============================================================================

echo -e "${YELLOW}[3/5] Creating embodiment configuration...${NC}"

CONFIG_DIR=${GROOT_DIR}/configs
mkdir -p ${CONFIG_DIR}

cat > ${CONFIG_DIR}/unitree_g1_config.json << 'EOF'
{
    "embodiment_tag": "UNITREE_G1",
    "description": "Unitree G1 humanoid robot configuration for GROOT fine-tuning",
    "observation_space": {
        "state": {
            "shape": [48],
            "description": "23 DOF positions + velocities + contact flags",
            "components": {
                "left_arm": 7,
                "right_arm": 7,
                "left_hand": 6,
                "right_hand": 6,
                "waist": 3
            }
        },
        "images": {
            "camera_0": {
                "shape": [480, 640, 3],
                "format": "RGB",
                "description": "Head camera RGB image"
            }
        }
    },
    "action_space": {
        "shape": [29],
        "type": "continuous",
        "description": "Joint position targets",
        "groups": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26},
            "waist": {"start": 26, "end": 29}
        }
    },
    "action_type": "relative",
    "control_frequency_hz": 30,
    "physics_engine": "mujoco"
}
EOF

echo "Created config at ${CONFIG_DIR}/unitree_g1_config.json"

# =============================================================================
# Step 4: Prepare Logging
# =============================================================================

echo -e "${YELLOW}[4/5] Setting up logging...${NC}"

mkdir -p ${CHECKPOINT_DIR}
LOG_FILE=${CHECKPOINT_DIR}/training.log

# Save training config
cat > ${CHECKPOINT_DIR}/config.yaml << EOF
# Training Configuration
date: $(date)
dataset: ${DATASET}
dataset_path: ${DATASET_PATH}
base_model: nvidia/GR00T-N1.6-3B
batch_size: ${BATCH_SIZE}
num_steps: ${NUM_STEPS}
learning_rate: ${LEARNING_RATE}
gpu: ${GPU_ID}
save_every: ${SAVE_EVERY}
EOF

echo "Logs will be saved to: ${LOG_FILE}"
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}"

# =============================================================================
# Step 5: Start Training
# =============================================================================

echo -e "${YELLOW}[5/5] Starting training...${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Started at $(date)          ${NC}"
echo -e "${GREEN}  Expected completion: ~8-10 hours     ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To monitor training:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi -l 5"
echo ""

# Run training
uv run python gr00t/scripts/launch_finetune.py \
    --base-model nvidia/GR00T-N1.6-3B \
    --dataset-path ${DATASET_PATH} \
    --modality-config ${CONFIG_DIR}/unitree_g1_config.json \
    --output-dir ${CHECKPOINT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --num-steps ${NUM_STEPS} \
    --learning-rate ${LEARNING_RATE} \
    --save-every ${SAVE_EVERY} \
    --log-every ${LOG_EVERY} \
    --mixed-precision fp16 \
    2>&1 | tee ${LOG_FILE}

# =============================================================================
# Done
# =============================================================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!                   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "To deploy to GROOT server, run:"
echo ""
echo "  docker exec groot-server bash -c '"
echo "    pkill -f run_gr00t_server"
echo "    cd /workspace/gr00t &&"
echo "    python gr00t/eval/run_gr00t_server.py \\"
echo "      --model-path ${CHECKPOINT_DIR}/final \\"
echo "      --embodiment-tag UNITREE_G1 \\"
echo "      --port 5555"
echo "  '"
