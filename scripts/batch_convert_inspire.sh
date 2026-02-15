#!/bin/bash
# ==============================================================================
# Batch Convert All Datasets to Inspire Format
# ==============================================================================
# This script converts all downloaded G1 datasets to the unified 53 DOF
# Inspire hand format for multi-dataset GROOT training.
#
# Usage (run inside isaac-sim container):
#   source /opt/conda/etc/profile.d/conda.sh
#   conda activate grootenv
#   bash /workspace/scripts/batch_convert_inspire.sh
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="/workspace/datasets"
OUTPUT_DIR="/workspace/datasets_inspire"
CONVERT_SCRIPT="/workspace/scripts/convert_to_inspire.py"

echo "=============================================="
echo "Batch Dataset Conversion to Inspire Format"
echo "=============================================="
echo "Source: $DATASETS_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Counter for progress
TOTAL=0
SUCCESS=0
FAILED=0

# ==============================================================================
# CATEGORY 1: Hospitality Datasets (Simple Gripper -> Inspire)
# ==============================================================================
echo ""
echo "=== Category 1: Hospitality Datasets (Gripper -> Inspire) ==="
echo ""

HOSPITALITY_DATASETS=(
    "G1_Fold_Towel"
    "G1_Clean_Table"
    "G1_Wipe_Table"
    "G1_Prepare_Fruit"
    "G1_Pour_Medicine"
    "G1_Organize_Tools"
)

for dataset in "${HOSPITALITY_DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    INPUT="$DATASETS_DIR/$dataset"
    OUTPUT="$OUTPUT_DIR/${dataset}_Inspire"

    if [ -d "$INPUT" ]; then
        echo "[$TOTAL] Converting $dataset (gripper -> inspire)..."
        if python "$CONVERT_SCRIPT" --input "$INPUT" --output "$OUTPUT" --hand-type gripper; then
            SUCCESS=$((SUCCESS + 1))
            echo "    SUCCESS"
        else
            FAILED=$((FAILED + 1))
            echo "    FAILED"
        fi
    else
        echo "[$TOTAL] SKIP: $dataset not found at $INPUT"
        FAILED=$((FAILED + 1))
    fi
done

# ==============================================================================
# CATEGORY 2: Dex3 Datasets (Dex3 -> Inspire)
# ==============================================================================
echo ""
echo "=== Category 2: Dex3 Datasets (Dex3 -> Inspire) ==="
echo ""

DEX3_DATASETS=(
    "G1_Dex3_ToastedBread_Dataset"
    "G1_Dex3_BlockStacking_Dataset"
)

for dataset in "${DEX3_DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    INPUT="$DATASETS_DIR/$dataset"
    OUTPUT="$OUTPUT_DIR/${dataset%_Dataset}_Inspire"

    if [ -d "$INPUT" ]; then
        echo "[$TOTAL] Converting $dataset (dex3 -> inspire)..."
        if python "$CONVERT_SCRIPT" --input "$INPUT" --output "$OUTPUT" --hand-type dex3; then
            SUCCESS=$((SUCCESS + 1))
            echo "    SUCCESS"
        else
            FAILED=$((FAILED + 1))
            echo "    FAILED"
        fi
    else
        echo "[$TOTAL] SKIP: $dataset not found at $INPUT"
        FAILED=$((FAILED + 1))
    fi
done

# ==============================================================================
# CATEGORY 3: Teleop Dataset (Tri-finger -> Inspire)
# ==============================================================================
echo ""
echo "=== Category 3: Teleop Dataset (Tri-finger -> Inspire) ==="
echo ""

TELEOP_DATASETS=(
    "PhysicalAI-Robotics-GR00T-Teleop-G1"
)

for dataset in "${TELEOP_DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    INPUT="$DATASETS_DIR/$dataset"
    OUTPUT="$OUTPUT_DIR/G1_Teleop_Inspire"

    if [ -d "$INPUT" ]; then
        echo "[$TOTAL] Converting $dataset (trifinger -> inspire)..."
        if python "$CONVERT_SCRIPT" --input "$INPUT" --output "$OUTPUT" --hand-type trifinger; then
            SUCCESS=$((SUCCESS + 1))
            echo "    SUCCESS"
        else
            FAILED=$((FAILED + 1))
            echo "    FAILED"
        fi
    else
        echo "[$TOTAL] SKIP: $dataset not found at $INPUT"
        FAILED=$((FAILED + 1))
    fi
done

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "Conversion Summary"
echo "=============================================="
echo "Total datasets: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""
echo "Converted datasets are in: $OUTPUT_DIR"
echo ""

# List output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output contents:"
    ls -la "$OUTPUT_DIR"
fi

echo ""
echo "Next steps:"
echo "1. Verify converted datasets with: python convert_to_inspire.py --input <path> --dry-run"
echo "2. Combine all datasets with: python combine_inspire_datasets.py"
echo "3. Run test training: python gr00t/experiment/launch_finetune.py --max-steps 100"
