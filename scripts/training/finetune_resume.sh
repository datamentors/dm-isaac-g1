#!/bin/bash
# Resume fine-tuning from a HuggingFace checkpoint
#
# Usage:
#   bash scripts/training/finetune_resume.sh \
#     <base_checkpoint_path> <dataset_path> <output_dir> [additional_steps]
#
# Example (resume fold-towel from step 6000 for 4000 more steps):
#   bash /workspace/dm-isaac-g1/scripts/training/finetune_resume.sh \
#     /workspace/checkpoints/groot-g1-gripper-fold-towel \
#     /workspace/datasets/groot/G1_Fold_Towel \
#     /workspace/checkpoints/groot-g1-gripper-fold-towel-full \
#     4000
set -e

BASE_CHECKPOINT=${1:?Usage: $0 <base_checkpoint> <dataset_path> <output_dir> [steps]}
DATASET_PATH=${2:?Usage: $0 <base_checkpoint> <dataset_path> <output_dir> [steps]}
OUTPUT_DIR=${3:?Usage: $0 <base_checkpoint> <dataset_path> <output_dir> [steps]}
MAX_STEPS=${4:-4000}

cd /workspace/Isaac-GR00T

echo "================================================"
echo "Resuming GR00T Fine-tuning"
echo "Base checkpoint: $BASE_CHECKPOINT"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Additional steps: $MAX_STEPS"
echo "Embodiment: UNITREE_G1"
echo "================================================"

conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "$BASE_CHECKPOINT" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir "$OUTPUT_DIR" \
    --save_total_limit 2 \
    --max_steps "$MAX_STEPS" \
    --save_steps 2000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /workspace/logs/finetune_resume.log
