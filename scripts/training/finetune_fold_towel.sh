#!/bin/bash
# Fine-tune GR00T N1.6 on G1_Fold_Towel (single task)
#
# Prerequisites:
#   1. Convert dataset: python -m dm_isaac_g1.data.convert_to_groot \
#        --input /workspace/datasets/hospitality/G1_Fold_Towel \
#        --output /workspace/datasets/groot/G1_Fold_Towel
#   2. Generate stats: see docs/FINETUNING_GUIDE.md
#
# Run inside dm-workstation container:
#   bash /workspace/dm-isaac-g1/scripts/training/finetune_fold_towel.sh
set -e

cd /workspace/Isaac-GR00T

echo "================================================"
echo "Fine-tuning GR00T N1.6 on G1_Fold_Towel"
echo "Dataset: /workspace/datasets/groot/G1_Fold_Towel"
echo "Embodiment: UNITREE_G1 (31 DOF state, 23 DOF action)"
echo "================================================"

conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/groot/G1_Fold_Towel \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-gripper-fold-towel \
    --save_total_limit 2 \
    --max_steps 10000 \
    --save_steps 2000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /workspace/logs/finetune_fold_towel.log
