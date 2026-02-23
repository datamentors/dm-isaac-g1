#!/bin/bash
# Fine-tune GR00T N1.6 on ALL 7 Hospitality Datasets (merged)
#
# Prerequisites:
#   1. Convert all 7 datasets: for ds in G1_Fold_Towel G1_Clean_Table G1_Wipe_Table \
#        G1_Prepare_Fruit G1_Pour_Medicine G1_Organize_Tools G1_Pack_PingPong; do
#        python -m dm_isaac_g1.data.convert_to_groot \
#          --input /workspace/datasets/hospitality/$ds \
#          --output /workspace/datasets/groot/$ds; done
#   2. Merge datasets: python scripts/training/merge_datasets.py
#   3. Generate stats: see docs/FINETUNING_GUIDE.md
#
# Run inside dm-workstation container:
#   bash /workspace/dm-isaac-g1/scripts/training/finetune_hospitality_7ds.sh
set -e

cd /workspace/Isaac-GR00T

echo "================================================"
echo "Fine-tuning GR00T N1.6 on ALL 7 Hospitality Datasets"
echo "Dataset: /workspace/datasets/groot_merged (1400 episodes)"
echo "Embodiment: UNITREE_G1 (31 DOF state, 23 DOF action)"
echo "================================================"

conda run --no-capture-output -n unitree_sim_env \
torchrun --nproc_per_node=1 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /workspace/datasets/groot_merged \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir /workspace/checkpoints/groot-g1-gripper-hospitality-7ds \
    --save_total_limit 2 \
    --max_steps 10000 \
    --save_steps 2000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    2>&1 | tee /workspace/logs/finetune_hospitality_7ds.log
