#!/bin/bash

echo "Running Triple Input NAFNet reconstruction..."
echo "This script uses:"
echo "  1. Pre-trained VarNet model"
echo "  2. Trained Triple Input NAFNet model"
echo "  3. GRAPPA images from image_grappa key"
echo "  4. Input images from image_input key"

python reconstruct_dual_input_nafnet.py \
    --GPU-NUM 0 \
    --batch-size 1 \
    --path_data '/root/Data/leaderboard/' \
    --net_name 'dual_input_nafnet' \
    --varnet-checkpoint-path '/root/result/Varnet_augmentation_no_delay/checkpoints/best_model.pt' \

echo "Dual Input NAFNet reconstruction completed!"
