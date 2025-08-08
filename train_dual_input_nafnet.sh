#!/bin/bash

# dual Input NAFNet Training Configuration
SEED=216
NAFNET_WIDTH=48
NAFNET_MIDDLE_BLKS=10
NAFNET_ENC_BLKS="2 4 6 8"
NAFNET_DEC_BLKS="2 4 4 2"

echo "Starting Dual Input NAFNet training..."
echo "This model uses VarNet output + GRAPPA image as inputs"

python train_dual_input_nafnet.py \
    -b 1 \
    -r 100 \
    -e 20 \
    -l 1e-3 \
    -n 'dual_input_nafnet' \
    -t '/root/Data/train/' \
    -v '/root/Data/val/' \
    --varnet-checkpoint-path '/root/result/Varnet_augmentation_no_delay/checkpoints/best_model.pt' \
    --varnet-cascade 6 \
    --varnet-chans 13 \
    --varnet-sens-chans 5 \
    --nafnet-width $NAFNET_WIDTH \
    --nafnet-middle-blks $NAFNET_MIDDLE_BLKS \
    --nafnet-enc-blks $NAFNET_ENC_BLKS \
    --nafnet-dec-blks $NAFNET_DEC_BLKS \
    --seed $SEED 

echo $! > /root/result/dual_input_nafnet/logs/training.pid

echo "dual Input NAFNet training completed!"
echo "Model saved to /root/result/dual_input_nafnet/"
echo "This model processes:"
echo "  1. VarNet reconstructed image"
echo "  2. GRAPPA reconstructed image (from image_grappa key)"
echo "To produce enhanced final reconstruction"
echo ""
echo "Resume training: If checkpoint exists in exp_dir, training will resume automatically"
