#!/bin/bash

echo "Running Triple Input NAFNet reconstruction..."
echo "This script uses:"
echo "  1. Pre-trained VarNet model"
echo "  2. Trained Triple Input NAFNet model"
echo "  3. GRAPPA images from image_grappa key"
echo "  4. Input images from image_input key"

python reconstruct_triple_input_nafnet.py \
    --GPU-NUM 0 \
    --batch-size 1 \
    --data-path '/root/Data/test/' \
    --out-dir '/root/result/triple_input_reconstructions' \
    --varnet-checkpoint-path '/root/result/test_varnet/checkpoints/best_model.pt' \
    --nafnet-checkpoint-path '/root/result/triple_input_nafnet/checkpoints/best_model.pt'

echo "Triple Input NAFNet reconstruction completed!"
echo "Results saved to /root/result/triple_input_reconstructions/"
echo "  - enhanced/: Final enhanced reconstructions"
echo "  - varnet_only/: VarNet-only reconstructions for comparison"
