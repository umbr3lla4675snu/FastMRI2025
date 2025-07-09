#!/bin/bash

TRAINING_MODE="standalone"
SEED=216
NAFNET_WIDTH=32
NAFNET_MIDDLE_BLKS=8
NAFNET_ENC_BLKS="1 1 2 4"
NAFNET_DEC_BLKS="1 1 2 2"

python train_nafnet.py \
    -b 1 \
    -r 100 \
    -e 5 \
    -l 1e-4 \
    -n 'test_nafnet' \
    -t '/root/Data/train/' \
    -v '/root/Data/val/' \
    --training-mode $TRAINING_MODE \
    --varnet-checkpoint-path '/root/result/test_Varnet/checkpoints/best_model.pt' \
    --varnet-cascade 1 \
    --varnet-chans 9 \
    --varnet-sens-chans 4 \
    --nafnet-width $NAFNET_WIDTH \
    --nafnet-middle-blks $NAFNET_MIDDLE_BLKS \
    --nafnet-enc-blks $NAFNET_ENC_BLKS \
    --nafnet-dec-blks $NAFNET_DEC_BLKS \
    --use-weighted-loss \
    --seed $SEED

echo "NAFNet training completed!"

# Alternative: Train combined model end-to-end
# Uncomment the following section to train the combined model instead

# echo "Starting combined VarNet+NAFNet training..."
# python train_nafnet.py \
#     --GPU-NUM $GPU_NUM \
#     --batch-size $BATCH_SIZE \
#     --num-epochs 30 \
#     --lr 5e-5 \
#     --report-interval $REPORT_INTERVAL \
#     --net-name "${NET_NAME}_combined" \
#     --data-path-train $DATA_PATH_TRAIN \
#     --data-path-val $DATA_PATH_VAL \
#     --training-mode combined \
#     --varnet-checkpoint-path $VARNET_CHECKPOINT \
#     --varnet-cascade $VARNET_CASCADE \
#     --varnet-chans $VARNET_CHANS \
#     --varnet-sens-chans $VARNET_SENS_CHANS \
#     --nafnet-width $NAFNET_WIDTH \
#     --nafnet-middle-blks $NAFNET_MIDDLE_BLKS \
#     --nafnet-enc-blks $NAFNET_ENC_BLKS \
#     --nafnet-dec-blks $NAFNET_DEC_BLKS \
#     --seed $SEED
