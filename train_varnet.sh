python train_varnet.py \
  -b 1 \
  -e 10 \
  -l 0.0001 \
  -r 100 \
  -n 'test_Varnet' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --seed 246 \
  --use-weighted-loss