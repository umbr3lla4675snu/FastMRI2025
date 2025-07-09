import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
    
if os.getcwd() + '/utils/learning/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/learning/')

from utils.learning.train_nafnet import train_nafnet_standalone

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train NAFNet for FastMRI enhancement',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Basic training parameters
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate for NAFNet')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='nafnet_enhanced', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    # Training mode
    parser.add_argument('--training-mode', type=str, choices=['standalone', 'combined'], default='standalone',
                       help='Training mode: standalone (freeze VarNet) or combined (end-to-end)')
    
    # VarNet parameters (for loading pre-trained model)
    parser.add_argument('--varnet-checkpoint-path', type=Path, 
                       default='../result/test_Varnet/checkpoints/best_model.pt',
                       help='Path to pre-trained VarNet checkpoint')
    parser.add_argument('--varnet-cascade', type=int, default=12, help='Number of VarNet cascades')
    parser.add_argument('--varnet-chans', type=int, default=18, help='Number of VarNet channels')
    parser.add_argument('--varnet-sens-chans', type=int, default=8, help='Number of VarNet sensitivity channels')
    
    # NAFNet architecture parameters
    parser.add_argument('--nafnet-width', type=int, default=32, help='NAFNet width (base channels)')
    parser.add_argument('--nafnet-middle-blks', type=int, default=8, help='Number of middle blocks in NAFNet')
    parser.add_argument('--nafnet-enc-blks', type=int, nargs='+', default=[2, 2, 4, 8], 
                       help='Number of encoder blocks per level')
    parser.add_argument('--nafnet-dec-blks', type=int, nargs='+', default=[2, 2, 2, 2], 
                       help='Number of decoder blocks per level')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--use-weighted-loss', action='store_true', help='Use index-based weighted SSIM loss')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    
    # Fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    # Setup directories
    if args.training_mode == 'standalone':
        args.exp_dir = Path('../result') / f'{args.net_name}_standalone' / 'checkpoints'
        args.val_dir = Path('../result') / f'{args.net_name}_standalone' / 'reconstructions_val'
        args.val_loss_dir = Path('../result') / f'{args.net_name}_standalone'
    else:
        args.exp_dir = Path('../result') / f'{args.net_name}_combined' / 'checkpoints'
        args.val_dir = Path('../result') / f'{args.net_name}_combined' / 'reconstructions_val'
        args.val_loss_dir = Path('../result') / f'{args.net_name}_combined'

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    args.val_loss_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training mode: {args.training_mode}")
    print(f"NAFNet architecture: width={args.nafnet_width}, middle_blks={args.nafnet_middle_blks}")
    print(f"Encoder blocks: {args.nafnet_enc_blks}")
    print(f"Decoder blocks: {args.nafnet_dec_blks}")
    
    if args.training_mode == 'standalone':
        print("Training NAFNet in standalone mode (VarNet frozen)")
        if not args.varnet_checkpoint_path.exists():
            print(f"Warning: VarNet checkpoint not found at {args.varnet_checkpoint_path}")
            print("Please train VarNet first or provide correct checkpoint path")
            sys.exit(1)
        train_nafnet_standalone(args)
    else:
        print("Training combined VarNet+NAFNet model end-to-end")
        # train_combined_model(args)
