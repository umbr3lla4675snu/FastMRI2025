import torch
import argparse
import shutil
import os, sys
from pathlib import Path
import wandb

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
    
if os.getcwd() + '/utils/learning/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/learning/')

from utils.learning.train_triple_input_nafnet import train_triple_input_nafnet_standalone

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Triple Input NAFNet for FastMRI enhancement',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Basic training parameters
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate for Triple Input NAFNet')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='triple_input_nafnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/Data/val/', help='Directory of validation data')
    
    # VarNet parameters (for loading pre-trained model)
    parser.add_argument('--varnet-checkpoint-path', type=Path, 
                       default='/root/result/test_varnet/checkpoints/best_model.pt',
                       help='Path to pre-trained VarNet checkpoint')
    parser.add_argument('--varnet-cascade', type=int, default=1, help='Number of VarNet cascades')
    parser.add_argument('--varnet-chans', type=int, default=9, help='Number of VarNet channels')
    parser.add_argument('--varnet-sens-chans', type=int, default=4, help='Number of VarNet sensitivity channels')
    
    # Triple Input NAFNet architecture parameters
    parser.add_argument('--nafnet-width', type=int, default=32, help='Triple Input NAFNet width (base channels)')
    parser.add_argument('--nafnet-middle-blks', type=int, default=8, help='Number of middle blocks in Triple Input NAFNet')
    parser.add_argument('--nafnet-enc-blks', type=int, nargs='+', default=[2, 2, 4, 8], 
                       help='Number of encoder blocks per level')
    parser.add_argument('--nafnet-dec-blks', type=int, nargs='+', default=[2, 2, 2, 2], 
                       help='Number of decoder blocks per level')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, nargs='+', 
                    default=['kspace', 'image_input', 'image_grappa'], 
                    help='List of input keys')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--use-weighted-loss', type=lambda x: (str(x).lower() == 'true'), default=False, 
                       help='Use index-based weighted SSIM loss')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    
    # Fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    # Setup directories
    args.exp_dir = Path('/root/result') / f'{args.net_name}' / 'checkpoints'
    args.val_dir = Path('/root/result') / f'{args.net_name}' / 'reconstructions_val'
    args.val_loss_dir = Path('/root/result') / f'{args.net_name}'

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    args.val_loss_dir.mkdir(parents=True, exist_ok=True)

    print(f"Triple Input NAFNet architecture: width={args.nafnet_width}, middle_blks={args.nafnet_middle_blks}")
    print(f"Encoder blocks: {args.nafnet_enc_blks}")
    print(f"Decoder blocks: {args.nafnet_dec_blks}")
    print(f"VarNet checkpoint: {args.varnet_checkpoint_path}")
    print(f"Data paths - Train: {args.data_path_train}, Val: {args.data_path_val}")

    # Check if VarNet checkpoint exists
    if not args.varnet_checkpoint_path.exists():
        print(f"Warning: VarNet checkpoint not found at {args.varnet_checkpoint_path}")
        print("Please train VarNet first or provide correct checkpoint path")
        sys.exit(1)
        
    print("Training Triple Input NAFNet (VarNet frozen)")
    train_triple_input_nafnet_standalone(args)
