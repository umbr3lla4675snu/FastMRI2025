"""
Reconstruction script for Triple Input NAFNet
Generates enhanced reconstructions using VarNet + GRAPPA + Input images
"""

import torch
import argparse
import os, sys
from pathlib import Path
import numpy as np
import h5py
from collections import defaultdict

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.model.triple_input_nafnet import TripleInputNAFNetStandalone
from utils.model.varnet import VarNet
from utils.data.triple_input_load_data import create_triple_input_data_loaders
from utils.common.utils import save_reconstructions
import utils.model.fastmri


def parse():
    parser = argparse.ArgumentParser(description='Reconstruct using Triple Input NAFNet',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--data-path', type=Path, default='/root/Data/leaderboard', help='Directory of test data (should contain acc4 and acc8 folders)')
    parser.add_argument('--out-dir', type=Path, default='/root/result/triple_input_reconstructions', help='Output directory')
    
    # Model checkpoints
    parser.add_argument('--varnet-checkpoint-path', type=Path, 
                       default='/root/result/best_Varnet/checkpoints/best_model.pt',
                       help='Path to pre-trained VarNet checkpoint')
    parser.add_argument('--nafnet-checkpoint-path', type=Path, 
                       default='/root/result/triple_input_nafnet/checkpoints/best_model.pt',
                       help='Path to trained Triple Input NAFNet checkpoint')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    
    args = parser.parse_args()
    return args


def process_acceleration(args, varnet_model, nafnet_model, nafnet_checkpoint, acc_type):
    """Process a single acceleration type (acc4 or acc8)"""
    device = varnet_model.device if hasattr(varnet_model, 'device') else next(varnet_model.parameters()).device
    
    # Set data path for this acceleration type
    current_data_path = args.data_path / acc_type
    print(f"Processing {acc_type} data from: {current_data_path}")
    
    # Create data loader for this acceleration type
    data_loader = create_triple_input_data_loaders(
        data_path=current_data_path, 
        args=nafnet_checkpoint['args'], 
        shuffle=False, 
        isforward=True
    )
    
    reconstructions = defaultdict(dict)
    varnet_reconstructions = defaultdict(dict)
    
    print(f"Starting reconstruction for {acc_type}...")
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, maximum, fnames, slices, _, grappa_image, input_image = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            grappa_image = grappa_image.cuda(non_blocking=True)
            input_image = input_image.cuda(non_blocking=True)
            
            # Get VarNet output
            varnet_output = varnet_model(kspace, mask)
            
            # Scale VarNet output to match the scaled grappa/input images
            scale_factor = 1000.0
            varnet_output = varnet_output * scale_factor
            
            # Calculate proper maximum from scaled values
            varnet_max = varnet_output.max()
            grappa_max = grappa_image.max() 
            input_max = input_image.max()
            
            # Use the maximum among all three as normalization factor
            calculated_maximum = max(varnet_max.item(), grappa_max.item(), input_max.item(), maximum.item())
            
            # Use calculated maximum for normalization
            norm_maximum = torch.tensor(calculated_maximum).cuda()
            
            # Normalize inputs
            varnet_output_norm = varnet_output / norm_maximum
            grappa_image_norm = grappa_image / norm_maximum
            input_image_norm = input_image / norm_maximum
            
            # Enhance with Triple Input NAFNet
            enhanced_output_norm = nafnet_model(
                varnet_output_norm.float(), 
                grappa_image_norm.float(), 
                input_image_norm.float()
            )
            
            # Denormalize using the same maximum
            enhanced_output = enhanced_output_norm * norm_maximum
            
            for i in range(enhanced_output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = enhanced_output[i].cpu().numpy()
                varnet_reconstructions[fnames[i]][int(slices[i])] = varnet_output[i].cpu().numpy()
            
            if iter % 50 == 0:
                print(f'Processed {iter}/{len(data_loader)} batches for {acc_type}')
    
    # Stack slices for each file without additional scaling
    print(f"Stacking slices for {acc_type}...")
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
        varnet_reconstructions[fname] = np.stack(
            [out for _, out in sorted(varnet_reconstructions[fname].items())]
        )
    
    return reconstructions, varnet_reconstructions


def reconstruct_triple_input_nafnet(args):
    """Reconstruct using Triple Input NAFNet"""
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    # Load VarNet
    print("Loading VarNet...")
    varnet_checkpoint = torch.load(args.varnet_checkpoint_path, map_location=device, weights_only=False)
    varnet_model = VarNet(
        num_cascades=varnet_checkpoint['args'].cascade,
        chans=varnet_checkpoint['args'].chans,
        sens_chans=varnet_checkpoint['args'].sens_chans
    )
    varnet_model.load_state_dict(varnet_checkpoint['model'])
    varnet_model.to(device=device)
    varnet_model.eval()
    
    # Load Triple Input NAFNet
    print("Loading Triple Input NAFNet...")
    nafnet_checkpoint = torch.load(args.nafnet_checkpoint_path, map_location=device, weights_only=False)
    nafnet_model = TripleInputNAFNetStandalone(
        img_channel=1,
        width=nafnet_checkpoint['args'].nafnet_width,
        middle_blk_num=nafnet_checkpoint['args'].nafnet_middle_blks,
        enc_blk_nums=nafnet_checkpoint['args'].nafnet_enc_blks,
        dec_blk_nums=nafnet_checkpoint['args'].nafnet_dec_blks
    )
    nafnet_model.load_state_dict(nafnet_checkpoint['nafnet_model'])
    nafnet_model.to(device=device)
    nafnet_model.eval()
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process both acc4 and acc8
    for acc_type in ['acc4', 'acc8']:
        print(f"\n{'='*50}")
        print(f"Processing {acc_type}")
        print(f"{'='*50}")
        
        reconstructions, varnet_reconstructions = process_acceleration(
            args, varnet_model, nafnet_model, nafnet_checkpoint, acc_type
        )
        
        # Save reconstructions for this acceleration type
        print(f"Saving {acc_type} reconstructions...")
        
        # Save enhanced reconstructions
        enhanced_dir = args.out_dir / acc_type / 'enhanced'
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        save_reconstructions(reconstructions, enhanced_dir)
        
        # Save VarNet reconstructions for comparison
        varnet_dir = args.out_dir / acc_type / 'varnet_only'
        varnet_dir.mkdir(parents=True, exist_ok=True)
        save_reconstructions(varnet_reconstructions, varnet_dir)
        
        print(f"{acc_type} reconstructions saved!")
        print(f"Enhanced: {enhanced_dir}")
        print(f"VarNet only: {varnet_dir}")
    
    print(f"\nAll reconstructions completed and saved to {args.out_dir}")


if __name__ == '__main__':
    args = parse()
    reconstruct_triple_input_nafnet(args)
