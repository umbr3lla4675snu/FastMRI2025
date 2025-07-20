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
    parser.add_argument('--data-path', type=Path, default='/root/Data/test/', help='Directory of test data')
    parser.add_argument('--out-dir', type=Path, default='/root/result/triple_input_reconstructions', help='Output directory')
    
    # Model checkpoints
    parser.add_argument('--varnet-checkpoint-path', type=Path, 
                       default='/root/result/test_varnet/checkpoints/best_model.pt',
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
    
    # Create data loader
    data_loader = create_triple_input_data_loaders(
        data_path=args.data_path, 
        args=nafnet_checkpoint['args'], 
        shuffle=False, 
        isforward=True
    )
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    reconstructions = defaultdict(dict)
    varnet_reconstructions = defaultdict(dict)
    
    print("Starting reconstruction...")
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, _, maximum, fnames, slices, _, grappa_image, input_image = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True) if maximum != -1 else torch.tensor(1.0).cuda()
            grappa_image = grappa_image.cuda(non_blocking=True)
            input_image = input_image.cuda(non_blocking=True)
            
            # Get VarNet output
            varnet_output = varnet_model(kspace, mask)
            
            # Normalize inputs
            varnet_output_norm = varnet_output / maximum
            grappa_image_norm = grappa_image / maximum
            input_image_norm = input_image / maximum
            
            # Enhance with Triple Input NAFNet
            enhanced_output_norm = nafnet_model(
                varnet_output_norm.float(), 
                grappa_image_norm.float(), 
                input_image_norm.float()
            )
            
            # Denormalize
            enhanced_output = enhanced_output_norm * maximum
            
            for i in range(enhanced_output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = enhanced_output[i].cpu().numpy()
                varnet_reconstructions[fnames[i]][int(slices[i])] = varnet_output[i].cpu().numpy()
            
            if iter % 50 == 0:
                print(f'Processed {iter}/{len(data_loader)} batches')
    
    # Stack slices for each file
    print("Stacking slices...")
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
        varnet_reconstructions[fname] = np.stack(
            [out for _, out in sorted(varnet_reconstructions[fname].items())]
        )
    
    # Save reconstructions
    print("Saving reconstructions...")
    
    # Save enhanced reconstructions
    enhanced_dir = args.out_dir / 'enhanced'
    enhanced_dir.mkdir(exist_ok=True)
    save_reconstructions(reconstructions, enhanced_dir)
    
    # Save VarNet reconstructions for comparison
    varnet_dir = args.out_dir / 'varnet_only'
    varnet_dir.mkdir(exist_ok=True)
    save_reconstructions(varnet_reconstructions, varnet_dir)
    
    print(f"Reconstructions saved to {args.out_dir}")
    print(f"Enhanced: {enhanced_dir}")
    print(f"VarNet only: {varnet_dir}")


if __name__ == '__main__':
    args = parse()
    reconstruct_triple_input_nafnet(args)
