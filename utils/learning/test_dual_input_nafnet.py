import numpy as np
import torch
import h5py
import tqdm
import os
import matplotlib.pyplot as plt

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.dual_input_load_data import create_dual_input_data_loaders
from utils.model.varnet import VarNet
from utils.model.dual_input_nafnet import DualInputNAFNetStandalone


def visualize_dual_input_reconstruction(varnet_output, image_grappa, nafnet_output, fname, slice_idx, save_dir):
    """
    Visualize the dual input reconstruction process
    
    Args:
        varnet_output: VarNet reconstruction result
        image_grappa: GRAPPA reconstruction 
        nafnet_output: Final NAFNet enhanced result
        fname: Filename
        slice_idx: Slice index
        save_dir: Directory to save visualization
    """
    # Convert to numpy and squeeze
    varnet_img = varnet_output.cpu().numpy().squeeze()
    grappa_img = image_grappa.cpu().numpy().squeeze() 
    nafnet_img = nafnet_output.cpu().numpy().squeeze()
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Dual Input NAFNet Reconstruction\nFile: {fname}, Slice: {slice_idx}', fontsize=16)
    
    # GRAPPA Reconstruction (left)
    im1 = axes[0].imshow(grappa_img, cmap='gray', vmin=0, vmax=grappa_img.max())
    axes[0].set_title('GRAPPA Reconstruction', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # VarNet Reconstruction (center)
    im2 = axes[1].imshow(varnet_img, cmap='gray', vmin=0, vmax=varnet_img.max())
    axes[1].set_title('VarNet Reconstruction', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # NAFNet Enhanced Result (right)
    im3 = axes[2].imshow(nafnet_img, cmap='gray', vmin=0, vmax=nafnet_img.max())
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'dual_input_visualization_{fname}_{slice_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_path}")


def test(args, varnet_model, nafnet_model, data_loader):
    varnet_model.eval()
    nafnet_model.eval()
    reconstructions = defaultdict(dict)
    
    # Create visualization directory
    viz_dir = os.path.join(args.forward_dir, 'visualizations')
    
    # Counter for saving first sample only
    sample_count = 0
    
    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _, image_grappa) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            image_grappa = image_grappa.cuda(non_blocking=True)

            # Forward pass through VarNet
            varnet_output = varnet_model(kspace, mask)

            # Forward pass through Dual Input NAFNet (only VarNet + GRAPPA)
            nafnet_output = nafnet_model(varnet_output, image_grappa)
            
            # Save visualization for first sample of first batch only
            if sample_count == 0:
                fname = fnames[0]
                slice_idx = int(slices[0])
                
                visualize_dual_input_reconstruction(
                    varnet_output[0], 
                    image_grappa[0], 
                    nafnet_output[0],
                    fname,
                    slice_idx,
                    viz_dir
                )
                sample_count += 1

            for i in range(nafnet_output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = nafnet_output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None



def forward(args):
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
    
    # Load Dual Input NAFNet
    print("Loading Dual Input NAFNet...")
    nafnet_checkpoint = torch.load(args.nafnet_checkpoint_path, map_location=device, weights_only=False)
    nafnet_model = DualInputNAFNetStandalone(
        img_channel=1,
        width=nafnet_checkpoint['args'].nafnet_width,
        middle_blk_num=nafnet_checkpoint['args'].nafnet_middle_blks,
        enc_blk_nums=nafnet_checkpoint['args'].nafnet_enc_blks,
        dec_blk_nums=nafnet_checkpoint['args'].nafnet_dec_blks
    )
    nafnet_model.load_state_dict(nafnet_checkpoint['nafnet_model'])
    nafnet_model.to(device=device)

    forward_loader = create_dual_input_data_loaders(
        data_path=args.data_path,
        args=args,
        isforward=True
    )
    reconstructions, inputs = test(args, varnet_model, nafnet_model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)
    print("Dual Input NAFNet reconstruction completed!")
    