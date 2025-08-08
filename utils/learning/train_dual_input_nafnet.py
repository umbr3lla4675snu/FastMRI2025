"""
Training script for Dual Input NAFNet
Trains NAFNet to enhance VarNet reconstructions using GRAPPA images
"""
import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
import os
import sys

# Add model path
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.model.dual_input_nafnet import DualInputNAFNetStandalone, VarNetWithTripleInputNAFNet
from utils.model.varnet import VarNet
from utils.common.loss_function import SSIMLoss
from utils.common.utils import save_reconstructions, ssim_loss
from utils.data.dual_input_load_data import create_dual_input_data_loaders
import fastmri
from utils.common.utils import center_crop


def train_dual_input_nafnet_epoch(args, epoch, varnet_model, nafnet_model, data_loader, optimizer, loss_type):
    """Train Dual Input NAFNet for one epoch using VarNet as a feature extractor"""
    varnet_model.eval()  # VarNet stays frozen
    nafnet_model.train()
    
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, slices, max_slices, grappa_image = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        grappa_image = grappa_image.cuda(non_blocking=True)
        slices = torch.tensor(slices).cuda(non_blocking=True)
        max_slices = torch.tensor(max_slices).cuda(non_blocking=True)
        
        # Get VarNet reconstruction (frozen)
        with torch.no_grad():
            varnet_output = varnet_model(kspace, mask)
        
        # Normalize all inputs before feeding to NAFNet
        varnet_output_norm = varnet_output / maximum
        grappa_image_norm = grappa_image / maximum
        
        # Enhance with Dual Input NAFNet (only VarNet + GRAPPA)
        enhanced_output_norm = nafnet_model(varnet_output_norm.float(), grappa_image_norm.float())
        
        # Calculate loss on normalized data
        target_norm = target / maximum
        
        # 올바른 data_range 제공
        batch_size = enhanced_output_norm.shape[0]
        data_range = torch.ones(batch_size, device=enhanced_output_norm.device, 
                               dtype=enhanced_output_norm.dtype)
        
        # Use standard SSIM loss with proper data_range
        loss = loss_type(enhanced_output_norm.float(), target_norm.float(), data_range)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
            
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate_dual_input_nafnet(args, varnet_model, nafnet_model, data_loader):
    """Validate Dual Input NAFNet performance"""
    varnet_model.eval()
    nafnet_model.eval()
    
    reconstructions = defaultdict(dict)
    varnet_reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    
    # Loss function for validation
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    loss_type = SSIMLoss().to(device=device)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, fnames, slices, _, grappa_image = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            grappa_image = grappa_image.cuda(non_blocking=True)
            
            # Get VarNet output
            varnet_output = varnet_model(kspace, mask)
            
            # Normalize with same method as training
            varnet_output_norm = varnet_output / maximum
            grappa_image_norm = grappa_image / maximum
            
            # Enhance with Dual Input NAFNet (only VarNet + GRAPPA)
            enhanced_output_norm = nafnet_model(varnet_output_norm.float(), grappa_image_norm.float())

            # Calculate validation loss
            target_norm = target / maximum
            
            # 올바른 data_range 제공 (training과 동일)
            batch_size = enhanced_output_norm.shape[0]
            data_range = torch.ones(batch_size, device=enhanced_output_norm.device, 
                                   dtype=enhanced_output_norm.dtype)
            
            # Validation loss 계산
            val_loss = loss_type(enhanced_output_norm.float(), target_norm.float(), data_range)
            total_loss += val_loss.item() * batch_size
            total_samples += batch_size

            # Denormalize for saving
            enhanced_output = enhanced_output_norm * maximum

            for i in range(enhanced_output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = enhanced_output[i].cpu().numpy()
                varnet_reconstructions[fnames[i]][int(slices[i])] = varnet_output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

    # Stack slices for each file
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
        varnet_reconstructions[fname] = np.stack(
            [out for _, out in sorted(varnet_reconstructions[fname].items())]
        )
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    # Calculate metrics using utils function
    nafnet_metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    varnet_metric_loss = sum([ssim_loss(targets[fname], varnet_reconstructions[fname]) for fname in varnet_reconstructions])
    
    num_subjects = len(reconstructions)
    
    print(f"VarNet SSIM: {varnet_metric_loss/num_subjects:.4f}")
    print(f"Dual Input NAFNet Enhanced SSIM: {nafnet_metric_loss/num_subjects:.4f}")
    print(f"Improvement: {(varnet_metric_loss - nafnet_metric_loss)/num_subjects:.4f}")
    
    # Return SSIM-based validation loss (calculated with proper data_range)
    validation_loss = total_loss / total_samples if total_samples > 0 else nafnet_metric_loss
    
    return validation_loss, num_subjects, reconstructions, targets, varnet_reconstructions, time.perf_counter() - start


def save_dual_input_nafnet_model(args, exp_dir, epoch, varnet_model, nafnet_model, optimizer, scheduler, best_val_loss, is_new_best):
    """Save Dual Input NAFNet model and state"""
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'varnet_model': varnet_model.state_dict(),
            'nafnet_model': nafnet_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir,
            'lr': optimizer.param_groups[0]['lr']  # Save current learning rate
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")


def train_dual_input_nafnet_standalone(args):
    """Train Dual Input NAFNet to enhance VarNet reconstructions"""
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    # Load VarNet weights
    varnet_checkpoint = torch.load(args.varnet_checkpoint_path, map_location=device, weights_only=False)
    varnet_model = VarNet(
        num_cascades=varnet_checkpoint['args'].cascade,
        chans=varnet_checkpoint['args'].chans,
        sens_chans=varnet_checkpoint['args'].sens_chans
    )
    varnet_model.load_state_dict(varnet_checkpoint['model'])
    varnet_model.to(device=device)
    varnet_model.eval()  # Freeze VarNet
    
    # Initialize Dual Input NAFNet
    nafnet_model = DualInputNAFNetStandalone(
        img_channel=1,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blks,
        enc_blk_nums=args.nafnet_enc_blks,
        dec_blk_nums=args.nafnet_dec_blks
    )
    nafnet_model.to(device=device)

    # Use standard SSIM Loss only
    loss_type = SSIMLoss().to(device=device)
    print("Using standard SSIM Loss for Dual Input NAFNet")
    optimizer = torch.optim.AdamW(nafnet_model.parameters(), args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    best_val_loss = 1.

    # Check for existing checkpoint and resume if available
    checkpoint_path = args.exp_dir / 'model.pt'
    if checkpoint_path.exists():
        print(f"Found existing checkpoint at {checkpoint_path}")
        print("Resuming training from checkpoint...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load NAFNet model state
        nafnet_model.load_state_dict(checkpoint['nafnet_model'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state if available
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # If no scheduler state, adjust scheduler to current epoch
            for _ in range(checkpoint['epoch']):
                scheduler.step()
        
        # Load training state
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Load validation loss log if exists
        val_loss_log_path = os.path.join(args.val_loss_dir, "val_loss_log.npy")
        if os.path.exists(val_loss_log_path):
            val_loss_log = np.load(val_loss_log_path)
            print(f"Loaded validation loss log with {len(val_loss_log)} entries")
        else:
            val_loss_log = np.empty((0, 2))
    else:
        print("No existing checkpoint found. Starting training from scratch...")
        val_loss_log = np.empty((0, 2))

    # Data loaders
    train_loader = create_dual_input_data_loaders(data_path=args.data_path_train, args=args, shuffle=True)
    val_loader = create_dual_input_data_loaders(data_path=args.data_path_val, args=args)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... Dual Input NAFNet ...............')
        
        # Train NAFNet
        train_loss, train_time = train_dual_input_nafnet_epoch(
            args, epoch, varnet_model, nafnet_model, train_loader, optimizer, loss_type
        )
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_dual_input_nafnet(args, varnet_model, nafnet_model, data_loader=val_loader)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        
        # Save model every epoch
        save_dual_input_nafnet_model(args, args.exp_dir, epoch + 1, varnet_model, nafnet_model, optimizer, scheduler, best_val_loss, is_new_best)
        
        # Step the scheduler
        scheduler.step()
        
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s '
            f'LR = {optimizer.param_groups[0]["lr"]:.2e}',
        )
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
