"""
Training script for NAFNet to enhance VarNet reconstructions
"""
import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil

from utils.model.nafnet import NAFNetStandalone, VarNetWithNAFNet
from utils.model.varnet import VarNet
from utils.common.loss_function import SSIMLoss, WeightedSSIMLoss
from utils.common.utils import save_reconstructions, ssim_loss
from utils.data.load_data import create_data_loaders

import os


def train_nafnet_epoch(args, epoch, varnet_model, nafnet_model, data_loader, optimizer, loss_type):
    """Train NAFNet for one epoch using VarNet as a feature extractor"""
    varnet_model.eval()  # VarNet stays frozen
    nafnet_model.train()
    
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, slices, max_slices = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        slices = torch.tensor(slices).cuda(non_blocking=True)
        max_slices = torch.tensor(max_slices).cuda(non_blocking=True)

        # Get VarNet reconstruction (frozen)
        with torch.no_grad():
            varnet_output = varnet_model(kspace, mask)
        
        # Normalize VarNet output before feeding to NAFNet
        varnet_output_norm = varnet_output / maximum
        
        # Enhance with NAFNet (works on normalized data)
        enhanced_output_norm = nafnet_model(varnet_output_norm.float())
        
        # For loss calculation, we can either:
        # Option 1: Calculate loss on normalized data (more stable)
        target_norm = target / maximum
        
        # Use weighted loss if available
        if hasattr(loss_type, 'forward') and len(loss_type.forward.__code__.co_varnames) > 4:
            loss = loss_type(enhanced_output_norm.float(), target_norm.float(), torch.ones_like(maximum).float(), slices, max_slices)
        else:
            loss = loss_type(enhanced_output_norm.float(), target_norm.float(), torch.ones_like(maximum).float())
        
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


# def train_combined_epoch(args, epoch, combined_model, data_loader, optimizer, loss_type):
#     """Train the combined VarNet + NAFNet model end-to-end"""
#     combined_model.train()
    
#     start_epoch = start_iter = time.perf_counter()
#     len_loader = len(data_loader)
#     total_loss = 0.

#     for iter, data in enumerate(data_loader):
#         mask, kspace, target, maximum, _, _ = data
#         mask = mask.cuda(non_blocking=True)
#         kspace = kspace.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)
#         maximum = maximum.cuda(non_blocking=True)

#         # Forward through combined model
#         output = combined_model(kspace, mask)
        
#         # Calculate loss
#         loss = loss_type(output, target, maximum)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         if iter % args.report_interval == 0:
#             print(
#                 f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
#                 f'Iter = [{iter:4d}/{len(data_loader):4d}] '
#                 f'Loss = {loss.item():.4g} '
#                 f'Time = {time.perf_counter() - start_iter:.4f}s',
#             )
#             start_iter = time.perf_counter()
            
#     total_loss = total_loss / len_loader
#     return total_loss, time.perf_counter() - start_epoch


def validate_nafnet(args, varnet_model, nafnet_model, data_loader):
    """Validate NAFNet performance"""
    varnet_model.eval()
    nafnet_model.eval()
    
    reconstructions = defaultdict(dict)
    varnet_reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, maximum, fnames, slices, _ = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            
            # Get VarNet output
            varnet_output = varnet_model(kspace, mask)
            
            # Normalize with same method as training (use maximum, not target.max())
            varnet_output_norm = varnet_output / maximum
            
            # Enhance with NAFNet
            enhanced_output_norm = nafnet_model(varnet_output_norm.float())

            # Denormalize with same maximum used in normalization
            enhanced_output = enhanced_output_norm * maximum

            for i in range(enhanced_output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = enhanced_output[i].cpu().numpy()
                varnet_reconstructions[fnames[i]][int(slices[i])] = varnet_output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

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
    
    # Calculate metrics
    nafnet_metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    varnet_metric_loss = sum([ssim_loss(targets[fname], varnet_reconstructions[fname]) for fname in varnet_reconstructions])
    
    num_subjects = len(reconstructions)
    
    print(f"VarNet SSIM: {varnet_metric_loss/num_subjects:.4f}")
    print(f"NAFNet Enhanced SSIM: {nafnet_metric_loss/num_subjects:.4f}")
    print(f"Improvement: {(varnet_metric_loss - nafnet_metric_loss)/num_subjects:.4f}")
    
    return nafnet_metric_loss, num_subjects, reconstructions, targets, varnet_reconstructions, time.perf_counter() - start


# def validate_combined(args, combined_model, data_loader):
#     """Validate combined model performance"""
#     combined_model.eval()
    
#     reconstructions = defaultdict(dict)
#     targets = defaultdict(dict)
#     start = time.perf_counter()

#     with torch.no_grad():
#         for iter, data in enumerate(data_loader):
#             mask, kspace, target, _, fnames, slices = data
#             kspace = kspace.cuda(non_blocking=True)
#             mask = mask.cuda(non_blocking=True)
            
#             # Forward through combined model
#             output = combined_model(kspace, mask)

#             for i in range(output.shape[0]):
#                 reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
#                 targets[fnames[i]][int(slices[i])] = target[i].numpy()

#     # Stack slices for each file
#     for fname in reconstructions:
#         reconstructions[fname] = np.stack(
#             [out for _, out in sorted(reconstructions[fname].items())]
#         )
#         targets[fname] = np.stack(
#             [out for _, out in sorted(targets[fname].items())]
#         )
    
#     # Calculate metrics
#     metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
#     num_subjects = len(reconstructions)
    
#     return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_nafnet_model(args, exp_dir, epoch, varnet_model, nafnet_model, optimizer, best_val_loss, is_new_best):
    """Save NAFNet model and state"""
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'varnet_model': varnet_model.state_dict(),
            'nafnet_model': nafnet_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def save_combined_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """Save combined model"""
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'combined_model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train_nafnet_standalone(args):
    """Train NAFNet to enhance VarNet reconstructions"""
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
    
    # Initialize NAFNet
    nafnet_model = NAFNetStandalone(
        img_channel=1,
        width=args.nafnet_width,
        middle_blk_num=args.nafnet_middle_blks,
        enc_blk_nums=args.nafnet_enc_blks,
        dec_blk_nums=args.nafnet_dec_blks
    )
    nafnet_model.to(device=device)

    # Loss and optimizer for NAFNet only
    if hasattr(args, 'use_weighted_loss') and args.use_weighted_loss:
        loss_type = WeightedSSIMLoss().to(device=device)
        print("Using Weighted SSIM Loss with index-based weighting for NAFNet")
    else:
        loss_type = SSIMLoss().to(device=device)
        print("Using standard SSIM Loss for NAFNet")
    optimizer = torch.optim.AdamW(nafnet_model.parameters(), args.lr)

    start_epoch = 0
    best_val_loss = 1.

    # Data loaders
    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    val_loss_log = np.empty((0, 2))
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... NAFNet ...............')
        
        # Train NAFNet
        train_loss, train_time = train_nafnet_epoch(
            args, epoch, varnet_model, nafnet_model, train_loader, optimizer, loss_type
        )
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate_nafnet(args, varnet_model, nafnet_model, data_loader=val_loader)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        
        # Save model every epoch as there is no validation
        save_nafnet_model(args, args.exp_dir, epoch + 1, varnet_model, nafnet_model, optimizer, best_val_loss, is_new_best)
        
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )



# def train_combined_model(args):
#     """Train combined VarNet + NAFNet model end-to-end"""
#     device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
#     torch.cuda.set_device(device)
#     print('Current cuda device: ', torch.cuda.current_device())

#     # Initialize combined model
#     varnet_config = {
#         'num_cascades': args.varnet_cascade,
#         'sens_chans': args.varnet_sens_chans,
#         'sens_pools': 4,
#         'chans': args.varnet_chans,
#         'pools': 4
#     }
    
#     nafnet_config = {
#         'img_channel': 1,
#         'width': args.nafnet_width,
#         'middle_blk_num': args.nafnet_middle_blks,
#         'enc_blk_nums': args.nafnet_enc_blks,
#         'dec_blk_nums': args.nafnet_dec_blks
#     }
    
#     combined_model = VarNetWithNAFNet(varnet_config, nafnet_config)
    
#     # Load pre-trained VarNet weights if available
#     if hasattr(args, 'varnet_checkpoint_path') and args.varnet_checkpoint_path:
#         varnet_checkpoint = torch.load(args.varnet_checkpoint_path, map_location=device, weights_only=False)
#         combined_model.varnet.load_state_dict(varnet_checkpoint['model'])
#         print("Loaded pre-trained VarNet weights")
    
#     combined_model.to(device=device)

#     # Loss and optimizer
#     loss_type = SSIMLoss().to(device=device)
#     optimizer = torch.optim.Adam(combined_model.parameters(), args.lr)

#     best_val_loss = 1.
#     start_epoch = 0

#     # Data loaders
#     train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True)
    
#     for epoch in range(start_epoch, args.num_epochs):
#         print(f'Epoch #{epoch:2d} ............... Combined VarNet+NAFNet ...............')
        
#         # Train combined model
#         train_loss, train_time = train_combined_epoch(
#             args, epoch, combined_model, train_loader, optimizer, loss_type
#         )
        
#         train_loss = torch.tensor(train_loss).cuda(non_blocking=True)

#         # Save model every epoch as there is no validation
#         save_combined_model(args, args.exp_dir, epoch + 1, combined_model, optimizer, 0, False)
        
#         print(
#             f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
#             f'TrainTime = {train_time:.4f}s'
#         )
