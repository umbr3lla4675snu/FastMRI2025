import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy
import wandb

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, WeightedSSIMLoss
from utils.model.varnet import VarNet
from utils.data.Augmentation.data_augment import DataAugmentor

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, slices, max_slices = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        slices = slices.clone().detach().cuda(non_blocking=True)
        max_slices = max_slices.clone().detach().cuda(non_blocking=True)

        output = model(kspace, mask)
        
        # output과 target의 shape이 다르면 최소 크기로 crop
        if output.shape != target.shape:
            min_h = min(output.shape[-2], target.shape[-2])
            min_w = min(output.shape[-1], target.shape[-1])
            output = output[..., :min_h, :min_w]
            target = target[..., :min_h, :min_w]
        
        # Use weighted loss if available
        if isinstance(loss_type, WeightedSSIMLoss):
            loss = loss_type(output, target, maximum, slices, max_slices)
        else:
            loss = loss_type(output, target, maximum)
            
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


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, _ = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # Check if gradient checkpoint is enabled
    use_gradient_checkpoint = getattr(args, 'gradient_checkpoint', False)
    if use_gradient_checkpoint:
        print("Gradient checkpointing enabled - this will save memory but may slow down training")

    # Check augmentation status
    aug_enabled = getattr(args, 'aug_on', False)
    if aug_enabled:
        print("✓ Data augmentation ENABLED")
        print(f"  - Augmentation strength: {getattr(args, 'aug_strength', 'N/A')}")
        print(f"  - Augmentation schedule: {getattr(args, 'aug_schedule', 'N/A')}")
        print(f"  - Rotation weight: {getattr(args, 'aug_weight_rotation', 'N/A')}")
        print(f"  - Scaling weight: {getattr(args, 'aug_weight_scaling', 'N/A')}")
    else:
        print("✗ Data augmentation DISABLED")

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans,
                   use_gradient_checkpoint=use_gradient_checkpoint)
    model.to(device=device)
    wandb.watch(model, log='all', log_freq=100)

    # Choose loss function based on args
    if hasattr(args, 'use_weighted_loss') and args.use_weighted_loss:
        loss_type = WeightedSSIMLoss().to(device=device)
        print("Using Weighted SSIM Loss with index-based weighting")
    else:
        loss_type = SSIMLoss().to(device=device)
        print("Using standard SSIM Loss")
        
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    # Add learning rate scheduler
    scheduler = None
    if hasattr(args, 'lr_scheduler') and args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=getattr(args, 'lr_decay_step', 10), 
            gamma=getattr(args, 'lr_decay_gamma', 0.5)
        )
        print(f"Learning rate scheduler enabled: StepLR with step_size={args.lr_decay_step}, gamma={args.lr_decay_gamma}")

    best_val_loss = 1.
    start_epoch = 0

    # Validation loader는 한 번만 생성 (augmentation 없음)
    val_args = copy.deepcopy(args)
    val_args.aug_on = False  # Validation에서는 augmentation 비활성화
    val_loader = create_data_loaders(data_path = args.data_path_val, args = val_args)
    
    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # Epoch마다 새로운 augmentation train loader 생성
        # args에 현재 epoch 정보 추가
        args.current_epoch = epoch
        train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss/num_subjects]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # Update learning rate
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        wandb.log({
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'best_val_loss': best_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_subjects': num_subjects.item(),
        }, step=epoch)

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
