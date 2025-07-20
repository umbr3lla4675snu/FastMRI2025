#!/usr/bin/env python3
"""
Test script for visualizing the augmentation functionality in transforms.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from matplotlib.patches import Rectangle

# Add the project root to the path
sys.path.append('/root/FastMRI2025')

from utils.data.transforms import DataTransform
from utils.data.Augmentation.data_augment import DataAugmentor
# from utils.model.fastmri import ifft2c, complex_abs


def complex_abs(data):
    """Calculate complex absolute value"""
    return torch.sqrt(data.real**2 + data.imag**2)


def ifft2c(data):
    """Apply centered 2D IFFT"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))


class TestArgs:
    """Simple class to hold test arguments"""
    def __init__(self):
        # Basic parameters
        self.max_key = 'max'
        self.target_key = 'reconstruction_esc'
        self.input_key = 'kspace'
        self.batch_size = 1
        
        # Augmentation parameters
        self.aug_on = True
        self.current_epoch = 5  # Set to a value after delay
        
        # Augmentation scheduling parameters
        self.aug_schedule = 'constant'  # Use constant for easier testing
        self.aug_delay = 0  # No delay for testing
        self.aug_strength = 1.0  # Full strength
        self.aug_exp_decay = 5.0
        self.num_epochs = 50
        
        # Interpolation parameters
        self.aug_interpolation_order = 1
        self.aug_upsample = False
        self.aug_upsample_factor = 2
        self.aug_upsample_order = 1
        
        # Transformation weights
        self.aug_weight_translation = 1.0
        self.aug_weight_rotation = 1.0
        self.aug_weight_shearing = 1.0
        self.aug_weight_scaling = 1.0
        self.aug_weight_rot90 = 0.5
        self.aug_weight_fliph = 1.0
        self.aug_weight_flipv = 1.0
        
        # Transformation limits
        self.aug_max_translation_x = 0.06
        self.aug_max_translation_y = 0.06
        self.aug_max_rotation = 10.0
        self.aug_max_shearing_x = 5.0
        self.aug_max_shearing_y = 5.0
        self.aug_max_scaling = 0.25
        
        # Additional parameters
        self.max_train_resolution = None


def create_synthetic_data():
    """Create synthetic MRI data for testing"""
    print("Creating synthetic MRI data...")
    
    # Create a simple synthetic k-space data
    height, width = 256, 256
    
    # Create a simple image pattern
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a brain-like pattern
    brain_mask = (X**2 + Y**2) < 0.8
    image = brain_mask.astype(np.float32)
    
    # Add some internal structure
    image += 0.5 * np.exp(-(X**2 + Y**2) / 0.2)
    image += 0.3 * np.exp(-((X-0.3)**2 + (Y-0.3)**2) / 0.1)
    image += 0.3 * np.exp(-((X+0.3)**2 + (Y-0.3)**2) / 0.1)
    
    # Convert to k-space
    kspace = np.fft.fft2(image)
    kspace = np.fft.fftshift(kspace)
    
    # Create mask (simulate undersampling)
    mask = np.zeros(width, dtype=np.float32)
    center_fraction = 0.08
    acceleration = 4
    
    # Add center lines
    center_start = width // 2 - int(center_fraction * width // 2)
    center_end = width // 2 + int(center_fraction * width // 2)
    mask[center_start:center_end] = 1.0
    
    # Add acceleration lines
    for i in range(0, width, acceleration):
        if i < center_start or i >= center_end:
            mask[i] = 1.0
    
    # Create target image
    target = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))
    
    # Create attributes
    attrs = {
        'max': float(np.max(target)),
        'norm': float(np.linalg.norm(target))
    }
    
    return kspace, mask, target, attrs


def load_real_data():
    """Load real MRI data from the dataset"""
    print("Loading real MRI data...")
    
    data_path = "/root/Data/train/kspace/brain_acc4_1.h5"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Available files in /root/Data/train/kspace/:")
        if os.path.exists("/root/Data/train/kspace/"):
            files = os.listdir("/root/Data/train/kspace/")
            for f in files[:10]:  # Show first 10 files
                print(f"  {f}")
        return None, None, None, None
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"Keys in {data_path}: {list(f.keys())}")
            
            # Load k-space data
            kspace = f['kspace'][:]  # Get all slices
            print(f"K-space shape: {kspace.shape}")
            
            # Use middle slice for testing
            middle_slice = kspace.shape[0] // 2
            kspace_slice = kspace[middle_slice]
            print(f"Selected slice {middle_slice}, shape: {kspace_slice.shape}")
            
            # Load mask
            if 'mask' in f.keys():
                mask = f['mask'][:]
                print(f"Mask shape: {mask.shape}")
            else:
                # Create a default mask if not present
                print("No mask found, creating default undersampling mask...")
                width = kspace_slice.shape[-1]
                mask = np.zeros(width, dtype=np.float32)
                center_fraction = 0.08
                acceleration = 4
                
                # Add center lines
                center_start = width // 2 - int(center_fraction * width // 2)
                center_end = width // 2 + int(center_fraction * width // 2)
                mask[center_start:center_end] = 1.0
                
                # Add acceleration lines
                for i in range(0, width, acceleration):
                    if i < center_start or i >= center_end:
                        mask[i] = 1.0
            
            # Load attributes
            attrs = {}
            if 'reconstruction_esc' in f.keys():
                target = f['reconstruction_esc'][middle_slice]
                attrs['max'] = float(np.max(np.abs(target)))
                print(f"Target shape: {target.shape}")
            else:
                # Create target from k-space
                target = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_slice)))
                attrs['max'] = float(np.max(target))
                print("Created target from k-space")
            
            attrs['norm'] = float(np.linalg.norm(target))
            
            print(f"Data loaded successfully!")
            print(f"K-space slice shape: {kspace_slice.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Max value: {attrs['max']}")
            
            return kspace_slice, mask, target, attrs
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


def explore_data_structure():
    """Explore the structure of available data files"""
    print("Exploring data structure...")
    
    base_path = "/root/Data/train/kspace/"
    if os.path.exists(base_path):
        files = os.listdir(base_path)
        print(f"Found {len(files)} files in {base_path}")
        
        # Try to load the first few files to understand structure
        for filename in files[:3]:
            if filename.endswith('.h5'):
                filepath = os.path.join(base_path, filename)
                try:
                    with h5py.File(filepath, 'r') as f:
                        print(f"\n{filename}:")
                        print(f"  Keys: {list(f.keys())}")
                        for key in f.keys():
                            if hasattr(f[key], 'shape'):
                                print(f"  {key} shape: {f[key].shape}")
                            else:
                                print(f"  {key}: {f[key]}")
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")
    else:
        print(f"Directory not found: {base_path}")


def visualize_results(original_data, augmented_data, save_path=None):
    """Visualize original and augmented data"""
    original_mask, original_kspace, original_target, original_max = original_data
    augmented_mask, augmented_kspace, augmented_target, augmented_max = augmented_data
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original data
    # Handle mask visualization - convert 1D mask to 2D for display
    original_mask_np = original_mask.squeeze().numpy()
    if len(original_mask_np.shape) == 1:
        # Create a 2D representation of the 1D mask
        mask_2d = np.tile(original_mask_np, (50, 1))
        axes[0, 0].imshow(mask_2d, cmap='gray', aspect='auto')
    else:
        axes[0, 0].imshow(original_mask_np, cmap='gray')
    axes[0, 0].set_title('Original Mask')
    axes[0, 0].axis('off')
    
    original_kspace_abs = torch.sqrt(original_kspace[..., 0]**2 + original_kspace[..., 1]**2)
    # Take the first slice if it's multi-coil data
    if len(original_kspace_abs.shape) > 2:
        kspace_slice = original_kspace_abs[0]  # First coil
    else:
        kspace_slice = original_kspace_abs
    axes[0, 1].imshow(torch.log(kspace_slice + 1e-8).numpy(), cmap='gray')
    axes[0, 1].set_title('Original K-space (log)')
    axes[0, 1].axis('off')
    
    # Reconstruct image from k-space for visualization
    original_kspace_complex = torch.complex(original_kspace[..., 0], original_kspace[..., 1])
    # Handle multi-coil data
    if len(original_kspace_complex.shape) > 2:
        # Take RSS (root sum of squares) across coils
        recons_per_coil = complex_abs(ifft2c(original_kspace_complex))
        original_reconstructed = torch.sqrt(torch.sum(recons_per_coil**2, dim=0))
    else:
        original_reconstructed = complex_abs(ifft2c(original_kspace_complex))
    axes[0, 2].imshow(original_reconstructed.numpy(), cmap='gray')
    axes[0, 2].set_title('Original Reconstructed')
    axes[0, 2].axis('off')
    
    if isinstance(original_target, torch.Tensor):
        target_img = original_target
        if len(target_img.shape) > 2:
            # Take RSS across coils if multi-coil
            target_img = torch.sqrt(torch.sum(target_img**2, dim=0))
        axes[0, 3].imshow(target_img.numpy(), cmap='gray')
        axes[0, 3].set_title('Original Target')
    else:
        axes[0, 3].text(0.5, 0.5, 'No Target', ha='center', va='center', transform=axes[0, 3].transAxes)
        axes[0, 3].set_title('Original Target')
    axes[0, 3].axis('off')
    
    # Augmented data
    # Handle mask visualization - convert 1D mask to 2D for display
    augmented_mask_np = augmented_mask.squeeze().numpy()
    if len(augmented_mask_np.shape) == 1:
        # Create a 2D representation of the 1D mask
        mask_2d = np.tile(augmented_mask_np, (50, 1))
        axes[1, 0].imshow(mask_2d, cmap='gray', aspect='auto')
    else:
        axes[1, 0].imshow(augmented_mask_np, cmap='gray')
    axes[1, 0].set_title('Augmented Mask')
    axes[1, 0].axis('off')
    
    augmented_kspace_abs = torch.sqrt(augmented_kspace[..., 0]**2 + augmented_kspace[..., 1]**2)
    # Take the first slice if it's multi-coil data
    if len(augmented_kspace_abs.shape) > 2:
        kspace_slice = augmented_kspace_abs[0]  # First coil
    else:
        kspace_slice = augmented_kspace_abs
    axes[1, 1].imshow(torch.log(kspace_slice + 1e-8).numpy(), cmap='gray')
    axes[1, 1].set_title('Augmented K-space (log)')
    axes[1, 1].axis('off')
    
    # Reconstruct image from k-space for visualization
    augmented_kspace_complex = torch.complex(augmented_kspace[..., 0], augmented_kspace[..., 1])
    # Handle multi-coil data
    if len(augmented_kspace_complex.shape) > 2:
        # Take RSS (root sum of squares) across coils
        recons_per_coil = complex_abs(ifft2c(augmented_kspace_complex))
        augmented_reconstructed = torch.sqrt(torch.sum(recons_per_coil**2, dim=0))
    else:
        augmented_reconstructed = complex_abs(ifft2c(augmented_kspace_complex))
    axes[1, 2].imshow(augmented_reconstructed.numpy(), cmap='gray')
    axes[1, 2].set_title('Augmented Reconstructed')
    axes[1, 2].axis('off')
    
    if isinstance(augmented_target, torch.Tensor):
        target_img = augmented_target
        if len(target_img.shape) > 2:
            # Take RSS across coils if multi-coil
            target_img = torch.sqrt(torch.sum(target_img**2, dim=0))
        axes[1, 3].imshow(target_img.numpy(), cmap='gray')
        axes[1, 3].set_title('Augmented Target')
    else:
        axes[1, 3].text(0.5, 0.5, 'No Target', ha='center', va='center', transform=axes[1, 3].transAxes)
        axes[1, 3].set_title('Augmented Target')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def test_augmentation_multiple_times(num_tests=5):
    """Test augmentation multiple times to see different random results"""
    print(f"Testing augmentation {num_tests} times...")
    
    # Create test data
    kspace, mask, target, attrs = create_synthetic_data()
    
    # Create args
    args = TestArgs()
    
    # Create transforms
    transform_with_aug = DataTransform(isforward=False, max_key=args.max_key, args=args)
    transform_without_aug = DataTransform(isforward=False, max_key=args.max_key, args=None)
    
    # Test data
    fname = "test_file.h5"
    slice_idx = 0
    max_slice_index = 0
    
    # Get original (no augmentation)
    original_result = transform_without_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    fig, axes = plt.subplots(2, num_tests, figsize=(4*num_tests, 8))
    
    for i in range(num_tests):
        # Get augmented result
        augmented_result = transform_with_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
        
        # Visualize reconstructed images
        original_kspace_complex = torch.complex(original_result[1][..., 0], original_result[1][..., 1])
        original_reconstructed = complex_abs(ifft2c(original_kspace_complex))
        
        augmented_kspace_complex = torch.complex(augmented_result[1][..., 0], augmented_result[1][..., 1])
        augmented_reconstructed = complex_abs(ifft2c(augmented_kspace_complex))
        
        # Show original (only in first column)
        if i == 0:
            axes[0, i].imshow(original_reconstructed.squeeze().numpy(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
        else:
            axes[0, i].imshow(original_reconstructed.squeeze().numpy(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
        
        # Show augmented
        axes[1, i].imshow(augmented_reconstructed.squeeze().numpy(), cmap='gray')
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/root/FastMRI2025/augmentation_multiple_tests.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Multiple augmentation tests completed!")


def test_augmentation_multiple_times_real_data(num_tests, kspace, mask, target, attrs):
    """Test augmentation multiple times with real data"""
    print(f"Testing augmentation {num_tests} times with real data...")
    
    # Create args
    args = TestArgs()
    
    # Create transforms
    transform_with_aug = DataTransform(isforward=False, max_key=args.max_key, args=args)
    transform_without_aug = DataTransform(isforward=False, max_key=args.max_key, args=None)
    
    # Test data
    fname = "brain_acc4_1.h5"
    slice_idx = 0
    max_slice_index = 0
    
    # Get original (no augmentation)
    original_result = transform_without_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    fig, axes = plt.subplots(2, num_tests, figsize=(4*num_tests, 8))
    
    for i in range(num_tests):
        # Get augmented result
        augmented_result = transform_with_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
        
        # Visualize reconstructed images
        original_kspace_complex = torch.complex(original_result[1][..., 0], original_result[1][..., 1])
        # Handle multi-coil data for visualization
        if len(original_kspace_complex.shape) > 2:
            recons_per_coil = complex_abs(ifft2c(original_kspace_complex))
            original_reconstructed = torch.sqrt(torch.sum(recons_per_coil**2, dim=0))
        else:
            original_reconstructed = complex_abs(ifft2c(original_kspace_complex))
        
        augmented_kspace_complex = torch.complex(augmented_result[1][..., 0], augmented_result[1][..., 1])
        # Handle multi-coil data for visualization
        if len(augmented_kspace_complex.shape) > 2:
            recons_per_coil = complex_abs(ifft2c(augmented_kspace_complex))
            augmented_reconstructed = torch.sqrt(torch.sum(recons_per_coil**2, dim=0))
        else:
            augmented_reconstructed = complex_abs(ifft2c(augmented_kspace_complex))
        
        # Show original
        axes[0, i].imshow(original_reconstructed.numpy(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Show augmented
        axes[1, i].imshow(augmented_reconstructed.numpy(), cmap='gray')
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/root/FastMRI2025/real_data_augmentation_multiple_tests.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Multiple augmentation tests with real data completed!")


def test_transforms_with_real_data():
    """Main test function using real data"""
    print("Starting augmentation test with real data...")
    
    # Load real data
    kspace, mask, target, attrs = load_real_data()
    
    if kspace is None:
        print("Failed to load real data, falling back to synthetic data...")
        kspace, mask, target, attrs = create_synthetic_data()
    
    # Create args
    args = TestArgs()
    
    # Create transforms
    transform_with_aug = DataTransform(isforward=False, max_key=args.max_key, args=args)
    transform_without_aug = DataTransform(isforward=False, max_key=args.max_key, args=None)
    
    # Test data
    fname = "brain_acc4_1.h5"
    slice_idx = 0
    max_slice_index = 0
    
    print("Testing without augmentation...")
    original_result = transform_without_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    print("Testing with augmentation...")
    augmented_result = transform_with_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    print("Results:")
    print(f"Original - Mask shape: {original_result[0].shape}, K-space shape: {original_result[1].shape}")
    print(f"Augmented - Mask shape: {augmented_result[0].shape}, K-space shape: {augmented_result[1].shape}")
    
    # Check if augmentation was applied
    if hasattr(transform_with_aug, 'augmentor') and transform_with_aug.augmentor is not None:
        print("Augmentation is enabled!")
    else:
        print("Augmentation is disabled!")
    
    # Visualize results
    visualize_results(original_result[:4], augmented_result[:4], '/root/FastMRI2025/real_data_augmentation_test.png')
    
    # Test multiple times with real data
    test_augmentation_multiple_times_real_data(5, kspace, mask, target, attrs)
    
    print("Test with real data completed!")


def test_transforms():
    """Main test function"""
    print("Starting augmentation test...")
    
    # Create test data
    kspace, mask, target, attrs = create_synthetic_data()
    
    # Create args
    args = TestArgs()
    
    # Create transforms
    transform_with_aug = DataTransform(isforward=False, max_key=args.max_key, args=args)
    transform_without_aug = DataTransform(isforward=False, max_key=args.max_key, args=None)
    
    # Test data
    fname = "test_file.h5"
    slice_idx = 0
    max_slice_index = 0
    
    print("Testing without augmentation...")
    original_result = transform_without_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    print("Testing with augmentation...")
    augmented_result = transform_with_aug(mask, kspace, target, attrs, fname, slice_idx, max_slice_index)
    
    print("Results:")
    print(f"Original - Mask shape: {original_result[0].shape}, K-space shape: {original_result[1].shape}")
    print(f"Augmented - Mask shape: {augmented_result[0].shape}, K-space shape: {augmented_result[1].shape}")
    
    # Check if augmentation was applied
    if hasattr(transform_with_aug, 'augmentor') and transform_with_aug.augmentor is not None:
        print("Augmentation is enabled!")
    else:
        print("Augmentation is disabled!")
    
    # Visualize results
    visualize_results(original_result[:4], augmented_result[:4], '/root/FastMRI2025/augmentation_test_result.png')
    
    # Test multiple times
    test_augmentation_multiple_times(5)
    
    print("Test completed!")


if __name__ == "__main__":
    # First explore data structure
    explore_data_structure()
    
    # Then run test with real data
    test_transforms_with_real_data()
    
    # Also run synthetic data test for comparison
    print("\n" + "="*50)
    print("Running synthetic data test for comparison...")
    print("="*50)
    test_transforms()
