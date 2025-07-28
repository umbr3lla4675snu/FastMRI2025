import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

def save_augmentation_debug_images(original_kspace, augmented_kspace, original_target, augmented_target, fname, slice_num):
    """
    Save debug images to visualize augmentation effects
    """
    try:
        debug_dir = "/root/FastMRI2025/debug_augmentation"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Augmentation Debug - {fname}_slice_{slice_num}', fontsize=14)
        
        # Convert complex kspace to magnitude for visualization
        def process_kspace_for_visualization(kspace):
            if len(kspace.shape) == 4 and kspace.shape[-1] == 2:
                # Complex format [C, H, W, 2] or [H, W, 2] -> convert to complex then magnitude
                kspace_complex = torch.view_as_complex(kspace)
                kspace_mag = torch.abs(kspace_complex)
            else:
                kspace_mag = torch.abs(kspace)
            
            # Handle multi-coil data by taking RSS (Root Sum of Squares)
            if len(kspace_mag.shape) == 3:  # Multi-coil: [C, H, W]
                kspace_mag = torch.sqrt(torch.sum(kspace_mag**2, dim=0))
            elif len(kspace_mag.shape) > 3:
                # Flatten extra dimensions and take RSS
                original_shape = kspace_mag.shape
                kspace_mag = kspace_mag.view(-1, *original_shape[-2:])
                kspace_mag = torch.sqrt(torch.sum(kspace_mag**2, dim=0))
            
            return kspace_mag.squeeze()
        
        original_kspace_mag = process_kspace_for_visualization(original_kspace)
        augmented_kspace_mag = process_kspace_for_visualization(augmented_kspace)
        
        # Original k-space (log scale for better visualization)
        axes[0, 0].imshow(torch.log(original_kspace_mag + 1e-9).numpy(), cmap='gray')
        axes[0, 0].set_title('Original K-space (log)')
        axes[0, 0].axis('off')
        
        # Augmented k-space (log scale)
        axes[0, 1].imshow(torch.log(augmented_kspace_mag + 1e-9).numpy(), cmap='gray')
        axes[0, 1].set_title('Augmented K-space (log)')
        axes[0, 1].axis('off')
        
        # K-space difference
        if original_kspace_mag.shape == augmented_kspace_mag.shape:
            kspace_diff = torch.abs(augmented_kspace_mag - original_kspace_mag)
            axes[0, 2].imshow(kspace_diff.numpy(), cmap='hot')
            axes[0, 2].set_title('K-space Difference')
        else:
            axes[0, 2].text(0.5, 0.5, f'Size mismatch:\nOrig: {original_kspace_mag.shape}\nAug: {augmented_kspace_mag.shape}', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('K-space Size Comparison')
        axes[0, 2].axis('off')
        
        # K-space shapes info
        axes[0, 3].text(0.1, 0.7, f'Original K-space shape:\n{original_kspace.shape}', 
                       transform=axes[0, 3].transAxes, fontsize=10)
        axes[0, 3].text(0.1, 0.3, f'Augmented K-space shape:\n{augmented_kspace.shape}', 
                       transform=axes[0, 3].transAxes, fontsize=10)
        axes[0, 3].set_title('Shape Information')
        axes[0, 3].axis('off')
        
        # Helper function to process target images for visualization
        def process_target_for_visualization(target):
            if target is None:
                return None
            
            target_processed = target.clone()
            
            # Handle multi-dimensional targets
            if len(target_processed.shape) > 2:
                # Take the magnitude if complex
                if target_processed.shape[-1] == 2:  # Complex format
                    target_processed = torch.view_as_complex(target_processed)
                    target_processed = torch.abs(target_processed)
                
                # If still multi-dimensional, take the first slice or RSS
                while len(target_processed.shape) > 2:
                    if target_processed.shape[0] > 1:  # Multi-coil or batch
                        target_processed = torch.sqrt(torch.sum(target_processed**2, dim=0))
                    else:
                        target_processed = target_processed.squeeze(0)
            
            return target_processed.squeeze()
        
        # Original target image
        processed_original_target = process_target_for_visualization(original_target)
        if processed_original_target is not None:
            axes[1, 0].imshow(processed_original_target.numpy(), cmap='gray')
            axes[1, 0].set_title('Original Target')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Original Target', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Original Target')
        axes[1, 0].axis('off')
        
        # Augmented target image
        processed_augmented_target = process_target_for_visualization(augmented_target)
        if processed_augmented_target is not None:
            axes[1, 1].imshow(processed_augmented_target.numpy(), cmap='gray')
            axes[1, 1].set_title('Augmented Target')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Augmented Target', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Augmented Target')
        axes[1, 1].axis('off')
        
        # Target difference
        if processed_original_target is not None and processed_augmented_target is not None:
            if processed_original_target.shape == processed_augmented_target.shape:
                target_diff = torch.abs(processed_augmented_target - processed_original_target)
                axes[1, 2].imshow(target_diff.numpy(), cmap='hot')
                axes[1, 2].set_title('Target Difference')
            else:
                axes[1, 2].text(0.5, 0.6, f'Size mismatch detected!', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, 
                               fontweight='bold', fontsize=12, color='red')
                axes[1, 2].text(0.5, 0.4, f'Original: {processed_original_target.shape}', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10)
                axes[1, 2].text(0.5, 0.3, f'Augmented: {processed_augmented_target.shape}', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10)
                axes[1, 2].text(0.5, 0.1, f'Reason: Augmentation changed\nimage size during processing', 
                               ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=9, style='italic')
                axes[1, 2].set_title('Target Size Mismatch')
        else:
            axes[1, 2].text(0.5, 0.5, 'Cannot compute difference', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Target Difference')
        axes[1, 2].axis('off')
        
        # Target shapes info
        orig_target_shape = original_target.shape if original_target is not None else "None"
        aug_target_shape = augmented_target.shape if augmented_target is not None else "None"
        axes[1, 3].text(0.1, 0.7, f'Original Target shape:\n{orig_target_shape}', 
                       transform=axes[1, 3].transAxes, fontsize=10)
        axes[1, 3].text(0.1, 0.3, f'Augmented Target shape:\n{aug_target_shape}', 
                       transform=axes[1, 3].transAxes, fontsize=10)
        axes[1, 3].set_title('Target Shape Information')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(debug_dir, f'augmentation_debug_{fname}_slice_{slice_num}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug image saved: {save_path}")
        
    except Exception as e:
        print(f"Error in save_augmentation_debug_images: {e}")
        print(f"Original kspace shape: {original_kspace.shape if original_kspace is not None else 'None'}")
        print(f"Augmented kspace shape: {augmented_kspace.shape if augmented_kspace is not None else 'None'}")
        print(f"Original target shape: {original_target.shape if original_target is not None else 'None'}")
        print(f"Augmented target shape: {augmented_target.shape if augmented_target is not None else 'None'}")
        # Clean up the figure if it was created
        plt.close('all')

class DataTransform:
    def __init__(self, isforward, max_key, args=None, debug_augmentation=False, debug_max_samples=10):
        self.isforward = isforward
        self.max_key = max_key
        self.args = args
        self.debug_augmentation = debug_augmentation
        self.debug_max_samples = debug_max_samples
        self.debug_count = 0
        
        # Initialize DataAugmentor if augmentation is enabled and not in forward mode
        if not isforward and args is not None and hasattr(args, 'aug_on') and args.aug_on:
            from utils.data.Augmentation.data_augment import DataAugmentor
            # Create a function that returns current epoch
            def get_current_epoch():
                return getattr(args, 'current_epoch', 0)
            
            self.augmentor = DataAugmentor(args, get_current_epoch)
        else:
            self.augmentor = None
    
    def __call__(self, mask, input, target, attrs, fname, slice, max_slice_index):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        # Store original mask for later use
        original_mask = mask.copy()
        original_freq_dim = mask.shape[0]
        
        full_kspace = to_tensor(input)
        full_kspace = torch.stack((full_kspace.real, full_kspace.imag), dim=-1)
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Apply augmentation if available and not in forward mode
        if not self.isforward and self.augmentor is not None:
            # Get target size from original target, but ensure it's a consistent size
            if target is not None and hasattr(target, 'shape') and len(target.shape) >= 2:
                target_size = [target.shape[-2], target.shape[-1]]  # Use last two dimensions
            else:
                target_size = [384, 384]  # Default target size
                
            # Store original target size for debugging
            original_target_size = target_size.copy()
                
            augmented_kspace, augmented_target, augmentation_applied = self.augmentor(full_kspace, target_size)
            
            # Only apply augmentation if it was actually applied and target is valid
            if augmentation_applied and augmented_target is not None:
                # Save debug images for visualization (limited samples)
                if self.debug_augmentation and self.debug_count < self.debug_max_samples:
                    try:
                        save_augmentation_debug_images(
                            original_kspace=full_kspace,
                            augmented_kspace=augmented_kspace,
                            original_target=target,
                            augmented_target=augmented_target,
                            fname=fname,
                            slice_num=slice
                        )
                        self.debug_count += 1
                        print(f"Saved debug image {self.debug_count}/{self.debug_max_samples} for {fname}_slice_{slice}")
                        print(f"  Original target size requested: {original_target_size}")
                        print(f"  Augmented target size received: {list(augmented_target.shape) if hasattr(augmented_target, 'shape') else 'Unknown'}")
                    except Exception as e:
                        print(f"Failed to save debug image for {fname}_slice_{slice}: {e}")
                
                target = augmented_target
                # Update kspace to the augmented version
                kspace = augmented_kspace
                
                # Adjust mask to match the new kspace dimensions
                new_freq_dim = augmented_kspace.shape[-2]
                # print(f"Augmented kspace shape: {augmented_kspace.shape}, New frequency dimension: {new_freq_dim}")
                # print(f"Original kspace shape: {full_kspace.shape}, Original frequency dimension: {original_freq_dim}")
                # print(f"Augmented target shape: {augmented_target.shape} (if applicable)")
                # print(f"Original target shape: {target.shape} (if applicable)")
                # print(f"Original mask shape: {original_mask.shape}, Original frequency dimension: {original_freq_dim}")
                if new_freq_dim != original_freq_dim:
                    # Create a new mask that matches the augmented kspace size
                    if new_freq_dim > original_freq_dim:
                        # Pad the mask (add zeros for additional frequency components)
                        pad_size = (new_freq_dim - original_freq_dim) // 2
                        adjusted_mask = np.zeros(new_freq_dim, dtype=original_mask.dtype)
                        adjusted_mask[pad_size:pad_size + original_freq_dim] = original_mask.squeeze()
                    else:
                        # Crop the mask from center
                        start_idx = (original_freq_dim - new_freq_dim) // 2
                        adjusted_mask = original_mask.squeeze()[start_idx:start_idx + new_freq_dim]
                    
                    # Apply the adjusted mask to the augmented kspace
                    mask_tensor = torch.from_numpy(adjusted_mask.reshape(1, 1, new_freq_dim, 1).astype(np.float32))
                    kspace = kspace * mask_tensor
                    final_mask = adjusted_mask
                else:
                    # Mask dimensions match, apply original mask
                    mask_tensor = torch.from_numpy(original_mask.reshape(1, 1, original_freq_dim, 1).astype(np.float32))
                    kspace = kspace * mask_tensor
                    final_mask = original_mask
            else:
                final_mask = original_mask
        else:
            final_mask = original_mask
        
        # Final mask tensor creation for return
        final_mask_tensor = torch.from_numpy(final_mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return final_mask_tensor, kspace, target, maximum, fname, slice, max_slice_index
