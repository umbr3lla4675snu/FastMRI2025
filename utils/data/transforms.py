import numpy as np
import torch

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

class DataTransform:
    def __init__(self, isforward, max_key, args=None):
        self.isforward = isforward
        self.max_key = max_key
        self.args = args
        
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
            # Get target size from original target
            if target is not None:
                target_size = target.shape
            else:
                target_size = [384, 384]  # Default target size
                
            augmented_kspace, augmented_target, augmentation_applied = self.augmentor(full_kspace, target_size)
            
            # Only apply augmentation if it was actually applied and target is valid
            if augmentation_applied and augmented_target is not None:
                target = augmented_target
                # Update kspace to the augmented version
                kspace = augmented_kspace
                
                # Adjust mask to match the new kspace dimensions
                new_freq_dim = augmented_kspace.shape[-2]
                
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
