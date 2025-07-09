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
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Apply augmentation if available and not in forward mode
        if not self.isforward and self.augmentor is not None:
            # Get target size from original target
            if hasattr(attrs, self.max_key) and target is not None:
                target_size = target.shape
            else:
                target_size = [384, 384]  # Default target size
                
            kspace, augmented_target = self.augmentor(kspace, target_size)
            if augmented_target is not None:
                target = augmented_target
        
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice, max_slice_index
