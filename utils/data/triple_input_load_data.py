"""
Enhanced data loader for Triple Input NAFNet
Loads kspace, grappa image, and input image together
"""

import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
from utils.data.Augmentation.data_augment import DataAugmentor
import utils.model.fastmri as fastmri
from utils.model.fastmri.data import transforms


class TripleInputSliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input_kspace = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
            max_slice_index = hf[self.input_key].shape[0] - 1

        if self.forward:
            target = -1
            attrs = -1
            grappa_image = -1
            input_image = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
                
                # Load GRAPPA image
                if 'image_grappa' in hf.keys():
                    grappa_image = hf['image_grappa'][dataslice]
                else:
                    # If GRAPPA image is not available, create a placeholder
                    print(f"Warning: image_grappa not found in {image_fname.name}, using zero-filled reconstruction")
                    # Create zero-filled reconstruction as fallback
                    kspace_tensor = torch.from_numpy(input_kspace)
                    kspace_tensor = torch.stack([kspace_tensor.real, kspace_tensor.imag], dim=-1)
                    grappa_image = fastmri.complex_abs(fastmri.ifft2c(kspace_tensor)).numpy()
                
                # Load input aliased image  
                if 'image_input' in hf.keys():
                    input_image = hf['image_input'][dataslice]
                else:
                    print(f"Warning: image_input not found in {image_fname.name}, creating from kspace")
                    # Create aliased image from undersampled kspace
                    kspace_tensor = torch.from_numpy(input_kspace * mask)
                    kspace_tensor = torch.stack([kspace_tensor.real, kspace_tensor.imag], dim=-1)
                    
                    # Convert to image space
                    image_tensor = fastmri.ifft2c(kspace_tensor)
                    
                    # Apply root sum of squares for multicoil data
                    if image_tensor.dim() > 3:  # multicoil
                        input_image = fastmri.rss(fastmri.complex_abs(image_tensor), dim=0).numpy()
                    else:  # single coil
                        input_image = fastmri.complex_abs(image_tensor).numpy()
        
        return self.transform(mask, input_kspace, target, attrs, kspace_fname.name, dataslice, max_slice_index, grappa_image, input_image)


class TripleInputDataTransform:
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
    
    def __call__(self, mask, input_kspace, target, attrs, fname, slice_idx, max_slice_index, grappa_image, input_image):
        def to_tensor(data):
            return torch.from_numpy(data)
        
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
            grappa_image = to_tensor(grappa_image)
            input_image = to_tensor(input_image)
        else:
            target = -1
            maximum = -1
            grappa_image = -1
            input_image = -1
        
        # Store original mask for later use
        original_mask = mask.copy()
        original_freq_dim = mask.shape[0]
        
        full_kspace = to_tensor(input_kspace)
        full_kspace = torch.stack((full_kspace.real, full_kspace.imag), dim=-1)
        
        kspace = to_tensor(input_kspace * mask)
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
        
        return final_mask_tensor, kspace, target, maximum, fname, slice_idx, max_slice_index, grappa_image, input_image


def create_triple_input_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
        
    data_storage = TripleInputSliceData(
        root=data_path,
        transform=TripleInputDataTransform(isforward, max_key_, args),
        input_key=args.input_key,
        target_key=target_key_,
        forward=isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
