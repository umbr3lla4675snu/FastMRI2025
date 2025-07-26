import h5py
import random
from utils.data.transforms import DataTransform
from utils.model.fastmri.data.transforms import VarNetDataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch

class AugmentedVarNetDataTransform:
    """
    VarNetDataTransform wrapper with data augmentation support
    """
    def __init__(self, isforward=False, args=None):
        self.isforward = isforward
        self.args = args
        
        # Initialize base VarNetDataTransform
        self.base_transform = VarNetDataTransform(
            mask_func=None,
            use_seed=True
        )
        
        # Initialize DataAugmentor if augmentation is enabled and not in forward mode
        if not isforward and args is not None and hasattr(args, 'aug_on') and args.aug_on:
            from utils.data.Augmentation.data_augment import DataAugmentor
            # Create a function that returns current epoch
            def get_current_epoch():
                return getattr(args, 'current_epoch', 0)
            
            self.augmentor = DataAugmentor(args, get_current_epoch)
        else:
            self.augmentor = None
    
    def __call__(self, kspace, mask, target, attrs, fname, slice_num, max_slice_index):
        # Apply base transform first
        if not self.isforward and self.augmentor is not None:
            # For augmentation, we need to work with the original data
            from utils.model.fastmri.data.transforms import to_tensor
            
            # Convert to tensor format for augmentation
            kspace_tensor = to_tensor(kspace)
            # kspace_tensor is already in [real, imag] format in the last dimension
            # No need to call .real and .imag - it's already stacked
            full_kspace = kspace_tensor
            
            # Get target size
            if target is not None:
                target_tensor = to_tensor(target)
                target_size = target_tensor.shape
            else:
                target_size = [384, 384]  # Default target size
            
            # Apply augmentation
            augmented_kspace, augmented_target, augmentation_applied = self.augmentor(full_kspace, target_size)
            
            if augmentation_applied and augmented_target is not None:
                # Convert back to numpy for base transform
                # augmented_kspace is already in [real, imag] format
                augmented_kspace_np = augmented_kspace[..., 0].numpy() + 1j * augmented_kspace[..., 1].numpy()
                augmented_target_np = augmented_target.numpy()
                
                # Adjust mask if necessary
                original_shape = kspace.shape
                new_shape = augmented_kspace_np.shape
                
                # Handle mask adjustment for frequency dimension changes
                if new_shape != original_shape:
                    # Create new mask that matches augmented data dimensions
                    if mask.ndim == 1:
                        # 1D mask case
                        original_freq_dim = len(mask)
                        new_freq_dim = new_shape[-2]
                        
                        if new_freq_dim != original_freq_dim:
                            if new_freq_dim > original_freq_dim:
                                pad_size = (new_freq_dim - original_freq_dim) // 2
                                adjusted_mask = np.zeros(new_freq_dim, dtype=mask.dtype)
                                adjusted_mask[pad_size:pad_size + original_freq_dim] = mask
                            else:
                                start_idx = (original_freq_dim - new_freq_dim) // 2
                                adjusted_mask = mask[start_idx:start_idx + new_freq_dim]
                        else:
                            adjusted_mask = mask
                    else:
                        # Multi-dimensional mask case
                        adjusted_mask = mask
                        # Additional logic for multi-dim masks if needed
                else:
                    adjusted_mask = mask
                
                # Update attrs for reconstruction size and add missing VarNet attributes
                updated_attrs = attrs.copy()
                if 'recon_size' in updated_attrs:
                    updated_attrs['recon_size'] = list(augmented_target_np.shape[-2:])
                else:
                    updated_attrs['recon_size'] = list(augmented_target_np.shape[-2:])
                
                # Add required VarNet attributes if missing
                if 'padding_left' not in updated_attrs:
                    updated_attrs['padding_left'] = 0
                if 'padding_right' not in updated_attrs:
                    updated_attrs['padding_right'] = augmented_kspace_np.shape[-1]
                if 'max' not in updated_attrs:
                    updated_attrs['max'] = float(np.abs(augmented_target_np).max()) if augmented_target_np is not None else 1.0
                
                # Apply base transform with augmented data
                base_result = self.base_transform(
                    kspace=augmented_kspace_np,
                    mask=adjusted_mask,
                    target=augmented_target_np,
                    attrs=updated_attrs,
                    fname=fname,
                    slice_num=slice_num
                )
                
                # VarNetDataTransform returns: (masked_kspace, mask, target, fname, slice_num, max_value, crop_size)
                # We need to add max_slice_index to the return tuple
                return base_result + (max_slice_index,)
        
        # Apply base transform without augmentation
        # Add required VarNet attributes if missing
        updated_attrs = attrs.copy()
        if 'padding_left' not in updated_attrs:
            updated_attrs['padding_left'] = 0
        if 'padding_right' not in updated_attrs:
            updated_attrs['padding_right'] = kspace.shape[-1]
        if 'max' not in updated_attrs and target is not None:
            updated_attrs['max'] = float(np.abs(target).max())
        elif 'max' not in updated_attrs:
            updated_attrs['max'] = 1.0
        if 'recon_size' not in updated_attrs:
            if target is not None:
                updated_attrs['recon_size'] = list(target.shape[-2:])
            else:
                updated_attrs['recon_size'] = [384, 384]  # Default size
        
        base_result = self.base_transform(
            kspace=kspace,
            mask=mask,
            target=target,
            attrs=updated_attrs,
            fname=fname,
            slice_num=slice_num
        )
        
        # Add max_slice_index to the return tuple
        return base_result + (max_slice_index,)

class SliceData(Dataset):
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
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
            max_slice_index = hf[self.input_key].shape[0] - 1  # Get total number of slices - 1
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, max_slice_index)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_, args),  # args 전달
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
