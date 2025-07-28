import h5py
import random
from utils.data.transforms import DataTransform
from utils.model.fastmri.data.transforms import VarNetDataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch

# class DataTransform:
#     """
#     Data Transformer using VarNet's transform with augmentation support
#     """
#     def __init__(self, isforward=False, args=None):
#         self.isforward = isforward
#         self.args = args
        
#         # Initialize DataAugmentor if augmentation is enabled and not in forward mode
#         augmentor = None
#         if not isforward and args is not None and hasattr(args, 'aug_on') and args.aug_on:
#             from utils.data.Augmentation.data_augment import DataAugmentor
#             # Create a function that returns current epoch
#             def get_current_epoch():
#                 return getattr(args, 'current_epoch', 0)
            
#             augmentor = DataAugmentor(args, get_current_epoch)

#         mask = create_mask_for_mask_type(
#         args.mask_type, args.center_fractions, args.accelerations
#     )
        
#         # Initialize VarNetDataTransform with augmentor
#         self.transform = VarNetDataTransform(
#             augmentor=augmentor,
#             mask_func=None,
#             use_seed=True
#         )
    
#     def __call__(self, kspace, mask, target, attrs, fname, slice_num, max_slice_index):
#         # Add required VarNet attributes if missing
#         if attrs != -1 and isinstance(attrs, dict):  # Check if not in forward mode and is dict
#             updated_attrs = attrs.copy()
#             if 'padding_left' not in updated_attrs:
#                 updated_attrs['padding_left'] = 0
#             if 'padding_right' not in updated_attrs:
#                 updated_attrs['padding_right'] = kspace.shape[-1]
#             if 'max' not in updated_attrs and target is not None and hasattr(target, 'shape'):
#                 updated_attrs['max'] = float(np.abs(target).max())
#             elif 'max' not in updated_attrs:
#                 updated_attrs['max'] = 1.0
#             if 'recon_size' not in updated_attrs:
#                 if target is not None and hasattr(target, 'shape'):
#                     updated_attrs['recon_size'] = list(target.shape[-2:])
#                 else:
#                     updated_attrs['recon_size'] = [384, 384]  # Default size
#         else:
#             # Forward mode - create minimal attrs
#             updated_attrs = {
#                 'padding_left': 0,
#                 'padding_right': kspace.shape[-1],
#                 'max': 1.0,
#                 'recon_size': [384, 384]
#             }
        
#         # Call VarNetDataTransform
#         # Handle forward mode where target might be -1 (int) or an array
#         if isinstance(target, (int, float)) and target == -1:
#             target_to_pass = None
#         elif target is None:
#             target_to_pass = None
#         else:
#             target_to_pass = target
        
#         base_result = self.transform(
#             kspace=kspace,
#             mask=mask,
#             target=target_to_pass,
#             attrs=updated_attrs,
#             fname=fname,
#             slice_num=slice_num
#         )
        
#         # VarNetDataTransform returns: (masked_kspace, mask, target, fname, slice_num, max_value, crop_size)
#         # Reorder to match train_part.py expectations: (mask, kspace, target, maximum, fname, slices, crop_size, max_slice_index)
#         masked_kspace, mask, target, fname, slice_num, max_value, crop_size = base_result
        
#         return (mask, masked_kspace, target, max_value, fname, slice_num, crop_size, max_slice_index)

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
        transform=DataTransform(isforward, max_key_, args, debug_augmentation=True),  # args 전달
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
