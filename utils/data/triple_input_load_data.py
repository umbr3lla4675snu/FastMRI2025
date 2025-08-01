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
import utils.model.fastmri as fastmri
from utils.model.fastmri.data import transforms
from utils.common.utils import center_crop


class TripleInputSliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        
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
            if self.input_key[0] in hf.keys():
                num_slices = hf[self.input_key[0]].shape[0]
            elif self.input_key[1] in hf.keys():
                num_slices = hf[self.input_key[1]].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        
        image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        
        if image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input_kspace = hf[self.input_key[0]][dataslice]
            mask = np.array(hf["mask"])
            max_slice_index = hf[self.input_key[0]].shape[0] - 1

        with h5py.File(image_fname, "r") as hf:
            input_alise = hf[self.input_key[1]][dataslice]
            input_grappa = hf[self.input_key[2]][dataslice]
        if self.forward:
            target = -1
            attrs = -1
            
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        
        return self.transform(mask, input_kspace, target, attrs, kspace_fname.name, dataslice, max_slice_index, input_grappa, input_alise)


class TripleInputDataTransform:
    def __init__(self, isforward, max_key, args=None):
        self.isforward = isforward
        self.max_key = max_key
        self.args = args
        
    
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
            # Calculate maximum value from the generated images for proper normalization
            grappa_tensor = to_tensor(grappa_image)
            input_tensor = to_tensor(input_image)
            
            # Scale images to more appropriate range (similar to training data)
            # Multiply by a factor to bring values to typical MRI range
            scale_factor = 1000.0  # Bring values from ~0.0005 to ~0.5 range
            grappa_tensor = grappa_tensor * scale_factor
            input_tensor = input_tensor * scale_factor
            
            # Use maximum value for normalization
            maximum = max(grappa_tensor.max().item(), input_tensor.max().item())
            
            # Ensure maximum is reasonable
            if maximum < 0.1:
                maximum = 1.0
            
            # Even in forward mode, we need the actual images for triple input NAFNet
            grappa_image = grappa_tensor
            input_image = input_tensor
        
        # Store original mask for later use
        original_mask = mask.copy()
        original_freq_dim = mask.shape[0]
        
        full_kspace = to_tensor(input_kspace)
        full_kspace = torch.stack((full_kspace.real, full_kspace.imag), dim=-1)
        
        kspace = to_tensor(input_kspace * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        
        
        # Final mask tensor creation for return
        final_mask_tensor = torch.from_numpy(original_mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
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
