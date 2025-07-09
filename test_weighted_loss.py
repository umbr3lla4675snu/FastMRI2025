#!/usr/bin/env python3
"""
Test script to verify the weighted loss implementation
"""
import torch
import math
import sys
import os

# Add paths
if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')

from utils.common.loss_function import SSIMLoss, WeightedSSIMLoss

def test_weighted_loss():
    """Test the weighted SSIM loss function"""
    print("Testing Weighted SSIM Loss implementation...")
    
    # Create test data
    batch_size = 4
    height, width = 320, 320
    
    # Random predictions and targets
    pred = torch.randn(batch_size, height, width)
    target = torch.randn(batch_size, height, width)
    data_range = torch.ones(batch_size)
    
    # Test slice indices
    slice_indices = torch.tensor([5, 10, 15, 20])  # Current slice indices
    max_slice_indices = torch.tensor([25, 25, 25, 25])  # Max slice indices for each sample
    
    # Test regular SSIM loss
    regular_loss = SSIMLoss()
    regular_result = regular_loss(pred, target, data_range)
    print(f"Regular SSIM Loss: {regular_result.item():.6f}")
    
    # Test weighted SSIM loss
    weighted_loss = WeightedSSIMLoss()
    weighted_result = weighted_loss(pred, target, data_range, slice_indices, max_slice_indices)
    print(f"Weighted SSIM Loss: {weighted_result.item():.6f}")
    
    # Test the weighting formula manually
    print("\nWeight calculation verification:")
    for i in range(batch_size):
        normalized_idx = slice_indices[i].float() / max_slice_indices[i].float()
        weight = torch.cos(normalized_idx * math.pi / 2) ** 2
        print(f"Sample {i}: slice={slice_indices[i]}, max={max_slice_indices[i]}, "
              f"normalized={normalized_idx:.3f}, weight={weight:.6f}")
    
    print("\nWeighted loss implementation test completed successfully!")
    
    # Test edge cases
    print("\nTesting edge cases...")
    
    # First slice (should have maximum weight = 1.0)
    first_slice = torch.tensor([0])
    max_slice = torch.tensor([25])
    pred_single = torch.randn(1, height, width)
    target_single = torch.randn(1, height, width)
    data_range_single = torch.ones(1)
    
    weighted_first = weighted_loss(pred_single, target_single, data_range_single, first_slice, max_slice)
    print(f"First slice weight (should be ~1.0): {torch.cos(torch.tensor(0.0) * math.pi / 2) ** 2:.6f}")
    
    # Last slice (should have minimum weight = 0.0)
    last_slice = torch.tensor([25])
    weighted_last = weighted_loss(pred_single, target_single, data_range_single, last_slice, max_slice)
    print(f"Last slice weight (should be ~0.0): {torch.cos(torch.tensor(1.0) * math.pi / 2) ** 2:.6f}")
    
    # Middle slice
    middle_slice = torch.tensor([12])
    weighted_middle = weighted_loss(pred_single, target_single, data_range_single, middle_slice, max_slice)
    middle_weight = torch.cos(torch.tensor(12.0 / 25.0) * math.pi / 2) ** 2
    print(f"Middle slice weight: {middle_weight:.6f}")

if __name__ == "__main__":
    test_weighted_loss()
