"""
Reconstruction script for Triple Input NAFNet
Generates enhanced reconstructions using VarNet + GRAPPA + Input images
"""

import argparse
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

import time
from utils.learning.test_triple_input_nafnet import forward


def parse():
    parser = argparse.ArgumentParser(description='Reconstruct using Triple Input NAFNet',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--path_data', type=Path, default='/root/Data/leaderboard', help='Directory of test data (should contain acc4 and acc8 folders)')
    parser.add_argument('--net_name', type=Path, default='triple_input_nafnet', help='Name of network')
    
    # Model checkpoints
    parser.add_argument('--varnet-checkpoint-path', type=Path, 
                       default='/root/result/best_Varnet/checkpoints/best_model.pt',
                       help='Path to pre-trained VarNet checkpoint')
    
    # Data parameters
    parser.add_argument('--input-key', type=str, nargs='+', 
                    default=['kspace', 'image_input', 'image_grappa'], 
                    help='List of input keys')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    start_time = time.time()

    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args)

    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args)

    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')
