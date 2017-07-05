# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
A script to check the structure of CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import argparse
import sys
import torch

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Script to check CUHK03 dataset (mat file).')
    parser.add_argument('--mat',
                        dest='mat_file_path',
                        help='Original CUHK03 file path.',
                        required=True)
    args = parser.parse_args()

    return args

def create_dataset(file_path):

    with h5py.File(file_path,'r') as f:
        val_index = np.array(f[f['testsets'][0][0]][:].T - 1)
        print('validation set dimension: ', val_index.shape)
        print('first pair (camera_num and identity): ', val_index[0])
        print('')
        tes_index = np.array(f[f['testsets'][0][1]][:].T - 1)
        print('test set dimension: ',tes_index.shape)
        print('first pair (camera_num and identity): ', tes_index[0])
        print('')
        print('labeled:(5*1 cells)')
        for i in range(5):
            print('                     %d * 10' %(f[f['labeled'][0][i]][0].size))
        image0 = np.array(f[f[f['labeled'][0][0]][0][0]][:], dtype=np.int32)
        image0_tensor = torch.from_numpy(image0)
        print('first image size: ', image0_tensor.size())
        print('')
        print('detected:(5*1 cells)')
        for i in range(5):
            print('                     %d * 10' %(f[f['detected'][0][i]][0].size))

        sys.exit('exit')



if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
