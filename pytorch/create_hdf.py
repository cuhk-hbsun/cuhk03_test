# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Implementation-CVPR2015-CNN-for-ReID
# Copyright (c) 2017 Ning Ding
# Licensed under The MIT License [see LICENSE for details]
# Written by Ning Ding
# --------------------------------------------------------

"""
A script to create a HDF5 dataset from original CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import argparse
import sys

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Script to Create HDF5 dataset.')
    parser.add_argument('--mat',
                        dest='mat_file_path',
                        help='Original CUHK03 file path.',
                        required=True)
    args = parser.parse_args()

    return args

def create_dataset(file_path):

    with h5py.File(file_path,'r') as f, h5py.File('cuhk-03.h5') as fw:

        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()

        fwa = fw.create_group('a')
        fwb = fw.create_group('b')
        fwat = fwa.create_group('train')
        fwat_id = fwa.create_group('train_id')
        fwbv = fwb.create_group('validation')
        fwbv_id = fwb.create_group('validation_id')
        fwbe = fwb.create_group('test')
        fwbe_id = fwb.create_group('test_id')

        temp = []
        temp_t_id = []
        temp_v_id = []
        temp_e_id = []
        count_t = 0
        count_v = 0
        count_e = 0
        count_t_id = 0
        count_v_id = 0
        count_e_id = 0
        # just use camera pair 1 (totally 5 pairs)
        for i in xrange(1):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                # train dataset: from camera 1a
                temp_t_id.append(k)
                for j in xrange(5):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                fwat.create_dataset(str(count_t),data = np.array(temp))
                temp = []
                count_t += 1
                # valication and test dataset: from camera 1b
                if [i,k] in val_index:
                    temp_v_id.append(k)
                    for j in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                    fwbv.create_dataset(str(count_v),data = np.array(temp))
                    temp = []
                    count_v += 1
                if [i,k] in tes_index:
                    temp_e_id.append(k)
                    for j in xrange(5,10):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((224,224))) / 255.)
                    fwbe.create_dataset(str(count_e),data = np.array(temp))
                    temp = []
                    count_e += 1
            fwat_id.create_dataset(str(count_t_id),data = np.array(temp_t_id))
            fwbv_id.create_dataset(str(count_v_id),data = np.array(temp_v_id))
            fwbe_id.create_dataset(str(count_e_id),data = np.array(temp_e_id))


if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
