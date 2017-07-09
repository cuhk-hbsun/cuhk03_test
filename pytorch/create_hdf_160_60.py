# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Group a has train, test, and validation datasets
# Group b has test and validation datasets
# --------------------------------------------------------

"""
A script to create a HDF5 dataset from original CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import argparse
import sys
import itertools as it

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

    with h5py.File(file_path,'r') as f, h5py.File('cuhk-03-160-60.h5') as fw:

        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()

        fwa = fw.create_group('a')
        fwb = fw.create_group('b')
        fwat = fwa.create_group('train')
        fwae = fwa.create_group('test')
        fwav = fwa.create_group('validation')
        fwat_id = fwa.create_group('train_id')
        fwae_id = fwa.create_group('test_id')
        fwav_id = fwa.create_group('validation_id')

        # fwbt = fwb.create_group('train')
        # fwbt_id = fwb.create_group('train_id')
        fwbv = fwb.create_group('validation')
        fwbv_id = fwb.create_group('validation_id')
        fwbe = fwb.create_group('test')
        fwbe_id = fwb.create_group('test_id')

        temp = []
        a_temp_t_id = []
        a_temp_v_id = []
        a_temp_e_id = []
        b_temp_v_id = []
        b_temp_e_id = []
        a_count_t = 0
        a_count_v = 0
        a_count_e = 0
        b_count_v = 0
        b_count_e = 0
        a_count_t_id = 0
        a_count_v_id = 0
        a_count_e_id = 0
        b_count_v_id = 0
        b_count_e_id = 0
        # just use camera pair 1 (totally 5 pairs)
        for i in xrange(1):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                a_temp_t_id.append(k)
                for j in it.chain(xrange(1, 5), xrange(6, 10)):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((160,60))) / 255.)
                fwat.create_dataset(str(a_count_t),data = np.array(temp))
                temp = []
                a_count_t += 1

                if [i,k] in val_index:
                    a_temp_v_id.append(k)
                    for j in xrange(1):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((160,60))) / 255.)
                    fwav.create_dataset(str(a_count_v),data = np.array(temp))
                    temp = []
                    a_count_v += 1

                    b_temp_v_id.append(k)
                    for j in xrange(5,6):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((160,60))) / 255.)
                    fwbv.create_dataset(str(b_count_v),data = np.array(temp))
                    temp = []
                    b_count_v += 1

                if [i,k] in tes_index:
                    a_temp_e_id.append(k)
                    for j in xrange(1):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((160,60))) / 255.)
                    fwae.create_dataset(str(a_count_e),data = np.array(temp))
                    temp = []
                    a_count_e += 1

                    b_temp_e_id.append(k)
                    for j in xrange(5,6):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            temp.append(np.array((Image.fromarray(f[f[f['labeled'][0][i]][j][k]][:].transpose(2,1,0))).resize((160,60))) / 255.)
                    fwbe.create_dataset(str(b_count_e),data = np.array(temp))
                    temp = []
                    b_count_e += 1

            fwat_id.create_dataset(str(a_count_t_id),data = np.array(a_temp_t_id))
            fwav_id.create_dataset(str(a_count_v_id),data = np.array(a_temp_v_id))
            fwae_id.create_dataset(str(a_count_e_id),data = np.array(a_temp_e_id))
            fwbv_id.create_dataset(str(b_count_v_id),data = np.array(b_temp_v_id))
            fwbe_id.create_dataset(str(b_count_e_id),data = np.array(b_temp_e_id))


if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
