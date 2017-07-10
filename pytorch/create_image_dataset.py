# -*- coding: utf-8 -*-
# --------------------------------------------------------
#
# --------------------------------------------------------

"""
A script to image dataset from original CUHK03 mat file.
"""

import h5py
import numpy as np
from PIL import Image
import scipy.misc
import scipy.io as sio
import argparse
import sys
import itertools as it
import os

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

    with h5py.File(file_path,'r') as f:

        val_index = (f[f['testsets'][0][0]][:].T - 1).tolist()
        tes_index = (f[f['testsets'][0][1]][:].T - 1).tolist()

        # just use camera pair 1 (totally 5 pairs)
        for i in xrange(1):
            for k in xrange(f[f['labeled'][0][i]][0].size):
                print i,k
                for j in it.chain(xrange(1, 5), xrange(6, 10)):
                    if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                        img1 = np.array(f[f[f['labeled'][0][i]][j][k]][:]).transpose(2,1,0)
                        img1 = scipy.misc.imresize(img1, (224,224))
                        file_path = 'train/id'+str(k)+'/'
                        directory = os.path.dirname(file_path)
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        scipy.misc.imsave(directory+'/'+str(j)+'.png', img1)
                # sys.exit('exit')
                if [i,k] in val_index:
                    for j in it.chain(xrange(1), xrange(5, 6)):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            img1 = np.array(f[f[f['labeled'][0][i]][j][k]][:]).transpose(2,1,0)
                            file_path = 'validation/id'+str(k)+'/'
                            directory = os.path.dirname(file_path)
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            scipy.misc.imsave(directory+'/'+str(j)+'.png', img1)

                if [i,k] in tes_index:
                    for j in it.chain(xrange(1), xrange(5, 6)):
                        if len(f[f[f['labeled'][0][i]][j][k]].shape) == 3:
                            img1 = np.array(f[f[f['labeled'][0][i]][j][k]][:]).transpose(2,1,0)
                            file_path = 'test/id'+str(k)+'/'
                            directory = os.path.dirname(file_path)
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            scipy.misc.imsave(directory+'/'+str(j)+'.png', img1)



if __name__ == '__main__':

    args = parse_args()
    create_dataset(args.mat_file_path)
