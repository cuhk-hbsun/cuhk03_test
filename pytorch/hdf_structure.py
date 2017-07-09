import h5py
import sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    # list0 = ['train', 'validation', 'test']
    # list0 = ['train', 'train_id']
    # list1 = ['validation', 'test', 'validation_id', 'test_id']
    list0 = ['train', 'validation', 'test', 'train_id', 'validation_id', 'test_id']
    for train_or_validation in list0:
        length = len(item['a'][train_or_validation].keys())
        print('length of %s dataset: %d' %(train_or_validation, length))
    # for train_or_validation in list1:
    #     length = len(item['b'][train_or_validation].keys())
    #     print('length of %s dataset: %d' %(train_or_validation, length))
    print_hdf5_item_structure(item)
    file.close()

def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file
        print '(File)', g.name

    elif isinstance(g,h5py.Dataset) :
        print '(Dataset) ', g.name, '    len =', g.shape #, g.dtype
        # print '(Dataset_content, 5 images with size (160*60*3)) '
        # print g.value
        # sys.exit ( "scan five image from one camera" )

    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name

    else :
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )

    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            # print offset, 'key=', key #,"   ", subg.name , val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

if __name__ == "__main__" :
    print_hdf5_file_structure('cuhk-03.h5')
    # print_hdf5_file_structure('cuhk-03-id.h5')
    sys.exit ( "End of test" )
