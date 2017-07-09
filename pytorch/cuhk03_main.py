from __future__ import print_function
import argparse
import h5py
import sys
import argparse
import numpy as np
import scipy.misc
import scipy.io as sio
# import matplotlib
# import matplotlib.image as matimg
# from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision import models
import torch.utils.data as data_utils
from cuhk03_alexnet import AlexNet

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 5)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def _get_train_data(train):
    with h5py.File('cuhk-03.h5', 'r') as ff:
        temp = []
        num_sample = len(ff['a'][train+'_id'][str(0)])
        num_of_same_image_array = []
        num_sample_total = 0
        for i in range(num_sample):
            num_of_same_image = len(ff['a'][train][str(i)])
            num_sample_total += num_of_same_image
            num_of_same_image_array.append(num_of_same_image)
            for k in range(num_of_same_image):
                temp.append(np.array(ff['a'][train][str(i)][k]))
        image_set = np.array(temp)
        image_id_temp = np.array(ff['a'][train+'_id'][str(0)])
        image_id = []
        for i in range(num_sample):
            for k in range(num_of_same_image_array[i]):
                image_id.append(image_id_temp[i])
        image_id = np.array(image_id)
        return image_set, image_id, num_sample_total
        """below for image save"""
        # num_of_same_image = len(ff['a'][train][str(292)])
        # num_sample_total += num_of_same_image
        # num_of_same_image_array.append(num_of_same_image)
        # for k in range(num_of_same_image):
        #     temp.append(np.array(ff['a'][train][str(292)][k]))
        # image_set = np.array(temp)
        # image_id_temp = np.array(ff['a'][train+'_id'][str(0)])
        # image_id = []
        # for k in range(num_of_same_image_array[0]):
        #     image_id.append(image_id_temp[292])
        # image_id = np.array(image_id)
        # return image_set, image_id, num_sample_total


def _get_data(val_or_test):
    with h5py.File('cuhk-03.h5','r') as ff:
        num_sample = len(ff['a'][val_or_test+'_id'][str(0)])
        image_set = np.array([ff['a'][val_or_test][str(i)][0] for i in range(num_sample)])
        image_id = np.array(ff['a'][val_or_test+'_id'][str(0)])
        return image_set, image_id, num_sample

def _normalize(train_or_val_or_test):
    if train_or_val_or_test == 'train':
        image_set, image_id, num_sample = _get_train_data(train_or_val_or_test)
    else:
        image_set, image_id, num_sample = _get_data(train_or_val_or_test)

    data = image_set

    data = data.transpose(0, 3, 1, 2)

    data_tensor = torch.from_numpy(data)

    data_mean = np.mean(data, (2,3))
    data_std = np.std(data, (2,3))
    data_mean_tensor = torch.from_numpy(data_mean)
    data_std_tensor = torch.from_numpy(data_std)


    data_tensor_nor = data_tensor

    for i in range(num_sample):
        transform=transforms.Compose([
            transforms.Normalize(data_mean_tensor[i], data_std_tensor[i])
        ])
        data_tensor_nor[i] = transform(data_tensor[i])

    features = data_tensor_nor
    targets = torch.from_numpy(image_id)

    return features, targets


model = models.alexnet(pretrained=False)
# model = AlexNet()
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

train_features, train_targets = _normalize('train')
print('train data size', train_features.size())
print('train target size', train_targets.size())
train = data_utils.TensorDataset(train_features, train_targets)
train_loader = data_utils.DataLoader(train, batch_size=args.train_batch_size, shuffle=True)

test_features, test_targets = _normalize('test')
print('test data size', test_features.size())
print('test target size', test_targets.size())
test = data_utils.TensorDataset(test_features, test_targets)
test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        data = data.float()
        target = target.long()
        # print(target)
        # target_id = target.data.numpy()
        # image_set = data.data.numpy()
        # for i in range(5):
        #     img1 = image_set[i].transpose(1,2,0)
        #     scipy.misc.imsave('train_'+str(i)+'_'+str(target_id[i])+'.png', img1)
        # sys.exit('exit')
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.float()
        target = target.long()
        # print('target', target)
        # image_set = data.data.numpy()
        # target_id = target.data.numpy()
        # for i in range(10):
        #     img1 = image_set[i].transpose(1,2,0)
        #     scipy.misc.imsave('test_'+str(i)+'_'+str(target_id[i])+'.png', img1)
        # sys.exit('exit')
        output = model(data)
        test_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
