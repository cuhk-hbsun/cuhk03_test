from __future__ import print_function
import argparse
import h5py
import sys
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from cuhk03_alexnet import AlexNet

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
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

def _get_train_data(train='train'):
    with h5py.File('cuhk-03.h5','r') as ff:
        a = np.array([ff['a'][train][str(i)][0] for i in range(1163)])
        b = np.array([ff['b'][train][str(i)][0] for i in range(1163)])
        c = np.array([ff['a'][train].keys()[i] for i in range(1163)], dtype=np.int32)
        return a,b,c

def _get_data(val_or_test):
    with h5py.File('cuhk-03.h5','r') as ff:
        a = np.array([ff['a'][val_or_test][str(i)][0] for i in range(100)])
        b = np.array([ff['b'][val_or_test][str(i)][0] for i in range(100)])
        c = np.array([ff['a'][val_or_test].keys()[i] for i in range(100)], dtype=np.int32)
        class_length = len(ff['a'][val_or_test].keys())
        return a,b,c

def _normalize(train_or_val_or_test, use_camera_a=True):
    if train_or_val_or_test == 'train':
        a,b,c = _get_train_data(train_or_val_or_test)
        num_sample = 1163
    else:
        a,b,c = _get_data(train_or_val_or_test)
        num_sample = 100

    data = a
    if not use_camera_a:
        data = b

    data = data.transpose(0, 3, 1, 2)
    data_tensor = torch.from_numpy(data)
    print(data_tensor.size())

    data_mean = np.mean(data, (2,3))
    data_std = np.std(data, (2,3))
    data_mean_tensor = torch.from_numpy(data_mean)
    data_std_tensor = torch.from_numpy(data_std)


    data_tensor_nor =  data_tensor

    for i in range(num_sample):
        transform=transforms.Compose([
            transforms.Normalize(data_mean_tensor[i], data_std_tensor[i])
        ])
        data_tensor_nor[i] = transform(data_tensor[i])

    features = data_tensor_nor
    targets = torch.from_numpy(c)

    return features, targets


model = AlexNet()
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

train_features, train_targets = _normalize('train', use_camera_a=True)
train = data_utils.TensorDataset(train_features, train_targets)
train_loader = data_utils.DataLoader(train, batch_size=args.train_batch_size, shuffle=True)

test_features, test_targets = _normalize('test', use_camera_a=True)
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
        # print('target', target)
        optimizer.zero_grad()
        output = model(data)
        # print('output', output)
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
        output = model(data)
        # print('output', output)
        test_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        # print('pred', pred)
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
