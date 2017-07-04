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
import scipy.io as sio


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CUHK03 Example')
parser.add_argument('--train-batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--test-batch-size', type=int, default=5, metavar='N',
                    help='input batch size for testing (default: 5)')
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
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


def _get_data(train_or_val):
    with h5py.File('../CUHK03/cuhk-03.h5','r') as ff:
        a = np.array([ff['a'][train_or_val][str(i)][1] for i in range(10)])
        b = np.array([ff['b'][train_or_val][str(i)][0] for i in range(10)])
        c = np.array([ff['a'][train_or_val].keys()[i] for i in range(10)], dtype=np.int32)
        class_length = len(ff['a'][train_or_val].keys())
        return a,b,c,class_length

def _normalize(train_or_val):
    a,b,c,class_length = _get_data(train_or_val)
    # print(c[9])
    a_mean = np.mean(a, (1,2))
    a_std = np.std(a, (1,2))

    a_mean_tensor = torch.from_numpy(a_mean)
    # print(a_mean_tensor[0])
    a_std_tensor = torch.from_numpy(a_std)
    # print(a_std_tensor[0])

    a = a.transpose(0,3,1,2)
    a_tensor = torch.from_numpy(a)
    print(a_tensor.size())
    print(a_tensor[0][0])
    keys = np.arange(60)
    array = a[0][0]
    dict_a = dict(zip(keys, array.T))
    print(dict_a)
    sio.savemat('image.mat', dict_a)
    sys.exit('exit')
    # print(a_tensor.size())
    a_tensor_tran = torch.transpose(a_tensor, 1, 3)
    # print(a_tensor_tran.size())
    a_tensor_tran = torch.transpose(a_tensor_tran, 2, 3)
    print(a_tensor_tran.size())
    print(a_tensor_tran[0][0])
    sys.exit('exit')
    a_tensor_nor = a_tensor_tran
    for i in range(10):
        transform=transforms.Compose([
            # transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
            transforms.Normalize(a_mean_tensor[i], a_std_tensor[i])
        ])
        a_tensor_nor[i] = transform(a_tensor_tran[i])

    features = a_tensor_nor
    # a_tensor_nor = [transform(a_tensor_tran[i]) for i in range(10)]
    # print(a_tensor_nor.size())
    # print(a_tensor_nor[0][1])
    # sys.exit('exit')

    # target = np.zeros((len(c), class_length))
    # i = 0
    # for item in c:
    #     target[i][item] = 1
    #     i += 1
    #targets = torch.from_numpy(target)
    targets = torch.from_numpy(c)

    return features, targets

model = AlexNet()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

train_features, train_targets = _normalize('train')
# print(train_features.size(), train_targets.size())
train = data_utils.TensorDataset(train_features, train_targets)
train_loader = data_utils.DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
test_features, test_targets = _normalize('test')
# print(train_features.size(), train_targets.size())
test = data_utils.TensorDataset(test_features, test_targets)
test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True)



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.float()
        target = target.long()
        # print(data.size(), target.size())
        optimizer.zero_grad()
        output = model(data)
        # print(output.size())
        # print(target)
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
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    # test(epoch)
