from __future__ import print_function
import argparse
import h5py
import sys
import argparse
import numpy as np
# import scipy.misc
# import scipy.io as sio
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
parser.add_argument('--train-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                    help='learning rate (default: 1e-6)')
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

def _get_train_data(train, group):
    with h5py.File('cuhk-03.h5', 'r') as ff:
        temp = []
        num_sample = len(ff[group][train+'_id'][str(0)])
        num_of_same_image_array = []
        num_sample_total = 0
        for i in range(num_sample):
            num_of_same_image = len(ff[group][train][str(i)])
            num_sample_total += num_of_same_image
            num_of_same_image_array.append(num_of_same_image)
            for k in range(num_of_same_image):
                temp.append(np.array(ff[group][train][str(i)][k]))
        image_set = np.array(temp)
        image_id_temp = np.array(ff[group][train+'_id'][str(0)])
        image_id = []
        for i in range(num_sample):
            for k in range(num_of_same_image_array[i]):
                image_id.append(image_id_temp[i])
        image_id = np.array(image_id)

        data = image_set.transpose(0, 3, 1, 2)
        data_for_mean = image_set.transpose(3, 0, 1, 2)
        data_mean = np.mean(data_for_mean, (1, 2, 3))
        data_std = np.std(data_for_mean, (1, 2, 3))
        features = torch.from_numpy(data)
        targets = torch.from_numpy(image_id)

        return features, targets, data_mean, data_std


def _get_data(val_or_test, group):
    with h5py.File('cuhk-03.h5','r') as ff:
        num_sample = len(ff[group][val_or_test+'_id'][str(0)])
        image_set = np.array([ff[group][val_or_test][str(i)][0] for i in range(num_sample)])
        image_id = np.array(ff[group][val_or_test+'_id'][str(0)])
        data = image_set.transpose(0, 3, 1, 2)
        features = torch.from_numpy(data)
        targets = torch.from_numpy(image_id)
        return features, targets


# model = AlexNet()
model = models.alexnet(pretrained=True)

# remove last fully-connected layer
# new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# model.classifier = new_classifier
# modify last fc layer
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(4096, 843)
if args.cuda:
    model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

train_features, train_targets, mean, std = _get_train_data('train', 'a')
print('train data size', train_features.size())
print('train target size', train_targets.size())
train = data_utils.TensorDataset(train_features, train_targets)
train_loader = data_utils.DataLoader(train, batch_size=args.train_batch_size, shuffle=True)

test_features, test_targets = _get_data('test', 'b')
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
        data = data.float()  # with size of (batch_size * 3 * 224 * 224)
        target = target.long() # with size of (batch_size)

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
