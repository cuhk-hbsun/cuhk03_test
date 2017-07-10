from __future__ import print_function
import argparse
import h5py
import sys
import argparse
import numpy as np
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
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
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

# model = models.alexnet(pretrained=True)
# model.classifier._modules['6'] = nn.Linear(4096, 843)
model = AlexNet()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Data loading code
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('./train',
                   transform=transforms.Compose([
                       transforms.RandomSizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                            std = [ 0.229, 0.224, 0.225 ]),
                   ])),
    batch_size=args.train_batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('./test', transform=transforms.Compose([
                       transforms.RandomSizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                            std = [ 0.229, 0.224, 0.225 ]),
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
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
