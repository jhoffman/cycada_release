from __future__ import print_function
from os.path import join
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix

from ..data.data_loader import load_data
from ..models.models import get_model
from .util import make_variable

def test(loader, net):
    net.eval()
    test_loss = 0
    correct = 0
    cm = 0
   
    N = len(loader.dataset)
    for idx, (data, target) in enumerate(loader):
        
        # setup data and target #
        data = make_variable(data, requires_grad=False)
        target = make_variable(target, requires_grad=False)
        
        # forward pass
        score = net(data)
        
        # compute loss
        test_loss += net.criterion(score, target).item()
        
        # compute predictions and true positive count
        pred = score.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        #cm += confusion_matrix(target.data.cpu().numpy(), pred.cpu().numpy())
        
    test_loss /= len(loader) # loss function already averages over batch size
    print('[Evaluate] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, N, 100. * correct / N))
    return cm


def load_and_test_net(data, datadir, weights, model, num_cls, 
        dset='test', base_model=None):
    
    # Setup GPU Usage
    if torch.cuda.is_available(): 
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    # Eval tgt from AddaNet or TaskNet model #
    if model == 'AddaNet':
        net = get_model(model, num_cls=num_cls, weights_init=weights, 
                model=base_model)
        net = net.tgt_net
    else:
        net = get_model(model, num_cls=num_cls, weights_init=weights)

    # Load data
    test_data = load_data(data, dset, batch=100, 
        rootdir=datadir, num_channels=net.num_channels, 
        image_size=net.image_size, download=True, kwargs=kwargs)
    if test_data is None:
        print('skipping test')
    else:
        return test(test_data, net)
