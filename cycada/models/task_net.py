import torch
import torch.nn as nn
from torch.nn import init
from .models import register_model 
from .util import init_weights
import numpy as np

class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.setup_net()
        self.criterion = nn.CrossEntropyLoss()
        if weights_init is not None:
            self.load(weights_init)
        else:
            init_weights(self)

    def forward(self, x, with_ft=False):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        score = self.classifier(x)
        if with_ft:
            return score, x
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

@register_model('LeNet')
class LeNet(TaskNet):
    "Network used for MNIST or USPS experiments."    

    num_channels = 1
    image_size = 28
    name = 'LeNet'
    out_dim = 500 # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
                nn.Conv2d(self.num_channels, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Linear(50*4*4, 500)
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(500, self.num_cls)
                )


@register_model('DTN')
class DTNClassifier(TaskNet):
    "Classifier used for SVHN->MNIST Experiment"

    num_channels = 3
    image_size = 32
    name = 'DTN'
    out_dim = 512 # dim of last feature layer

    def setup_net(self):
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_cls)
                )
