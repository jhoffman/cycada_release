
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .util import init_weights
from .models import register_model, get_model 

@register_model('AddaNet')
class AddaNet(nn.Module):
    "Defines and Adda Network."
    def __init__(self, num_cls=10, model='LeNet', src_weights_init=None,
            weights_init=None):
        super(AddaNet, self).__init__()
        self.name = 'AddaNet'
        self.base_model = model
        self.num_cls = num_cls
        self.cls_criterion = nn.CrossEntropyLoss()
        self.gan_criterion = nn.CrossEntropyLoss()
      
        self.setup_net()
        if weights_init is not None:
            self.load(weights_init)
        elif src_weights_init is not None:
            self.load_src_net(src_weights_init)
        else:
            raise Exception('AddaNet must be initialized with weights.')
        

    def forward(self, x_s, x_t):
        """Pass source and target images through their
        respective networks."""
        score_s, x_s = self.src_net(x_s, with_ft=True)
        score_t, x_t = self.tgt_net(x_t, with_ft=True)

        if self.discrim_feat:
            d_s = self.discriminator(x_s)
            d_t = self.discriminator(x_t)
        else:
            d_s = self.discriminator(score_s)
            d_t = self.discriminator(score_t)
        return score_s, score_t, d_s, d_t

    def setup_net(self):
        """Setup source, target and discriminator networks."""
        self.src_net = get_model(self.base_model, num_cls=self.num_cls)
        self.tgt_net = get_model(self.base_model, num_cls=self.num_cls)

        input_dim = self.num_cls 
        self.discriminator = nn.Sequential(
                nn.Linear(input_dim, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, 2),
                )

        self.image_size = self.src_net.image_size
        self.num_channels = self.src_net.num_channels

    def load(self, init_path):
        "Loads full src and tgt models."
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def load_src_net(self, init_path):
        """Initialize source and target with source
        weights."""
        self.src_net.load(init_path)
        self.tgt_net.load(init_path)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

    def save_tgt_net(self, out_path):
        torch.save(self.tgt_net.state_dict(), out_path)

