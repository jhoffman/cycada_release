import logging
import logging.config
import os.path
from collections import OrderedDict

import numpy as np
import torch
import yaml
from torch.nn.parameter import Parameter
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)


def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def safe_load_state_dict(net, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. Any params in :attr:`state_dict`
    that do not match the keys returned by :attr:`net`'s :func:`state_dict()`
    method or have differing sizes are skipped.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = net.state_dict()
    skipped = []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].size() != param.size():
            skipped.append(name)
            continue
        own_state[name].copy_(param)

    if skipped:
        logging.info('Skipped loading some parameters: {}'.format(skipped))

def step_lr(optimizer, mult):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = lr * mult
