from __future__ import print_function
import os
from os.path import join
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms

from ..util import to_tensor_raw

def load_data(name, dset, batch=64, rootdir='', num_channels=3,
        image_size=32, download=True, kwargs={}):
    is_train = (dset == 'train')
    if isinstance(name, list) and len(name) == 2: # load adda data
        src_dataset = get_dataset(name[0], join(rootdir, name[0]), dset, 
                image_size, num_channels, download=download)
        tgt_dataset = get_dataset(name[1], join(rootdir, name[1]), dset, 
                image_size, num_channels, download=download)
        dataset = AddaDataset(src_dataset, tgt_dataset)
    else:
        dataset = get_dataset(name, rootdir, dset, image_size, num_channels,
                download=download)
    if len(dataset) == 0:
        return None
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, 
            shuffle=is_train, **kwargs)
    return loader

def get_transform_dataset(dataset_name, rootdir, net_transform, downscale):
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    transform, target_transform = get_transform2(dataset_name, net_transform, downscale)
    return get_fcn_dataset(dataset_name, rootdir, transform=transform,
            target_transform=target_transform)

sizes = {'cityscapes': 1024, 'gta5': 1024, 'cyclegta5': 1024}
def get_orig_size(dataset_name):
    "Size of images in the dataset for relative scaling."
    try:
        return sizes[dataset_name]
    except:
        raise Exception('Unknown dataset size:', dataset_name)

def get_transform2(dataset_name, net_transform, downscale):
    "Returns image and label transform to downscale, crop and prepare for net."
    orig_size = get_orig_size(dataset_name)
    transform = []
    target_transform = []
    if downscale is not None:
        transform.append(transforms.Resize(orig_size // downscale))
        target_transform.append(
                transforms.Resize(orig_size // downscale,
                    interpolation=Image.NEAREST))
    transform.extend([transforms.Resize(orig_size), net_transform]) 
    target_transform.extend([transforms.Resize(orig_size, interpolation=Image.NEAREST),
        to_tensor_raw]) 
    transform = transforms.Compose(transform)
    target_transform = transforms.Compose(target_transform)
    return transform, target_transform



def get_transform(params, image_size, num_channels):
    # Transforms for PIL Images: Gray <-> RGB
    Gray2RGB = transforms.Lambda(lambda x: x.convert('RGB'))
    RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))

    transform = []
    # Does size request match original size?
    if not image_size == params.image_size:
        transform.append(transforms.Resize(image_size))
   
    # Does number of channels requested match original?
    if not num_channels == params.num_channels:
        if num_channels == 1:
            transform.append(RGB2Gray)
        elif num_channels == 3:
            transform.append(Gray2RGB)
        else:
            print('NumChannels should be 1 or 3', num_channels)
            raise Exception

    transform += [transforms.ToTensor(), 
            transforms.Normalize((params.mean,), (params.std,))]

    return transforms.Compose(transform)

def get_target_transform(params):
    transform = params.target_transform
    t_uniform = transforms.Lambda(lambda x: x[:,0] 
            if isinstance(x, (list, np.ndarray)) and len(x) == 2 else x)
    if transform is None:
        return t_uniform
    else:
        return transforms.Compose([transform, t_uniform])

class AddaDataset(data.Dataset):

    def __init__(self, src_data, tgt_data):
        self.src = src_data
        self.tgt = tgt_data

    def __getitem__(self, index):
        ns = len(self.src)
        nt = len(self.tgt)
        xs, ys = self.src[index % ns]
        xt, yt = self.tgt[index % nt]
        return (xs, ys), (xt, yt)

    def __len__(self):
        return min(len(self.src), len(self.tgt))


data_params = {}
def register_data_params(name):
    def decorator(cls):
        data_params[name] = cls
        return cls
    return decorator

dataset_obj = {}
def register_dataset_obj(name):
    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


class DatasetParams(object):
    "Class variables defined."
    num_channels = 1
    image_size   = 16
    mean         = 0.1307
    std          = 0.3081
    num_cls      = 10
    target_transform = None

def get_dataset(name, rootdir, dset, image_size, num_channels, download=True):
    is_train = (dset == 'train')
    print('get dataset:', name, rootdir, dset)
    params = data_params[name] 
    transform = get_transform(params, image_size, num_channels)
    target_transform = get_target_transform(params)
    return dataset_obj[name](rootdir, train=is_train, transform=transform,
            target_transform=target_transform, download=download)

def get_fcn_dataset(name, rootdir, **kwargs):
    return dataset_obj[name](rootdir, **kwargs)
