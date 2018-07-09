import os
from os.path import join
import glob
from PIL import Image

import torch.utils.data as data
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

class CycleGANDataset(data.Dataset):

    def __init__(self, root, regexp, transform=None, target_transform=None, 
            download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.image_paths, self.labels = self.find_images(regexp)

    def find_images(self, regexp='*.png'):
        basenames = sorted(glob.glob(join(self.root, regexp)))
        image_paths = []
        labels = []
        for basename in basenames:
            image_paths.append(os.path.join(self.root, basename))
            labels.append(int(basename.split('/')[-1].split('_')[0]))
        return image_paths, labels

    def __getitem__(self, index):
        im = Image.open(self.image_paths[index]) #.convert('L')
        target = self.labels[index]

        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.image_paths)


@register_dataset_obj('svhn2mnist')
class Svhn2MNIST(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for svhn2mnist.')
            self.image_paths = []
        else:
            super(Svhn2MNIST, self).__init__(root, '*_fake_B.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('svhn2mnist')
class Svhn2MNISTParams(DatasetParams):
    num_channels = 3
    image_size = 32
    mean = 0.5
    std = 0.5
    #mean = 0.1307
    #std = 0.3081
    
    # mean and std (when scaled between [0,1])
    #mean = 0.127 # ep50
    #mean = 0.21 # ep100 -- more white pixels...
    #std = 0.29

    #mean = 0.21
    #std = 0.2
    
    num_cls = 10
    target_transform = None

@register_dataset_obj('usps2mnist')
class Usps2Mnist(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for usps2mnist.')
            self.image_paths = []
        else:
            super(Usps2Mnist, self).__init__(root, '*_fake_A.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('usps2mnist')
class Usps2MnistParams(DatasetParams):
    num_channels = 3
    image_size = 16
    #mean = 0.1307
    #std = 0.3081
    mean = 0.5
    std = 0.5
    num_cls = 10
    target_transform = None


@register_dataset_obj('mnist2usps')
class Mnist2Usps(CycleGANDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, 
            download=False):
        if not train:
            print('No test set for mnist2usps.')
            self.image_paths = []
        else:
            super(Mnist2Usps, self).__init__(root, '*_fake_B.png',
                    transform=transform, target_transform=target_transform, 
                    download=download)

@register_data_params('mnist2usps')
class Mnist2UspsParams(DatasetParams):
    num_channels = 3
    image_size = 16 # this seems wrong...
    #mean = 0.25
    #std = 0.37
    
    #mean = 0.1307
    #std = 0.3081
    mean = 0.5
    std = 0.5
    num_cls = 10
    target_transform = None
