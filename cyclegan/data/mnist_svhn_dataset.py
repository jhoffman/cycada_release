import random
import os.path
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from data.base_dataset import BaseDataset
import scipy.io
import numpy as np

from PIL import Image
from PIL.ImageOps import invert


class MnistSvhnDataset(BaseDataset):
    def name(self):
        return 'MnistSvhnDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print(opt)
        self.mnist = MNIST(os.path.join(opt.dataroot, 'mnist'),
                           train=opt.isTrain, download=True)
        #svhn_mat_extra = scipy.io.loadmat(os.path.join(opt.dataroot,
        #                                               'svhn/extra_32x32.mat'))
        svhn_mat_train = scipy.io.loadmat(os.path.join(opt.dataroot,
                                                       'svhn/train_32x32.mat'))
        #svhn_np = np.concatenate((np.array(svhn_mat_train['X']),
        #                          np.array(svhn_mat_extra['X'])),
        #                         axis=3)
        svhn_np = np.array(svhn_mat_train['X'])
        self.svhn = np.transpose(svhn_np, (3, 0, 1, 2))
        self.svhn_label = np.array(svhn_mat_train['y'])

        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        self.shuffle_indices()

    def shuffle_indices(self):
        self.mnist_indices = list(range(len(self.mnist)))
        self.svhn_indices = list(range(self.svhn.shape[0]))
        print('num mnist', len(self.mnist_indices), 'num svhn', len(self.svhn_indices))
        if not self.opt.serial_batches:
            random.shuffle(self.mnist_indices)
            random.shuffle(self.svhn_indices)

    def __getitem__(self, index):

        Gray2RGB = transforms.Lambda(lambda x: x.convert('RGB'))
        if index == 0:
            self.shuffle_indices()

        A_img, A_label = self.mnist[self.mnist_indices[index % len(self.mnist)]]
        #if random.random() < 0.5: # invert the color with 50% prob
        #    A_img = invert(A_img)
        A_img = A_img.resize((32, 32))
        A_img = A_img.convert('RGB')
        #A_img = np.expand_dims(np.array(A_img), 0)
        #print('mnist after expand dims:', np.array(A_img).shape)
        #A_img = np.transpose(A_img, (1, 2, 0))
        A_img = self.transform(A_img)
        A_path = '%01d_%05d.png' % (A_label, index)

        B_img = self.svhn[self.svhn_indices[index]]
        B_label = self.svhn_label[self.svhn_indices[index % self.svhn.shape[0]]][0] % 10 # 10->0 
        B_img = self.transform(B_img)
        B_path = '%01d_%05d.png' % (B_label, index)

            
        #A_img, B_img = B_img, A_img
        #A_path, B_path = B_path, A_path
        #A_label, B_label = B_label, A_label

        item = {}
        item.update({'A': A_img,
                     'A_paths': A_path,
                     'A_label': A_label
                 })
        
        item.update({'B': B_img,
                     'B_paths': B_path,
                     'B_label': B_label
                 })
        return item
        
    def __len__(self):
        #if self.opt.which_direction == 'AtoB':
        #    return len(self.mnist)
        #else:            
        #    return self.svhn.shape[0]

        return self.svhn.shape[0] #min(len(self.mnist), self.svhn.shape[0])
        
