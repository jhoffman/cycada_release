import enum
import glob
import os.path

import numpy as np
from numpy.core.fromnumeric import sort
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import (DatasetParams, register_data_params,
                          register_dataset_obj)


def get_class_info():
    classes_csv = '''0,unlabeled,0,0,0,0,255,0
    1,ambiguous,111,74,0,0,255,0
    2,sky,70,130,180,1,0,0
    3,road,128,64,128,1,1,0
    4,sidewalk,244,35,232,1,2,0
    5,railtrack,230,150,140,0,255,0
    6,terrain,152,251,152,1,3,0
    7,tree,87,182,35,1,4,0
    8,vegetation,35,142,35,1,5,0
    9,building,70,70,70,1,6,0
    10,infrastructure,153,153,153,1,7,0
    11,fence,190,153,153,1,8,0
    12,billboard,150,20,20,1,9,0
    13,traffic light,250,170,30,1,10,1
    14,traffic sign,220,220,0,1,11,0
    15,mobilebarrier,180,180,100,1,12,0
    16,firehydrant,173,153,153,1,13,1
    17,chair,168,153,153,1,14,1
    18,trash,81,0,21,1,15,0
    19,trashcan,81,0,81,1,16,1
    20,person,220,20,60,1,17,1
    21,animal,255,0,0,0,255,0
    22,bicycle,119,11,32,0,255,0
    23,motorcycle,0,0,230,1,18,1
    24,car,0,0,142,1,19,1
    25,van,0,80,100,1,20,1
    26,bus,0,60,100,1,21,1
    27,truck,0,0,70,1,22,1
    28,trailer,0,0,90,0,255,0
    29,train,0,80,100,0,255,0
    30,plane,0,100,100,0,255,0
    31,boat,50,0,90,0,255,0'''

    classes = {}

    for line in classes_csv.splitlines():
        col = line.split(',')
        class_info = {}
        class_info['id'] = int(col[0])
        class_info['classname'] = col[1]
        class_info['red'] = int(col[2])
        class_info['green'] = int(col[3])
        class_info['blue'] = int(col[4])
        class_info['class_eval'] = int(col[5])
        class_info['trainid'] = int(col[6])
        class_info['instance_eval'] = int(col[7])
        classes[class_info['classname']] = class_info

    return classes


ignore_label = 255
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
           'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle']
raw_class_info = get_class_info()


def remap_labels_to_train_ids(img):
    target_img = np.full((img.shape[0], img.shape[1]), ignore_label, dtype='uint8')
    for classname in enumerate(classes):
        if classname in raw_class_info.keys():
            class_info = raw_class_info[classname]
            target_img[(img[:, :, 0] == class_info['red']) & (img[:, :, 1] == class_info['green']) & (
                img[:, :, 2] == class_info['blue'])] = class_info['trainid']


@register_data_params('newgta5')
class GTA5Params(DatasetParams):
    num_channels = 3
    image_size = 1024
    mean = 0.5
    std = 0.5
    num_cls = 19
    target_transform = None


@register_dataset_obj('gta5')
class GTA5(data.Dataset):

    def __init__(self, root, num_cls=19, split='train', remap_labels=True,
                 transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.num_cls = 19

    def collect_ids(self):
        img_files = glob(os.path.join(self.root, self.split, 'img', '**', '*'))
        img_files = [file.replace(os.path.join(self.root, self.split, 'img'), '') for file in img_files]
        img_files = sorted(img_files)
        return img_files

    def img_path(self, id):
        return os.path.join(self.root, self.split, 'img', id)

    def label_path(self, id):
        return os.path.join(self.root, self.split, 'cls', id)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_path(id)
        label_path = self.label_path(id)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(label_path)
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)
