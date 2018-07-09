import os.path
import sys 

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams

ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']


def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

@register_data_params('cityscapes')
class CityScapesParams(DatasetParams):
    num_channels = 3
    image_size   = 1024
    mean         = 0.5
    std          = 0.5
    num_cls      = 19
    target_transform = None


@register_dataset_obj('cityscapes')
class Cityscapes(data.Dataset):

    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        sys.path.append(root)
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.num_cls = 19
        
        self.id2label = id2label
        self.classes = classes

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        ids = []
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    ids.append('_'.join(filename.split('_')[:3]))
        return ids

    def img_path(self, id):
        fmt = 'leftImg8bit/{}/{}/{}_leftImg8bit.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def label_path(self, id):
        fmt = 'gtFine/{}/{}/{}_gtFine_labelIds.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        id = self.ids[index]
        img = Image.open(self.img_path(id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(self.label_path(id)).convert('L')
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(np.uint8(target), 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    cs = Cityscapes('/x/CityScapes')
