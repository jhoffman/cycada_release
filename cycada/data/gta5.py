import os.path

from .cityscapes import remap_labels_to_train_ids
import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams
from .cityscapes import id2label as LABEL2TRAIN


@register_data_params('gta5')
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
        m = scipy.io.loadmat(os.path.join(self.root, 'mapping.mat'))
        full_classes = [x[0] for x in m['classes'][0]]
        self.classes = []
        for old_id, new_id in LABEL2TRAIN.items():
            if not new_id == 255 and old_id > 0:
                self.classes.append(full_classes[old_id])
        self.num_cls = 19

    
    def collect_ids(self):
        splits = scipy.io.loadmat(os.path.join(self.root, 'split.mat'))
        ids = splits['{}Ids'.format(self.split)].squeeze()
        return ids

    def img_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'images', filename)

    def label_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'labels', filename)

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

