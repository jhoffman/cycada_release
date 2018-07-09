import os.path

import numpy as np
from PIL import Image

from .cityscapes import remap_labels_to_train_ids
from .gta5 import GTA5 #, LABEL2TRAIN

from .data_loader import register_data_params, register_dataset_obj


@register_dataset_obj('cyclegta5')
class CycleGTA5(GTA5):

    def collect_ids(self):
        ids = GTA5.collect_ids(self)
        existing_ids = []
        for id in ids:
            filename = '{:05d}.png'.format(id)
            if os.path.exists(os.path.join(self.root, 'images', filename)):
                existing_ids.append(id)
        return existing_ids

    def __getitem__(self, index):
        id = self.ids[index]
        filename = '{:05d}.png'.format(id)
        img_path = os.path.join(self.root, 'images', filename)
        label_path = os.path.join(self.root, 'labels', filename)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(label_path)
        img = img.resize(target.size, resample=Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            #target = self.label2train(target)
            target = Image.fromarray(target, 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
