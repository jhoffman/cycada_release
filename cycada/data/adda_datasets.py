import os.path

from PIL import Image
import torch.utils.data

from .data_loader import get_transform_dataset
from ..util import to_tensor_raw
from ..transforms import RandomCrop
from ..transforms import augment_collate

class AddaDataLoader(object):
    def __init__(self, net_transform, dataset, rootdir, downscale, crop_size=None, 
            batch_size=1, shuffle=False, num_workers=2, half_crop=None):
        self.dataset = dataset
        self.downscale = downscale
        self.crop_size = crop_size
        self.half_crop = half_crop
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        assert len(self.dataset)==2, 'Requires two datasets: source, target'
        sourcedir = os.path.join(rootdir, self.dataset[0])
        targetdir = os.path.join(rootdir, self.dataset[1])
        self.source = get_transform_dataset(self.dataset[0], sourcedir, 
                net_transform, downscale) 
        self.target = get_transform_dataset(self.dataset[1], targetdir, 
                net_transform, downscale)
        print('Source length:', len(self.source), 'Target length:', len(self.target))
        self.n = max(len(self.source), len(self.target)) # make sure you see all images
        self.num = 0
        self.set_loader_src()
        self.set_loader_tgt()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num % len(self.iters_src) == 0:
            print('restarting source dataset')
            self.set_loader_src()
        if self.num % len(self.iters_tgt) == 0:
            print('restarting target dataset')
            self.set_loader_tgt()

        img_src, label_src = next(self.iters_src)
        img_tgt, label_tgt = next(self.iters_tgt)
            
        self.num += 1
        return img_src, img_tgt, label_src, label_tgt


    def __len__(self):
        return min(len(self.source), len(self.target))

    def set_loader_src(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        if self.crop_size is not None:
            collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
                    halfcrop=self.half_crop, flip=True)
        else:
            collate_fn=torch.utils.data.dataloader.default_collate
        self.loader_src = torch.utils.data.DataLoader(self.source, 
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True)
        self.iters_src = iter(self.loader_src)


    def set_loader_tgt(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        if self.crop_size is not None:
            collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
                    halfcrop=self.half_crop, flip=True)
        else:
            collate_fn=torch.utils.data.dataloader.default_collate
        self.loader_tgt = torch.utils.data.DataLoader(self.target, 
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True)
        self.iters_tgt = iter(self.loader_tgt)



