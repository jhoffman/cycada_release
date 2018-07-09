from torchvision import datasets, transforms
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

import numpy as np

@register_data_params('svhn_bal')
class SVHNParams(DatasetParams):
    
    num_channels = 3
    image_size   = 32
    mean         = 0.5
    std          = 0.5
    num_cls      = 10
    # converts 10->0 for svhn so label space [0-9]
    target_transform = transforms.Lambda(lambda x: int(x) % 10) 
    
@register_dataset_obj('svhn_bal')
class SVHN(datasets.SVHN):
    def __init__(self, root, train=True,
            transform=None, target_transform=None, download=False):
        if train:
            split = 'train'
        else:
            split = 'test'
        super(SVHN, self).__init__(root, split=split, transform=transform,
                target_transform=target_transform, download=download)

        # Subsample images to balance the training set
       
        if split == 'train':
            # compute the histogram of original label set
            label_set = np.unique(self.labels)
            num_cls = len(label_set)
            count,_ = np.histogram(self.labels.squeeze(), bins=num_cls)
            min_num = min(count)
            
            # subsample
            ind = np.zeros((num_cls, min_num), dtype=int)
            for i in label_set:
                binary_ind = np.where(self.labels.squeeze() == i)[0]
                np.random.shuffle(binary_ind)
                
                ind[i % num_cls,:] = binary_ind[:min_num]
            
            ind = ind.flatten()
            # shuffle 5 times
            for i in range(100):
                np.random.shuffle(ind)
            self.labels = self.labels[ind]
            self.data = self.data[ind]
