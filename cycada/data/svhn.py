from torchvision import datasets, transforms
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

@register_data_params('svhn')
class SVHNParams(DatasetParams):
    
    num_channels = 3
    image_size   = 32
    mean         = 0.5
    std          = 0.5
    num_cls      = 10
    # converts 10->0 for svhn so label space [0-9]
    target_transform = transforms.Lambda(lambda x: int(x) % 10) 
    
@register_dataset_obj('svhn')
class SVHN(datasets.SVHN):
    def __init__(self, root, train=True,
            transform=None, target_transform=None, download=False):
        if train:
            split = 'train'
        else:
            split = 'test'
        super(SVHN, self).__init__(root, split=split, transform=transform,
                target_transform=target_transform, download=download)
