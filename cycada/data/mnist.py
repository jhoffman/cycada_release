from torchvision import datasets
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

@register_data_params('mnist')
class MNIStParams(DatasetParams):
    
    num_channels = 1
    image_size   = 28
    #mean         = 0.1307
    #std          = 0.3081
    mean = 0.5
    std = 0.5
    num_cls      = 10

@register_dataset_obj('mnist')
class MNIST(datasets.MNIST):
    def __init__(self, root, train=True,
            transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, train=train, transform=transform,
                target_transform=target_transform, download=download)
