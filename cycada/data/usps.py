import gzip
import os.path

from urllib.parse import urljoin
import numpy as np
from PIL import Image
from torch.utils import data

# Within package imports
from .data_loader import register_dataset_obj, register_data_params
from . import util
from .data_loader import DatasetParams

@register_data_params('usps')
class USPSParams(DatasetParams):
    
    num_channels = 1
    image_size   = 16
    #mean = 0.1307
    #std = 0.30
    #mean         = 0.254
    #std          = 0.369
    mean = 0.5
    std = 0.5
    num_cls      = 10

@register_dataset_obj('usps')
class USPS(data.Dataset):

    """USPS handwritten digits.
    Homepage: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    params = USPSParams()

    def __init__(self, root, train=True, transform=None, target_transform=None,
            download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
	
        if download:
            self.download()

        if self.train:
            datapath = os.path.join(self.root, self.data_files['train'])
        else:
            datapath = os.path.join(self.root, self.data_files['test'])

        self.images, self.targets = self.read_data(datapath)
    
    def get_path(self, filename):
        return os.path.join(self.root, filename)

    def download(self):
        data_dir = self.root
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def read_data(self, path):
        images = []
        targets = []
        with gzip.GzipFile(path, 'r') as f:
            for line in f:
                split = line.strip().split()
                label = int(float(split[0]))
                pixels = np.array([(float(x) + 1) / 2 for x in split[1:]]) * 255
                num_pix = self.params.image_size
                pixels = pixels.reshape(num_pix, num_pix).astype('uint8')
                img = Image.fromarray(pixels, mode='L')
                images.append(img)
                targets.append(label)
        return images, targets

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)
