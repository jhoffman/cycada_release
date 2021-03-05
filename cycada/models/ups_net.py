import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
from torchvision.models import vgg

from cycada.models.models import register_model
from upsnet.models.resnet_upsnet import resnet_upsnet

# @register_model('fcn8s')
# class VGG16_FCN8s(nn.Module):

#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]),
#         ])

#     def __init__(self, num_cls=19, pretrained=True, weights_init=None, 
#             output_last_ft=False):
#         super().__init__()
#         self.output_last_ft = output_last_ft
#         self.vgg = make_layers(vgg.cfg['D'])
#         self.vgg_head = nn.Sequential(
#             nn.Conv2d(512, 4096, 7),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(4096, 4096, 1),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(4096, num_cls, 1)
#             )
#         self.upscore2 = self.upscore_pool4 = Bilinear(2, num_cls)
#         self.upscore8 = Bilinear(8, num_cls)
#         self.score_pool4 = nn.Conv2d(512, num_cls, 1)
#         for param in self.score_pool4.parameters():
#             init.constant(param, 0)
#         self.score_pool3 = nn.Conv2d(256, num_cls, 1)
#         for param in self.score_pool3.parameters():
#             init.constant(param, 0)
        
#         if pretrained:
#             if weights_init is not None:
#                 self.load_weights(torch.load(weights_init))
#             else:
#                 self.load_base_weights()
 
#     def load_base_vgg(self, weights_state_dict):
#         vgg_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg.')
#         self.vgg.load_state_dict(vgg_state_dict)
     
#     def load_vgg_head(self, weights_state_dict):
#         vgg_head_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg_head.') 
#         self.vgg_head.load_state_dict(vgg_head_state_dict)
    
#     def get_dict_by_prefix(self, weights_state_dict, prefix):
#         return {k[len(prefix):]: v 
#                 for k,v in weights_state_dict.items()
#                 if k.startswith(prefix)}


#     def load_weights(self, weights_state_dict):
#         self.load_base_vgg(weights_state_dict)
#         self.load_vgg_head(weights_state_dict)

#     def split_vgg_head(self):
#         self.classifier = list(self.vgg_head.children())[-1]
#         self.vgg_head_feat = nn.Sequential(*list(self.vgg_head.children())[:-1])


#     def forward(self, x):
#         input = x
#         x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0)
#         intermediates = {}
#         fts_to_save = {16: 'pool3', 23: 'pool4'}
#         for i, module in enumerate(self.vgg):
#             x = module(x)
#             if i in fts_to_save:
#                 intermediates[fts_to_save[i]] = x
       
#         ft_to_save = 5 # Dropout before classifier
#         last_ft = {}
#         for i, module in enumerate(self.vgg_head):
#             x = module(x)
#             if i == ft_to_save:
#                 last_ft = x      
        
#         _, _, h, w = x.size()
#         upscore2 = self.upscore2(x)
#         pool4 = intermediates['pool4']
#         score_pool4 = self.score_pool4(0.01 * pool4)
#         score_pool4c = _crop(score_pool4, upscore2, offset=5)
#         fuse_pool4 = upscore2 + score_pool4c
#         upscore_pool4 = self.upscore_pool4(fuse_pool4)
#         pool3 = intermediates['pool3']
#         score_pool3 = self.score_pool3(0.0001 * pool3)
#         score_pool3c = _crop(score_pool3, upscore_pool4, offset=9)
#         fuse_pool3 = upscore_pool4 + score_pool3c
#         upscore8 = self.upscore8(fuse_pool3)
#         score = _crop(upscore8, input, offset=31)
#         if self.output_last_ft: 
#             return score, last_ft
#         else:
#             return score


#     def load_base_weights(self):
#         """This is complicated because we converted the base model to be fully
#         convolutional, so some surgery needs to happen here."""
#         base_state_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
#         vgg_state_dict = {k[len('features.'):]: v
#                           for k, v in base_state_dict.items()
#                           if k.startswith('features.')}
#         self.vgg.load_state_dict(vgg_state_dict)
#         vgg_head_params = self.vgg_head.parameters()
#         for k, v in base_state_dict.items():
#             if not k.startswith('classifier.'):
#                 continue
#             if k.startswith('classifier.6.'):
#                 # skip final classifier output
#                 continue
#             vgg_head_param = next(vgg_head_params)
#             vgg_head_param.data = v.view(vgg_head_param.size())

@register_model('upsnet')
class UPSNET(resnet_upsnet):

    # def __init__(self):
        # print("Hello! I am the initial guy")
        # pass
    pass

if __name__ == '__main__':

    model = UPSNET([3, 4, 23, 3])