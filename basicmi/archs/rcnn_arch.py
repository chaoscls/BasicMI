import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
from monai.networks.layers.factories import Act, Norm

from basicmi.utils.registry import ARCH_REGISTRY
from basicmi.archs.unet_arch import UNet

@ARCH_REGISTRY.register()
class RCNN(nn.Module):

    def __init__(self, channels, step_num):
        super().__init__()
        self.rcnn = nn.Sequential(
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.step_num = step_num

    def forward(self, x, multi_ret=False):
        ret = [x]
        for _ in range(self.step_num):
            x = x + self.rcnn(x)
            ret.append(x)
        return ret if multi_ret else x

@ARCH_REGISTRY.register()
class UNetRCNN(nn.Module):

    def __init__(self, step_num, **kwargs):
        super().__init__()
        self.unet = UNet(**kwargs)
        self.rcnn = RCNN(kwargs['out_channels'], step_num)
    
    def forward(self, x, multi_ret=False):
        x = self.unet(x)
        return self.rcnn(x, multi_ret)
