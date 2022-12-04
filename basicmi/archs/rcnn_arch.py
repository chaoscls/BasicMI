import torch
import torch.nn as nn

from basicmi.utils.registry import ARCH_REGISTRY
from basicmi.archs.unet_arch import UNet

@ARCH_REGISTRY.register()
class RCNN(nn.Module):

    def __init__(self, channels, step_num):
        super().__init__()
        self.step_num = step_num
        if step_num > 0:
            self.rcnn = nn.Sequential(
                nn.InstanceNorm3d(channels),
                nn.LeakyReLU(),
                nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(channels),
                nn.LeakyReLU(),
                nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x, multi_ret=False):
        ret = [x]
        for _ in range(self.step_num):
            x = x + self.rcnn(x)
            ret.append(x)
        return ret if multi_ret else x

@ARCH_REGISTRY.register()
class UNetRCNN(nn.Module):

    def __init__(self, step_num, unet_load_path=None, fix_unet=False, **kwargs):
        super().__init__()
        self.unet = UNet(**kwargs)
        if unet_load_path:
            self.unet.load_state_dict(torch.load(unet_load_path, map_location=lambda storage, loc: storage)["params"])
        if fix_unet:
            for _, param in self.unet.named_parameters():
                param.requires_grad = False
        self.rcnn = RCNN(kwargs['out_channels'], step_num)
    
    def forward(self, x, multi_ret=False):
        x = self.unet(x)
        return self.rcnn(x, multi_ret)
