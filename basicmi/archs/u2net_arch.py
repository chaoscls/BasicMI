import torch
import torch.nn as nn

import math

from monai.networks.layers.factories import Conv, Norm, Pool

from basicmi.utils.registry import ARCH_REGISTRY

__all__ = ['U2NET_full', 'U2NET_lite']


def _upsample_like(x, size):
    mode = {2: 'bilinear', 3: 'trilinear'}
    return nn.Upsample(size=size, mode=mode[len(size)], align_corners=False)(x)


def _size_map(x, spatial_dims, height):
    # {height: size} for Upsample
    size = list(x.shape[-spatial_dims:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, spatial_dims, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = Conv['conv', spatial_dims](in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = Norm['INSTANCE', spatial_dims](out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, spatial_dims, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self.spatial_dims = spatial_dims
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.spatial_dims, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(self.spatial_dims, in_ch, out_ch))
        self.add_module('downsample', Pool['MAX', self.spatial_dims](2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(self.spatial_dims, out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(self.spatial_dims, mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(self.spatial_dims, mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(self.spatial_dims, mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(self.spatial_dims, mid_ch, mid_ch, dilate=dilate))


class U2NET(nn.Module):
    def __init__(self, spatial_dims, cfgs, out_ch):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.spatial_dims, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            # return [torch.sigmoid(x) for x in maps]
            return maps

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', Pool['MAX', self.spatial_dims](2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(self.spatial_dims, v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', Conv['conv', self.spatial_dims](v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', Conv['conv', self.spatial_dims](int(self.height * self.out_ch), self.out_ch, 1))


@ARCH_REGISTRY.register()
def U2NET_full(spatial_dims, in_ch, out_ch):
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, in_ch, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(spatial_dims, cfgs=full, out_ch=out_ch)


@ARCH_REGISTRY.register()
def U2NET_lite(spatial_dims, in_ch, out_ch):
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, in_ch, 16, 64), -1],
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(spatial_dims, cfgs=lite, out_ch=out_ch)

if __name__ == '__main__':
    X = torch.randn(1, 1, 96, 160, 160).cuda()
    net = U2NET_lite(3, 1, 19).cuda()
    # print(net)
    net.train()
    # from thop import profile
    y_arr = net(X)
    for y in y_arr:
        print(y.shape)
    # flops, params = profile(net, (X,))
    # print("flops:", flops, "params:", params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

