# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.utils import deprecated_arg

from basicmi.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DSUNet(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.sides = nn.ModuleList()
        self.neck: nn.Module

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ):
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]
            
            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            self.downs.append(down)

            if len(channels) > 2:
                _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                self.neck = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]
            
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            self.ups.append(up)
            if not is_top:
                self.sides.append(self._get_side_layer(outc, 2**(len(self.strides) - len(strides))))

            # return self._get_connection_block(down, up, subblock)

        # self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)
        _create_block(in_channels, out_channels, self.channels, self.strides, True)

    # def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
    #     """
    #     Returns the block object defining a layer of the UNet structure including the implementation of the skip
    #     between encoding (down) and decoding (up) sides of the network.

    #     Args:
    #         down_path: encoding half of the layer
    #         up_path: decoding half of the layer
    #         subblock: block defining the next layer in the network.
    #     Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
    #     """
    #     return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def _get_side_layer(self, in_channels: int, strides: int) -> nn.Module:
        return self._get_up_layer(in_channels, self.out_channels, strides, True)

    def cal_losses(self, x, y, criterions, suffix=""):
        loss_dict = {}
        for criterion in criterions:
            loss_dict.update(criterion(x, y, return_dict=True, suffix=suffix))
        return loss_dict

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, criterions = None):
        if self.training:
            loss_dict = {}
            idx = len(self.ups) - 1
        encs = []
        for down in self.downs:
            x = down(x)
            encs.append(x)
        
        x = self.neck(x)
        
        # prediction head
        up, side = self.ups[0], self.sides[0]
        x = up(torch.cat([encs.pop(), x], dim=1))
        out = side(x)
        if self.training:
            loss_dict.update(self.cal_losses(out, y, criterions, suffix=str(idx)))
            idx -= 1

        # residual parts
        for up, side in zip(self.ups[1:], self.sides[1:]):
            x = up(torch.cat([encs.pop(), x], dim=1))
            residual = side(x)
            out = out + residual
            if self.training:
                loss_dict.update(self.cal_losses(out, y, criterions, suffix=str(idx)))
                idx -= 1
        
        residual = self.ups[-1](torch.cat([encs.pop(), x], dim=1))
        out = out + residual
        if self.training:
            loss_dict.update(self.cal_losses(out, y, criterions))
            return loss_dict

        return out

# if __name__ == "__main__":
#     net = DSUNet(3,1,19,channels=[32, 64, 128, 256, 512], strides=[2, 2, 2, 2], num_res_units=2, act="LEAKYRELU", norm="INSTANCE")
#     net.train()
#     x = torch.randn(1,1,96,96,96)
#     outs = net(x)
#     for i, out in enumerate(outs):
#         print(i, out.shape)
