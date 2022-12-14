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

import torch
import torch.nn.functional as F
from torch import nn

from monai.utils.type_conversion import convert_to_dst_type
from monai.networks import one_hot

from basicmi.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """
    Build a Pytorch version of the SSIM loss function based on the original formula of SSIM

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, spatial_dims: int = 2, sigmoid: bool = True, to_onehot_y: bool = True):
        """
        Args:
            win_size: gaussian weighting window size
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            spatial_dims: if 2, input shape is expected to be (B,C,H,W). if 3, it is expected to be (B,C,H,W,D)
            sigmoid: if True, apply a sigmoid function to the prediction.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims
        self.register_buffer(
            "w", torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims
        )
        self.cov_norm = (win_size**2) / (win_size**2 - 1)
        self.sigmoid = sigmoid
        self.to_onehot_y = to_onehot_y

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range = None, return_dict = False, suffix=""):
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D and pseudo-3D data,
                and (B,C,W,H,D) for 3D data,
            y: second sample (e.g., the reconstructed image). It has similar shape as x.
            data_range: dynamic range of the data

        Returns:
            1-ssim_value (recall this is meant to be a loss function)

        Example:
            .. code-block:: python

                import torch

                # 2D data
                x = torch.ones([1,1,10,10])/2
                y = torch.ones([1,1,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # pseudo-3D data
                x = torch.ones([1,5,10,10])/2  # 5 could represent number of slices
                y = torch.ones([1,5,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=2)(x,y,data_range))

                # 3D data
                x = torch.ones([1,1,10,10,10])/2
                y = torch.ones([1,1,10,10,10])/2
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(1-SSIMLoss(spatial_dims=3)(x,y,data_range))
        """
        n_pred_ch = x.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y = one_hot(y, num_classes=n_pred_ch)
        # accelerate?
        _,_,H,W,D = x.shape
        x, y = x.reshape(-1,1,H,W,D), y.reshape(-1,1,H,W,D)
        # only calculate effective region
        x = x * y
        if data_range == None:
            data_range = y.max().unsqueeze(0)
        if self.sigmoid:
            x = torch.sigmoid(x)
        # if x.shape[1] > 1:  # handling multiple channels (C>1)
        #     if x.shape[1] != y.shape[1]:
        #         raise ValueError(
        #             f"x and y should have the same number of channels, "
        #             f"but x has {x.shape[1]} channels and y has {y.shape[1]} channels."
        #         )
        #     losses = torch.stack(
        #         [
        #             SSIMLoss(self.win_size, self.k1, self.k2, self.spatial_dims)(
        #                 x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1), data_range
        #             )
        #             for i in range(x.shape[1])
        #         ]
        #     )
        #     channel_wise_loss: torch.Tensor = losses.mean()
        #     if return_dict:
        #         return {"ssim": channel_wise_loss}
        #     return channel_wise_loss

        data_range = data_range[(None,) * (self.spatial_dims + 2)]
        # determine whether to work with 2D convolution or 3D
        conv = getattr(F, f"conv{self.spatial_dims}d")
        w = convert_to_dst_type(src=self.w, dst=x)[0]

        c1 = (self.k1 * data_range) ** 2  # stability constant for luminance
        c2 = (self.k2 * data_range) ** 2  # stability constant for contrast
        ux = conv(x, w)  # mu_x
        uy = conv(y, w)  # mu_y
        uxx = conv(x * x, w)  # mu_x^2
        uyy = conv(y * y, w)  # mu_y^2
        uxy = conv(x * y, w)  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denom = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        ssim_value = numerator / denom
        loss: torch.Tensor = 1 - ssim_value.mean()
        if return_dict:
            return {"ssim"+suffix: loss}
        return loss

@LOSS_REGISTRY.register()
class StructureLoss(nn.Module):

    def __init__(self, win_size: int = 7, k: float = 0.03, spatial_dims: int = 2, sigmoid: bool = True, to_onehot_y: bool = True):
        """
        Args:
            win_size: gaussian weighting window size
            k1: stability constant used in the luminance denominator
            k2: stability constant used in the contrast denominator
            spatial_dims: if 2, input shape is expected to be (B,C,H,W). if 3, it is expected to be (B,C,H,W,D)
            sigmoid: if True, apply a sigmoid function to the prediction.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        """
        super().__init__()
        self.win_size = win_size
        self.k = k
        self.spatial_dims = spatial_dims
        self.register_buffer(
            "w", torch.ones([1, 1] + [win_size for _ in range(spatial_dims)]) / win_size**spatial_dims
        )
        self.cov_norm = (win_size**2) / (win_size**2 - 1)
        self.sigmoid = sigmoid
        self.to_onehot_y = to_onehot_y

    def forward(self, x: torch.Tensor, y: torch.Tensor, data_range = None, return_dict = False, suffix=""):
        
        n_pred_ch = x.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y = one_hot(y, num_classes=n_pred_ch)
        # accelerate?
        _,_,H,W,D = x.shape
        x, y = x.reshape(-1,1,H,W,D), y.reshape(-1,1,H,W,D)
        # only calculate effective region
        x = x * y
        if data_range == None:
            data_range = y.max().unsqueeze(0)
        if self.sigmoid:
            x = torch.sigmoid(x)
        # if x.shape[1] > 1:  # handling multiple channels (C>1)
        #     if x.shape[1] != y.shape[1]:
        #         raise ValueError(
        #             f"x and y should have the same number of channels, "
        #             f"but x has {x.shape[1]} channels and y has {y.shape[1]} channels."
        #         )
        #     losses = torch.stack(
        #         [
        #             StructureLoss(self.win_size, self.k, self.spatial_dims)(
        #                 x[:, i, ...].unsqueeze(1), y[:, i, ...].unsqueeze(1), data_range
        #             )
        #             for i in range(x.shape[1])
        #         ]
        #     )
        #     channel_wise_loss: torch.Tensor = losses.mean()
        #     if return_dict:
        #         return {"ssim": channel_wise_loss}
        #     return channel_wise_loss

        data_range = data_range[(None,) * (self.spatial_dims + 2)]
        # determine whether to work with 2D convolution or 3D
        conv = getattr(F, f"conv{self.spatial_dims}d")
        w = convert_to_dst_type(src=self.w, dst=x)[0]

        c = (self.k * data_range) ** 2 / 2
        ux = conv(x, w)  # mu_x
        uy = conv(y, w)  # mu_y
        uxx = conv(x * x, w)  # mu_x^2
        uyy = conv(y * y, w)  # mu_y^2
        uxy = conv(x * y, w)  # mu_xy
        vx = self.cov_norm * (uxx - ux * ux)  # sigma_x
        vy = self.cov_norm * (uyy - uy * uy)  # sigma_y
        vxy = self.cov_norm * (uxy - ux * uy)  # sigma_xy

        s_value = (vxy + c) / (vx * vy + c)

        loss: torch.Tensor = 1 - s_value.mean()
        if return_dict:
            return {"struct"+suffix: loss}
        return loss
