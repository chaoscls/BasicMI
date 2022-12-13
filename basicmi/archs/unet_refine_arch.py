import torch
import torch.nn as nn

from basicmi.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RefineNet(nn.Module):
    
    def __init__(self, unet_opt, refine_net_opt):
        super().__init__()
        self.unet = ARCH_REGISTRY.get("UNet")(**unet_opt)
        self.refine_net = ARCH_REGISTRY.get("UNet")(**refine_net_opt)
        self.add_image = (refine_net_opt["in_channels"] == unet_opt["out_channels"] + 1)

    def forward(self, x, refine_step=1, return_mid=False):
        out = self.unet(x)
        outs = [out]
        for _ in range(refine_step):
            if self.add_image:
                inp = torch.cat([x, out], dim=1)
            # refine net is served as a residual function
            out = out + self.refine_net(inp)
            outs.append(out)
        return outs if return_mid else out
