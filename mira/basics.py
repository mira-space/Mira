# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import torch.nn as nn

from utils.utils import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNormSpecific(num_groups, channels)


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}
    
import torch.nn.functional as F
import math
class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.dilation = dilation
        if isinstance(dilation, tuple) and isinstance(dilation[0],float):
            dilation = (math.ceil(dilation[0]), math.ceil(dilation[1]))
            self.fractional = True
        elif isinstance(dilation, float):
            dilation = math.ceil(dilation)
            self.fractional = True
        else:
            self.fractional = False
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv = conv

    def forward(self, x):
        if self.fractional:
            if isinstance(self.dilation, tuple):
                dilation_max1, dilation_max2 = math.ceil(self.dilation[0]), math.ceil(self.dilation[1])
                ratio1, ratio2 = dilation_max1 / self.dilation[0], dilation_max2 / self.dilation[1]
                x = F.interpolate(x, scale_factor=(ratio1, ratio2), mode="bilinear") #
            elif isinstance(self.dilation, int) or isinstance(self.dilation, float):
                dilation_max= math.ceil(self.dilation)
                ratio = dilation_max / self.dilation
                x = F.interpolate(x, scale_factor=ratio, mode="bilinear")
        x = self.conv(x)
        if self.fractional:
            if isinstance(self.dilation, tuple):
                x = F.interpolate(x, scale_factor=(1/ratio1, 1/ratio2), mode="bilinear")
            elif isinstance(self.dilation, int) or isinstance(self.dilation, float):
                x = F.interpolate(x, scale_factor=1/ratio, mode="bilinear")
        return x

def make_2d_dilate_conv(oriconv, *args, kernel_size=3, stride=1, padding=1, dilation=1):
    conv = DilatedConv2d(*args, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    conv.conv.weight = oriconv.weight
    conv.conv.bias = oriconv.bias
    return conv
