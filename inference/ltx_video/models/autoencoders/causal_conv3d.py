# Source Generated with Decompyle++
# File: causal_conv3d.pyc (Python 3.10)

from typing import Tuple, Union
import torch
from torch.nn import nn

class CausalConv3d(nn.Module):
    
    def __init__(self = None, in_channels = None, out_channels = None, kernel_size = None, stride = None, dilation = None, groups = None, spatial_padding_mode = None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]
        dilation = (dilation, 1, 1)
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, dilation, padding, spatial_padding_mode, groups, **('stride', 'dilation', 'padding', 'padding_mode', 'groups'))

    
    def forward(self = None, x = None, causal = None):
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
            x = torch.concatenate((first_frame_pad, x), 2, **('dim',))
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            last_frame_pad = x[:, :, -1:, :, :].repeat((1, 1, (self.time_kernel_size - 1) // 2, 1, 1))
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), 2, **('dim',))
        x = self.conv(x)
        return x

    
    def weight(self):
        return self.conv.weight

    weight = property(weight)

