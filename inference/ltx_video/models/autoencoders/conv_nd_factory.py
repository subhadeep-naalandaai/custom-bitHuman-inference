# Source Generated with Decompyle++
# File: conv_nd_factory.pyc (Python 3.10)

from typing import Tuple, Union
import torch
from inference.ltx_video.models.autoencoders.dual_conv3d import DualConv3d
from inference.ltx_video.models.autoencoders.causal_conv3d import CausalConv3d

def make_conv_nd(dims, in_channels, out_channels, kernel_size, stride, padding, dilation, groups = None, bias = None, causal = None, spatial_padding_mode = (1, 0, 1, 1, True, False, 'zeros', 'zeros'), temporal_padding_mode = ('dims', Union[(int, Tuple[(int, int)])], 'in_channels', int, 'out_channels', int, 'kernel_size', int)):
    if not spatial_padding_mode == temporal_padding_mode and causal:
        raise NotImplementedError('spatial and temporal padding modes must be equal')
    if dims == 2:
        return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, spatial_padding_mode, **('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode'))
    if dims == 3:
        if causal:
            return CausalConv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, spatial_padding_mode, **('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'spatial_padding_mode'))
        return torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, spatial_padding_mode, **('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode'))
    if dims == (2, 1):
        return DualConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, spatial_padding_mode, **('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'padding_mode'))
    raise ValueError(f'''unsupported dimensions: {dims}''')


def make_linear_nd(dims = None, in_channels = None, out_channels = None, bias = (True,)):
    if dims == 2:
        return torch.nn.Conv2d(in_channels, out_channels, 1, bias, **('in_channels', 'out_channels', 'kernel_size', 'bias'))
    if dims == 3 or dims == (2, 1):
        return torch.nn.Conv3d(in_channels, out_channels, 1, bias, **('in_channels', 'out_channels', 'kernel_size', 'bias'))
    raise ValueError(f'''unsupported dimensions: {dims}''')

