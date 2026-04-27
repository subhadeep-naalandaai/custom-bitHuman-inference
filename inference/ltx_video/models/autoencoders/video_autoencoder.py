# Source Generated with Decompyle++
# File: video_autoencoder.pyc (Python 3.10)

import json
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple, Union
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional
from diffusers.utils import logging
from inference.ltx_video.utils.torch_utils import Identity
from inference.ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd, make_linear_nd
from inference.ltx_video.models.autoencoders.pixel_norm import PixelNorm
from inference.ltx_video.models.autoencoders.vae import AutoencoderKLWrapper
logger = logging.get_logger(__name__)

class VideoAutoencoder(AutoencoderKLWrapper):
    
    def from_pretrained(cls = None, pretrained_model_name_or_path = None, *args, **kwargs):
        config_local_path = pretrained_model_name_or_path / 'config.json'
    # WARNING: Decompyle incomplete

    from_pretrained = classmethod(from_pretrained)
    
    def from_config(config):
        pass
    # WARNING: Decompyle incomplete

    from_config = staticmethod(from_config)
    
    def config(self):
        # NOTE: pycdc could not fully decompile this constructor call.
        # It builds a config dict with these keys from encoder/decoder state.
        block_out_channels = [self.encoder.down_blocks[i].res_blocks[-1].conv1.out_channels for i in range(len(self.encoder.down_blocks))]
        return dict(
            block_out_channels=block_out_channels,
            scaling_factor=1,
            norm_layer=self.encoder.norm_layer,
            patch_size=self.encoder.patch_size,
            latent_log_var=self.encoder.latent_log_var,
            use_quant_conv=self.use_quant_conv,
            patch_size_t=self.encoder.patch_size_t,
            add_channel_padding=self.encoder.add_channel_padding,
        )

    config = property(config)
    
    def is_video_supported(self):
        '''
        Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
        '''
        return self.dims != 2

    is_video_supported = property(is_video_supported)
    
    def downscale_factor(self):
        return self.encoder.downsample_factor

    downscale_factor = property(downscale_factor)
    
    def to_json_string(self = None):
        import json
        return json.dumps(self.config.__dict__)

    
    def load_state_dict(self = None, state_dict = None, strict = None):
        model_keys = set(name for name, _ in self.named_parameters())
        key_mapping = {
            '.resnets.': '.res_blocks.',
            'downsamplers.0': 'downsample',
            'upsamplers.0': 'upsample' }
        converted_state_dict = { }
        for key, value in state_dict.items():
            for k, v in key_mapping.items():
                key = key.replace(k, v)
            if 'norm' in key and key not in model_keys:
                logger.info(f'''Removing key {key} from state_dict as it is not present in the model''')
                continue
            converted_state_dict[key] = value
        super().load_state_dict(converted_state_dict, strict, **('strict',))

    
    def last_layer(self):
        if hasattr(self.decoder, 'conv_out'):
            if isinstance(self.decoder.conv_out, nn.Sequential):
                last_layer = self.decoder.conv_out[-1]
                return last_layer
            last_layer = self.decoder.conv_out
            return last_layer
        last_layer = self.decoder.layers[-1]
        return last_layer


class Encoder(nn.Module):
    '''
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, or `none`.
    '''
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, block_out_channels = None, layers_per_block = None, norm_num_groups = None, patch_size = None, norm_layer = None, latent_log_var = None, patch_size_t = None, add_channel_padding = (3, 3, 3, (64,), 2, 32, 1, 'group_norm', 'per_channel', None, False)):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        if add_channel_padding:
            in_channels = in_channels * self.patch_size ** 3
        else:
            in_channels = in_channels * self.patch_size_t * self.patch_size ** 2
        self.in_channels = in_channels
        output_channel = block_out_channels[0]
        self.conv_in = make_conv_nd(dims, in_channels, output_channel, 3, 1, 1, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'))
        self.down_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if not is_final_block:
                pass
            down_block = DownEncoderBlock3D(dims, input_channel, output_channel, self.layers_per_block, 2 ** i >= patch_size, 1e-06, 0, norm_num_groups, norm_layer, **('dims', 'in_channels', 'out_channels', 'num_layers', 'add_downsample', 'resnet_eps', 'downsample_padding', 'resnet_groups', 'norm_layer'))
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock3D(dims, block_out_channels[-1], self.layers_per_block, 1e-06, norm_num_groups, norm_layer, **('dims', 'in_channels', 'num_layers', 'resnet_eps', 'resnet_groups', 'norm_layer'))
        if norm_layer == 'group_norm':
            self.conv_norm_out = nn.GroupNorm(block_out_channels[-1], norm_num_groups, 1e-06, **('num_channels', 'num_groups', 'eps'))
        elif norm_layer == 'pixel_norm':
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        conv_out_channels = out_channels
        if latent_log_var == 'per_channel':
            conv_out_channels *= 2
        elif latent_log_var == 'uniform':
            conv_out_channels += 1
        elif latent_log_var != 'none':
            raise ValueError(f'''Invalid latent_log_var: {latent_log_var}''')
        self.conv_out = make_conv_nd(dims, block_out_channels[-1], conv_out_channels, 3, 1, **('padding',))
        self.gradient_checkpointing = False

    
    def downscale_factor(self):
        return 2 ** len([ block for block in self.down_blocks if isinstance(block.downsample, Downsample3D) ]) * self.patch_size

    downscale_factor = property(downscale_factor)
    
    def forward(self = None, sample = None, return_features = None):
        '''The forward method of the `Encoder` class.'''
        downsample_in_time = sample.shape[2] != 1
        patch_size_t = self.patch_size_t if downsample_in_time else 1
        sample = patchify(sample, self.patch_size, patch_size_t, self.add_channel_padding, **('patch_size_hw', 'patch_size_t', 'add_channel_padding'))
        sample = self.conv_in(sample)
        checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, False, **('use_reentrant',)) if self.gradient_checkpointing and self.training else (lambda x: x)
        if return_features:
            features = []
        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(sample, downsample_in_time, **('downsample_in_time',))
            if return_features:
                features.append(sample)
        sample = checkpoint_fn(self.mid_block)(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if self.latent_log_var == 'uniform':
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()
            if num_dims == 4:
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1)
                sample = torch.cat([
                    sample,
                    repeated_last_channel], 1, **('dim',))
            elif num_dims == 5:
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1, 1)
                sample = torch.cat([
                    sample,
                    repeated_last_channel], 1, **('dim',))
            else:
                raise ValueError(f'''Invalid input shape: {sample.shape}''')
        if return_features:
            features.append(sample[:, :self.latent_channels, ...])
            return (sample, features)



class Decoder(nn.Module):
    '''
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
    '''
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, block_out_channels = None, layers_per_block = None, norm_num_groups = None, patch_size = None, norm_layer = None, patch_size_t = None, add_channel_padding = None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        if add_channel_padding:
            out_channels = out_channels * self.patch_size ** 3
        else:
            out_channels = out_channels * self.patch_size_t * self.patch_size ** 2
        self.out_channels = out_channels
        self.conv_in = make_conv_nd(dims, in_channels, block_out_channels[-1], 3, 1, 1, **('kernel_size', 'stride', 'padding'))
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.mid_block = UNetMidBlock3D(dims, block_out_channels[-1], self.layers_per_block, 1e-06, norm_num_groups, norm_layer, **('dims', 'in_channels', 'num_layers', 'resnet_eps', 'resnet_groups', 'norm_layer'))
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if not is_final_block:
                pass
            up_block = UpDecoderBlock3D(dims, self.layers_per_block + 1, prev_output_channel, output_channel, 2 ** (len(block_out_channels) - i - 1) > patch_size, 1e-06, norm_num_groups, norm_layer, **('dims', 'num_layers', 'in_channels', 'out_channels', 'add_upsample', 'resnet_eps', 'resnet_groups', 'norm_layer'))
            self.up_blocks.append(up_block)
        if norm_layer == 'group_norm':
            self.conv_norm_out = nn.GroupNorm(block_out_channels[0], norm_num_groups, 1e-06, **('num_channels', 'num_groups', 'eps'))
        elif norm_layer == 'pixel_norm':
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(dims, block_out_channels[0], out_channels, 3, 1, **('padding',))
        self.gradient_checkpointing = False

    
    def forward(self = None, sample = None, target_shape = None):
        '''The forward method of the `Decoder` class.'''
        pass
    # WARNING: Decompyle incomplete



class DownEncoderBlock3D(nn.Module):
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, dropout = None, num_layers = None, resnet_eps = None, resnet_groups = None, add_downsample = None, downsample_padding = None, norm_layer = None):
        super().__init__()
        res_blocks = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            res_blocks.append(ResnetBlock3D(dims, in_channels, out_channels, resnet_eps, resnet_groups, dropout, norm_layer, **('dims', 'in_channels', 'out_channels', 'eps', 'groups', 'dropout', 'norm_layer')))
        self.res_blocks = nn.ModuleList(res_blocks)
        if add_downsample:
            self.downsample = Downsample3D(dims, out_channels, out_channels, downsample_padding, **('out_channels', 'padding'))
            return None
        self.downsample = None

    
    def forward(self = None, hidden_states = None, downsample_in_time = None):
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)
        hidden_states = self.downsample(hidden_states, downsample_in_time, **('downsample_in_time',))
        return hidden_states



class UNetMidBlock3D(nn.Module):
    '''
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.

    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    '''
    
    def __init__(self = None, dims = None, in_channels = None, dropout = None, num_layers = None, resnet_eps = None, resnet_groups = None, norm_layer = None):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.res_blocks = nn.ModuleList([ResnetBlock3D(dims=dims, in_channels=in_channels, out_channels=in_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, norm_layer=norm_layer) for _ in range(num_layers)])

    
    def forward(self = None, hidden_states = None):
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)
        return hidden_states



class UpDecoderBlock3D(nn.Module):
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, resolution_idx = None, dropout = None, num_layers = None, resnet_eps = None, resnet_groups = None, add_upsample = None, norm_layer = None):
        super().__init__()
        res_blocks = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            res_blocks.append(ResnetBlock3D(dims, input_channels, out_channels, resnet_eps, resnet_groups, dropout, norm_layer, **('dims', 'in_channels', 'out_channels', 'eps', 'groups', 'dropout', 'norm_layer')))
        self.res_blocks = nn.ModuleList(res_blocks)
        if add_upsample:
            self.upsample = Upsample3D(dims, out_channels, out_channels, **('dims', 'channels', 'out_channels'))
        else:
            self.upsample = Identity()
        self.resolution_idx = resolution_idx

    
    def forward(self = None, hidden_states = None, upsample_in_time = None):
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)
        hidden_states = self.upsample(hidden_states, upsample_in_time, **('upsample_in_time',))
        return hidden_states



class ResnetBlock3D(nn.Module):
    '''
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    '''
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, conv_shortcut = None, dropout = None, groups = None, eps = None, norm_layer = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        if norm_layer == 'group_norm':
            self.norm1 = torch.nn.GroupNorm(groups, in_channels, eps, True, **('num_groups', 'num_channels', 'eps', 'affine'))
        elif norm_layer == 'pixel_norm':
            self.norm1 = PixelNorm()
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv_nd(dims, in_channels, out_channels, 3, 1, 1, **('kernel_size', 'stride', 'padding'))
        if norm_layer == 'group_norm':
            self.norm2 = torch.nn.GroupNorm(groups, out_channels, eps, True, **('num_groups', 'num_channels', 'eps', 'affine'))
        elif norm_layer == 'pixel_norm':
            self.norm2 = PixelNorm()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv_nd(dims, out_channels, out_channels, 3, 1, 1, **('kernel_size', 'stride', 'padding'))
        if in_channels != out_channels:
            self.conv_shortcut = make_linear_nd(dims, in_channels, out_channels, **('dims', 'in_channels', 'out_channels'))
            return None
        self.conv_shortcut = nn.Identity()

    
    def forward(self = None, input_tensor = None):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = input_tensor + hidden_states
        return output_tensor



class Downsample3D(nn.Module):
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, kernel_size = None, padding = None):
        super().__init__()
        stride = 2
        self.padding = padding
        self.in_channels = in_channels
        self.dims = dims
        self.conv = make_conv_nd(dims, in_channels, out_channels, kernel_size, stride, padding, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding'))

    
    def forward(self, x, downsample_in_time = (True,)):
        pass  # WARNING: decompile incomplete
