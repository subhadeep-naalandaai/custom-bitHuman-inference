# Source Generated with Decompyle++
# File: causal_video_autoencoder.pyc (Python 3.10)

import json
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple, Union, List
from pathlib import Path
import torch
import numpy as np
from einops import rearrange
from torch import nn
from diffusers.utils import logging
import torch.nn.functional
F = functional
nn
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from safetensors import safe_open
from inference.ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd, make_linear_nd
from inference.ltx_video.models.autoencoders.pixel_norm import PixelNorm
from inference.ltx_video.models.autoencoders.vae import AutoencoderKLWrapper
from inference.ltx_video.models.transformers.attention import Attention
from inference.ltx_video.utils.diffusers_config_mapping import diffusers_and_ours_config_mapping, make_hashable_key, VAE_KEYS_RENAME_DICT
PER_CHANNEL_STATISTICS_PREFIX = 'per_channel_statistics.'
logger = logging.get_logger(__name__)

class CausalVideoAutoencoder(AutoencoderKLWrapper):
    
    def from_pretrained(cls = None, pretrained_model_name_or_path = None, *args, **kwargs):
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
    # WARNING: Decompyle incomplete

    from_pretrained = classmethod(from_pretrained)
    
    def from_config(config):
        pass
    # WARNING: Decompyle incomplete

    from_config = staticmethod(from_config)
    
    def config(self):
        return SimpleNamespace('CausalVideoAutoencoder', self.dims, self.encoder.conv_in.in_channels // self.encoder.patch_size ** 2, self.decoder.conv_out.out_channels // self.decoder.patch_size ** 2, self.decoder.conv_in.in_channels, self.encoder.blocks_desc, self.decoder.blocks_desc, 1, self.encoder.norm_layer, self.encoder.patch_size, self.encoder.latent_log_var, self.use_quant_conv, self.decoder.causal, self.decoder.timestep_conditioning, self.normalize_latent_channels, **('_class_name', 'dims', 'in_channels', 'out_channels', 'latent_channels', 'encoder_blocks', 'decoder_blocks', 'scaling_factor', 'norm_layer', 'patch_size', 'latent_log_var', 'use_quant_conv', 'causal_decoder', 'timestep_conditioning', 'normalize_latent_channels'))

    config = property(config)
    
    def is_video_supported(self):
        '''
        Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
        '''
        return self.dims != 2

    is_video_supported = property(is_video_supported)
    
    def spatial_downscale_factor(self):
        return 2 ** len([block for block in self.encoder.blocks_desc if block[0] in ('compress_space', 'compress_all', 'compress_all_res', 'compress_space_res')]) * self.encoder.patch_size

    spatial_downscale_factor = property(spatial_downscale_factor)
    
    def temporal_downscale_factor(self):
        return 2 ** len([block for block in self.encoder.blocks_desc if block[0] in ('compress_time', 'compress_all', 'compress_all_res', 'compress_space_res')])

    temporal_downscale_factor = property(temporal_downscale_factor)
    
    def to_json_string(self = None):
        import json
        return json.dumps(self.config.__dict__)

    
    def load_state_dict(self = None, state_dict = None, strict = None):
        if any(key.startswith('vae.') for key in state_dict.keys()):
            state_dict = {k[len('vae.'):]: v for k, v in state_dict.items() if k.startswith('vae.')}
        ckpt_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('std-of-means') and not k.startswith('mean-of-means')}
        model_keys = set(name for name, _ in self.named_modules())
        key_mapping = {
            '.resnets.': '.res_blocks.',
            'downsamplers.0': 'downsample',
            'upsamplers.0': 'upsample' }
        converted_state_dict = { }
        for key, value in ckpt_state_dict.items():
            for k, v in key_mapping.items():
                key = key.replace(k, v)
            key_prefix = '.'.join(key.split('.')[:-1])
            if 'norm' in key and key_prefix not in model_keys:
                logger.info(f'''Removing key {key} from state_dict as it is not present in the model''')
                continue
            converted_state_dict[key] = value
        super().load_state_dict(converted_state_dict, strict, **('strict',))
        data_dict = {k: v for k, v in state_dict.items() if k.startswith('std-of-means') or k.startswith('mean-of-means')}
        if len(data_dict) > 0:
            self.register_buffer('std_of_means', data_dict['std-of-means'])
            self.register_buffer('mean_of_means', data_dict.get('mean-of-means', torch.zeros_like(data_dict['std-of-means'])))
            return None

    
    def last_layer(self):
        if hasattr(self.decoder, 'conv_out'):
            if isinstance(self.decoder.conv_out, nn.Sequential):
                last_layer = self.decoder.conv_out[-1]
                return last_layer
            last_layer = self.decoder.conv_out
            return last_layer
        last_layer = self.decoder.layers[-1]
        return last_layer

    
    def set_use_tpu_flash_attention(self):
        for block in self.decoder.up_blocks:
            if isinstance(block, UNetMidBlock3D) and block.attention_blocks:
                for attention_block in block.attention_blocks:
                    attention_block.set_use_tpu_flash_attention()



class Encoder(nn.Module):
    '''
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, `constant` or `none`.
    '''
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, blocks = None, base_channels = None, norm_num_groups = None, patch_size = None, norm_layer = None, latent_log_var = None, spatial_padding_mode = None):
        super().__init__()
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self.blocks_desc = blocks
        in_channels = in_channels * patch_size ** 2
        output_channel = base_channels
        self.conv_in = make_conv_nd(dims, in_channels, output_channel, 3, 1, 1, True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'causal', 'spatial_padding_mode'))
        self.down_blocks = nn.ModuleList([])
        for block_name, block_params in blocks:
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {
                    'num_layers': block_params }
            if block_name == 'res_x':
                block = UNetMidBlock3D(dims, input_channel, block_params['num_layers'], 1e-06, norm_num_groups, norm_layer, spatial_padding_mode, **('dims', 'in_channels', 'num_layers', 'resnet_eps', 'resnet_groups', 'norm_layer', 'spatial_padding_mode'))
            elif block_name == 'res_x_y':
                output_channel = block_params.get('multiplier', 2) * output_channel
                block = ResnetBlock3D(dims, input_channel, output_channel, 1e-06, norm_num_groups, norm_layer, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'eps', 'groups', 'norm_layer', 'spatial_padding_mode'))
            elif block_name == 'compress_time':
                block = make_conv_nd(dims, input_channel, output_channel, 3, (2, 1, 1), True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))
            elif block_name == 'compress_space':
                block = make_conv_nd(dims, input_channel, output_channel, 3, (1, 2, 2), True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))
            elif block_name == 'compress_all':
                block = make_conv_nd(dims, input_channel, output_channel, 3, (2, 2, 2), True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))
            elif block_name == 'compress_all_x_y':
                output_channel = block_params.get('multiplier', 2) * output_channel
                block = make_conv_nd(dims, input_channel, output_channel, 3, (2, 2, 2), True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))
            elif block_name == 'compress_all_res':
                output_channel = block_params.get('multiplier', 2) * output_channel
                block = SpaceToDepthDownsample(dims, input_channel, output_channel, (2, 2, 2), spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'stride', 'spatial_padding_mode'))
            elif block_name == 'compress_space_res':
                output_channel = block_params.get('multiplier', 2) * output_channel
                block = SpaceToDepthDownsample(dims, input_channel, output_channel, (1, 2, 2), spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'stride', 'spatial_padding_mode'))
            elif block_name == 'compress_time_res':
                output_channel = block_params.get('multiplier', 2) * output_channel
                block = SpaceToDepthDownsample(dims, input_channel, output_channel, (2, 1, 1), spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'stride', 'spatial_padding_mode'))
            else:
                raise ValueError(f'''unknown block: {block_name}''')
            self.down_blocks.append(block)
        if norm_layer == 'group_norm':
            self.conv_norm_out = nn.GroupNorm(output_channel, norm_num_groups, 1e-06, **('num_channels', 'num_groups', 'eps'))
        elif norm_layer == 'pixel_norm':
            self.conv_norm_out = PixelNorm()
        elif norm_layer == 'layer_norm':
            self.conv_norm_out = LayerNorm(output_channel, 1e-06, **('eps',))
        self.conv_act = nn.SiLU()
        conv_out_channels = out_channels
        if latent_log_var == 'per_channel':
            conv_out_channels *= 2
        elif latent_log_var == 'uniform':
            conv_out_channels += 1
        elif latent_log_var == 'constant':
            conv_out_channels += 1
        elif latent_log_var != 'none':
            raise ValueError(f'''Invalid latent_log_var: {latent_log_var}''')
        self.conv_out = make_conv_nd(dims, output_channel, conv_out_channels, 3, 1, True, spatial_padding_mode, **('padding', 'causal', 'spatial_padding_mode'))
        self.gradient_checkpointing = False

    
    def forward(self = None, sample = None):
        '''The forward method of the `Encoder` class.'''
        sample = patchify(sample, self.patch_size, 1, **('patch_size_hw', 'patch_size_t'))
        sample = self.conv_in(sample)
        checkpoint_fn = partial(torch.utils.checkpoint.checkpoint, False, **('use_reentrant',)) if self.gradient_checkpointing and self.training else (lambda x: x)
        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(sample)
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
                return sample
            if num_dims == 5:
                repeated_last_channel = last_channel.repeat(1, sample.shape[1] - 2, 1, 1, 1)
                sample = torch.cat([
                    sample,
                    repeated_last_channel], 1, **('dim',))
                return sample
            raise ValueError(f'''Invalid input shape: {sample.shape}''')
        if self.latent_log_var == 'constant':
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = torch.cat([
                sample,
                torch.ones_like(sample, sample.device, **('device',)) * approx_ln_0], 1, **('dim',))
        return sample



class Decoder(nn.Module):
    '''
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        dims (`int` or `Tuple[int, int]`, *optional*, defaults to 3):
            The number of dimensions to use in convolutions.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        blocks (`List[Tuple[str, int]]`, *optional*, defaults to `[("res_x", 1)]`):
            The blocks to use. Each block is a tuple of the block name and the number of layers.
        base_channels (`int`, *optional*, defaults to 128):
            The number of output channels for the first convolutional layer.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal (`bool`, *optional*, defaults to `True`):
            Whether to use causal convolutions or not.
    '''
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, blocks = None, base_channels = None, layers_per_block = None, norm_num_groups = None, patch_size = None, norm_layer = None, causal = None, timestep_conditioning = (3, 3, [
        ('res_x', 1)], 128, 2, 32, 1, 'group_norm', True, False, 'zeros'), spatial_padding_mode = (('in_channels', int, 'out_channels', int, 'blocks', List[Tuple[(str, int | dict)]], 'base_channels', int, 'layers_per_block', int, 'norm_num_groups', int, 'patch_size', int, 'norm_layer', str, 'causal', bool, 'timestep_conditioning', bool, 'spatial_padding_mode', str),)):
        super().__init__()
        self.patch_size = patch_size
        self.layers_per_block = layers_per_block
        out_channels = out_channels * patch_size ** 2
        self.causal = causal
        self.blocks_desc = blocks
        output_channel = base_channels
        for block_name, block_params in list(reversed(blocks)):
            block_params = block_params if isinstance(block_params, dict) else { }
            if block_name == 'res_x_y':
                output_channel = output_channel * block_params.get('multiplier', 2)
            if block_name == 'compress_all':
                output_channel = output_channel * block_params.get('multiplier', 1)
        self.conv_in = make_conv_nd(dims, in_channels, output_channel, 3, 1, 1, True, spatial_padding_mode, **('kernel_size', 'stride', 'padding', 'causal', 'spatial_padding_mode'))
        self.up_blocks = nn.ModuleList([])
        for block_name, block_params in list(reversed(blocks)):
            input_channel = output_channel
            if isinstance(block_params, int):
                block_params = {
                    'num_layers': block_params }
            if block_name == 'res_x':
                block = UNetMidBlock3D(dims, input_channel, block_params['num_layers'], 1e-06, norm_num_groups, norm_layer, block_params.get('inject_noise', False), timestep_conditioning, spatial_padding_mode, **('dims', 'in_channels', 'num_layers', 'resnet_eps', 'resnet_groups', 'norm_layer', 'inject_noise', 'timestep_conditioning', 'spatial_padding_mode'))
            elif block_name == 'attn_res_x':
                block = UNetMidBlock3D(dims, input_channel, block_params['num_layers'], norm_num_groups, norm_layer, block_params.get('inject_noise', False), timestep_conditioning, block_params['attention_head_dim'], spatial_padding_mode, **('dims', 'in_channels', 'num_layers', 'resnet_groups', 'norm_layer', 'inject_noise', 'timestep_conditioning', 'attention_head_dim', 'spatial_padding_mode'))
            elif block_name == 'res_x_y':
                output_channel = output_channel // block_params.get('multiplier', 2)
                block = ResnetBlock3D(dims, input_channel, output_channel, 1e-06, norm_num_groups, norm_layer, block_params.get('inject_noise', False), False, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'eps', 'groups', 'norm_layer', 'inject_noise', 'timestep_conditioning', 'spatial_padding_mode'))
            elif block_name == 'compress_time':
                block = DepthToSpaceUpsample(dims, input_channel, (2, 1, 1), spatial_padding_mode, **('dims', 'in_channels', 'stride', 'spatial_padding_mode'))
            elif block_name == 'compress_space':
                block = DepthToSpaceUpsample(dims, input_channel, (1, 2, 2), spatial_padding_mode, **('dims', 'in_channels', 'stride', 'spatial_padding_mode'))
            elif block_name == 'compress_all':
                output_channel = output_channel // block_params.get('multiplier', 1)
                block = DepthToSpaceUpsample(dims, input_channel, (2, 2, 2), block_params.get('residual', False), block_params.get('multiplier', 1), spatial_padding_mode, **('dims', 'in_channels', 'stride', 'residual', 'out_channels_reduction_factor', 'spatial_padding_mode'))
            else:
                raise ValueError(f'''unknown layer: {block_name}''')
            self.up_blocks.append(block)
        if norm_layer == 'group_norm':
            self.conv_norm_out = nn.GroupNorm(output_channel, norm_num_groups, 1e-06, **('num_channels', 'num_groups', 'eps'))
        elif norm_layer == 'pixel_norm':
            self.conv_norm_out = PixelNorm()
        elif norm_layer == 'layer_norm':
            self.conv_norm_out = LayerNorm(output_channel, 1e-06, **('eps',))
        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(dims, output_channel, out_channels, 3, 1, True, spatial_padding_mode, **('padding', 'causal', 'spatial_padding_mode'))
        self.gradient_checkpointing = False
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000, torch.float32, **('dtype',)))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2, 0)
            self.last_scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel ** 0.5)
            return None

    
    def forward(self = None, sample = None, target_shape = None, timestep = (None,)):
        '''The forward method of the `Decoder` class.'''
        pass
    # WARNING: Decompyle incomplete



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
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        inject_noise (`bool`, *optional*, defaults to `False`):
            Whether to inject noise into the hidden states.
        timestep_conditioning (`bool`, *optional*, defaults to `False`):
            Whether to condition the hidden states on the timestep.
        attention_head_dim (`int`, *optional*, defaults to -1):
            The dimension of the attention head. If -1, no attention is used.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    '''
    
    def __init__(self = None, dims = None, in_channels = None, dropout = None, num_layers = None, resnet_eps = None, resnet_groups = None, norm_layer = None, inject_noise = None, timestep_conditioning = None, attention_head_dim = None, spatial_padding_mode = (0, 1, 1e-06, 32, 'group_norm', False, False, -1, 'zeros')):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4, 0)
        self.res_blocks = nn.ModuleList([ResnetBlock3D(dims=dims, in_channels=in_channels, out_channels=in_channels, eps=resnet_eps, groups=resnet_groups, dropout=dropout, norm_layer=norm_layer, inject_noise=inject_noise, timestep_conditioning=timestep_conditioning, spatial_padding_mode=spatial_padding_mode) for _ in range(num_layers)])
        self.attention_blocks = None
        if attention_head_dim > 0:
            if attention_head_dim > in_channels:
                raise ValueError('attention_head_dim must be less than or equal to in_channels')
            self.attention_blocks = nn.ModuleList([Attention(in_channels, in_channels // attention_head_dim, attention_head_dim, True, True, 'rms_norm', True, **('query_dim', 'heads', 'dim_head', 'bias', 'out_bias', 'qk_norm', 'residual_connection')) for _ in range(num_layers)])
            return None

    
    def forward(self = None, hidden_states = None, causal = None, timestep = (True, None)):
        timestep_embed = None
    # WARNING: Decompyle incomplete



class SpaceToDepthDownsample(nn.Module):
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, stride = None, spatial_padding_mode = None):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * np.prod(stride) // out_channels
        self.conv = make_conv_nd(dims, in_channels, out_channels // np.prod(stride), 3, 1, True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))

    
    def forward(self = None, x = None, causal = None):
        if self.stride[0] == 2:
            x = torch.cat([
                x[:, :, :1, :, :],
                x], 2, **('dim',))
        x_in = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', self.stride[0], self.stride[1], self.stride[2], **('p1', 'p2', 'p3'))
        x_in = rearrange(x_in, 'b (c g) d h w -> b c g d h w', self.group_size, **('g',))
        x_in = x_in.mean(2, **('dim',))
        x = self.conv(x, causal, **('causal',))
        x = rearrange(x, 'b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w', self.stride[0], self.stride[1], self.stride[2], **('p1', 'p2', 'p3'))
        x = x + x_in
        return x



class DepthToSpaceUpsample(nn.Module):
    
    def __init__(self = None, dims = None, in_channels = None, stride = None, residual = None, out_channels_reduction_factor = None, spatial_padding_mode = None):
        super().__init__()
        self.stride = stride
        self.out_channels = np.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(dims, in_channels, self.out_channels, 3, 1, True, spatial_padding_mode, **('dims', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'causal', 'spatial_padding_mode'))
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    
    def forward(self = None, x = None, causal = None):
        if self.residual:
            x_in = rearrange(x, 'b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)', self.stride[0], self.stride[1], self.stride[2], **('p1', 'p2', 'p3'))
            num_repeat = np.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal, **('causal',))
        x = rearrange(x, 'b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)', self.stride[0], self.stride[1], self.stride[2], **('p1', 'p2', 'p3'))
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x



class LayerNorm(nn.Module):
    
    def __init__(self = None, dim = None, eps = None, elementwise_affine = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine, **('eps', 'elementwise_affine'))

    
    def forward(self, x):
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x



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
    
    def __init__(self = None, dims = None, in_channels = None, out_channels = None, dropout = None, groups = None, eps = None, norm_layer = None, inject_noise = None, timestep_conditioning = None, spatial_padding_mode = None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.inject_noise = inject_noise
        if norm_layer == 'group_norm':
            self.norm1 = nn.GroupNorm(groups, in_channels, eps, True, **('num_groups', 'num_channels', 'eps', 'affine'))
        elif norm_layer == 'pixel_norm':
            self.norm1 = PixelNorm()
        elif norm_layer == 'layer_norm':
            self.norm1 = LayerNorm(in_channels, eps, True, **('eps', 'elementwise_affine'))
        self.non_linearity = nn.SiLU()
        self.conv1 = make_conv_nd(dims, in_channels, out_channels, 3, 1, 1, True, spatial_padding_mode, **('kernel_size', 'stride', 'padding', 'causal', 'spatial_padding_mode'))
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros((in_channels, 1, 1)))
        if norm_layer == 'group_norm':
            self.norm2 = nn.GroupNorm(groups, out_channels, eps, True, **('num_groups', 'num_channels', 'eps', 'affine'))
        elif norm_layer == 'pixel_norm':
            self.norm2 = PixelNorm()
        elif norm_layer == 'layer_norm':
            self.norm2 = LayerNorm(out_channels, eps, True, **('eps', 'elementwise_affine'))
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv_nd(dims, out_channels, out_channels, 3, 1, 1, True, spatial_padding_mode, **('kernel_size', 'stride', 'padding', 'causal', 'spatial_padding_mode'))
        if inject_noise:
            self.per_channel_scale2 = nn.Parameter(torch.zeros((in_channels, 1, 1)))
        self.conv_shortcut = make_linear_nd(dims, in_channels, out_channels, **('dims', 'in_channels', 'out_channels')) if in_channels != out_channels else nn.Identity()
        self.norm3 = LayerNorm(in_channels, eps, True, **('eps', 'elementwise_affine')) if in_channels != out_channels else nn.Identity()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels ** 0.5)
            return None

    
    def _feed_spatial_noise(self = None, hidden_states = None, per_channel_scale = None):
        spatial_shape = hidden_states.shape[-2:]
        device = hidden_states.device
        dtype = hidden_states.dtype
        spatial_noise = torch.randn(spatial_shape, device, dtype, **('device', 'dtype'))[None]
        scaled_noise = spatial_noise * per_channel_scale[None, :, None, ...]
        hidden_states = hidden_states + scaled_noise
        return hidden_states

    
    def forward(self = None, input_tensor = None, causal = None, timestep = (True, None)):
        hidden_states = input_tensor
        batch_size = hidden_states.shape[0]
        hidden_states = self.norm1(hidden_states)
    # WARNING: Decompyle incomplete



def patchify(x, patch_size_hw, patch_size_t = (1,)):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, 'b c (h q) (w r) -> b (c r q) h w', patch_size_hw, patch_size_hw, **('q', 'r'))
        return x
    if x.dim() == 5:
        x = rearrange(x, 'b c (f p) (h q) (w r) -> b (c p r q) f h w', patch_size_t, patch_size_hw, patch_size_hw, **('p', 'q', 'r'))
        return x
    raise ValueError(f'''Invalid input shape: {x.shape}''')


def unpatchify(x, patch_size_hw, patch_size_t = (1,)):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, 'b (c r q) h w -> b c (h q) (w r)', patch_size_hw, patch_size_hw, **('q', 'r'))
        return x
    if x.dim() == 5:
        x = rearrange(x, 'b (c p r q) f h w -> b c (f p) (h q) (w r)', patch_size_t, patch_size_hw, patch_size_hw, **('p', 'q', 'r'))
    return x


def create_video_autoencoder_demo_config(latent_channels = None):
    encoder_blocks = [
        ('res_x', {
            'num_layers': 2 }),
        ('compress_space_res', {
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 2 }),
        ('compress_time_res', {
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 1 }),
        ('compress_all_res', {
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 1 }),
        ('compress_all_res', {
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 1 })]
    decoder_blocks = [
        ('res_x', {
            'num_layers': 2,
            'inject_noise': False }),
        ('compress_all', {
            'residual': True,
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 2,
            'inject_noise': False }),
        ('compress_all', {
            'residual': True,
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 2,
            'inject_noise': False }),
        ('compress_all', {
            'residual': True,
            'multiplier': 2 }),
        ('res_x', {
            'num_layers': 2,
            'inject_noise': False })]
    return {
        '_class_name': 'CausalVideoAutoencoder',
        'dims': 3,
        'encoder_blocks': encoder_blocks,
        'decoder_blocks': decoder_blocks,
        'latent_channels': latent_channels,
        'norm_layer': 'pixel_norm',
        'patch_size': 4,
        'latent_log_var': 'uniform',
        'use_quant_conv': False,
        'causal_decoder': False,
        'timestep_conditioning': True,
        'spatial_padding_mode': 'replicate' }


def test_vae_patchify_unpatchify():
    import torch
    x = torch.randn(2, 3, 8, 64, 64)
    x_patched = patchify(x, 4, 4, **('patch_size_hw', 'patch_size_t'))
    x_unpatched = unpatchify(x_patched, 4, 4, **('patch_size_hw', 'patch_size_t'))
# WARNING: Decompyle incomplete


def demo_video_autoencoder_forward_backward():
    config = create_video_autoencoder_demo_config()
    video_autoencoder = CausalVideoAutoencoder.from_config(config)
    print(video_autoencoder)
    video_autoencoder.eval()
    total_params = sum(p.numel() for p in video_autoencoder.parameters())
    print(f'''Total number of parameters in VideoAutoencoder: {total_params:,}''')
    input_videos = torch.randn(2, 3, 17, 64, 64)
    latent = video_autoencoder.encode(input_videos).latent_dist.mode()
    print(f'''input shape={input_videos.shape}''')
    print(f'''latent shape={latent.shape}''')
    timestep = torch.ones(input_videos.shape[0]) * 0.1
    reconstructed_videos = video_autoencoder.decode(latent, input_videos.shape, timestep, **('target_shape', 'timestep')).sample
    print(f'''reconstructed shape={reconstructed_videos.shape}''')
    input_image = input_videos[:, :, :1, :, :]
    image_latent = video_autoencoder.encode(input_image).latent_dist.mode()
    _ = video_autoencoder.decode(image_latent, image_latent.shape, timestep, **('target_shape', 'timestep')).sample
    first_frame_latent = latent[:, :, :1, :, :]
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    demo_video_autoencoder_forward_backward()
