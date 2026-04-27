# Source Generated with Decompyle++
# File: vae.pyc (Python 3.10)

from typing import Optional, Union
import torch
import inspect
import math
from torch.nn import nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from inference.ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd

class AutoencoderKLWrapper(ConfigMixin, ModelMixin):
    '''Variational Autoencoder (VAE) model with KL loss.

    VAE from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling.
    This model is a wrapper around an encoder and a decoder, and it adds a KL loss term to the reconstruction loss.

    Args:
        encoder (`nn.Module`):
            Encoder module.
        decoder (`nn.Module`):
            Decoder module.
        latent_channels (`int`, *optional*, defaults to 4):
            Number of latent channels.
    '''
    
    def __init__(self = None, encoder = None, decoder = None, latent_channels = None, dims = None, sample_size = None, use_quant_conv = None, normalize_latent_channels = None):
        super().__init__()
        self.encoder = encoder
        self.use_quant_conv = use_quant_conv
        self.normalize_latent_channels = normalize_latent_channels
        quant_dims = 2 if dims == 2 else 3
        self.decoder = decoder
        if use_quant_conv:
            self.quant_conv = make_conv_nd(quant_dims, 2 * latent_channels, 2 * latent_channels, 1)
            self.post_quant_conv = make_conv_nd(quant_dims, latent_channels, latent_channels, 1)
        else:
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        if normalize_latent_channels:
            if dims == 2:
                self.latent_norm_out = nn.BatchNorm2d(latent_channels, False, **('affine',))
            else:
                self.latent_norm_out = nn.BatchNorm3d(latent_channels, False, **('affine',))
        else:
            self.latent_norm_out = nn.Identity()
        self.use_z_tiling = False
        self.use_hw_tiling = False
        self.dims = dims
        self.z_sample_size = 1
        self.decoder_params = inspect.signature(self.decoder.forward).parameters
        self.set_tiling_params(sample_size, 0.25, **('sample_size', 'overlap_factor'))

    
    def set_tiling_params(self = None, sample_size = None, overlap_factor = None):
        self.tile_sample_min_size = sample_size
        num_blocks = len(self.encoder.down_blocks)
        self.tile_latent_min_size = int(sample_size / 2 ** (num_blocks - 1))
        self.tile_overlap_factor = overlap_factor

    
    def enable_z_tiling(self = None, z_sample_size = None):
        '''
        Enable tiling during VAE decoding.

        When this option is enabled, the VAE will split the input tensor in tiles to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        '''
        self.use_z_tiling = z_sample_size > 1
        self.z_sample_size = z_sample_size
    # WARNING: Decompyle incomplete

    
    def disable_z_tiling(self):
        '''
        Disable tiling during VAE decoding. If `use_tiling` was previously invoked, this method will go back to computing
        decoding in one step.
        '''
        self.use_z_tiling = False

    
    def enable_hw_tiling(self):
        '''
        Enable tiling during VAE decoding along the height and width dimension.
        '''
        self.use_hw_tiling = True

    
    def disable_hw_tiling(self):
        '''
        Disable tiling during VAE decoding along the height and width dimension.
        '''
        self.use_hw_tiling = False

    
    def _hw_tiled_encode(self = None, x = None, return_dict = None):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :, i:i + self.tile_sample_min_size, j:j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, 4, **('dim',)))
        moments = torch.cat(result_rows, 3, **('dim',))
        return moments

    
    def blend_z(self = None, a = None, b = None, blend_extent = ('a', torch.Tensor, 'b', torch.Tensor, 'blend_extent', int, 'return', torch.Tensor)):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for z in range(blend_extent):
            b[:, :, z, :, :] = a[:, :, -blend_extent + z, :, :] * (1 - z / blend_extent) + b[:, :, z, :, :] * (z / blend_extent)
        return b

    
    def blend_v(self = None, a = None, b = None, blend_extent = ('a', torch.Tensor, 'b', torch.Tensor, 'blend_extent', int, 'return', torch.Tensor)):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    
    def blend_h(self = None, a = None, b = None, blend_extent = ('a', torch.Tensor, 'b', torch.Tensor, 'blend_extent', int, 'return', torch.Tensor)):
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    
    def _hw_tiled_decode(self = None, z = None, target_shape = None):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
    # WARNING: Decompyle incomplete

    
    def encode(self = None, z = None, return_dict = None):
        if self.use_z_tiling:
            if self.z_sample_size > self.z_sample_size or self.z_sample_size > 1:
                pass
            else:
                z.shape[2]
        else:
            num_splits = z.shape[2] // self.z_sample_size
            sizes = [
                self.z_sample_size] * num_splits
            sizes = sizes + [
                z.shape[2] - sum(sizes)] if z.shape[2] - sum(sizes) > 0 else sizes
            tiles = z.split(sizes, 2, **('dim',))
            moments_tiles = [self._hw_tiled_encode(z_tile, return_dict) if self.use_hw_tiling else self._encode(z_tile) for z_tile in tiles]
            moments = torch.cat(moments_tiles, 2, **('dim',))
        moments = self._hw_tiled_encode(z, return_dict) if self.use_hw_tiling else self._encode(z)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return None(posterior, **('latent_dist',))

    
    def _normalize_latent_channels(self = None, z = None):
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            (_, c, _, _, _) = z.shape
            z = torch.cat([
                self.latent_norm_out(z[:, :c // 2, :, :, :]),
                z[:, c // 2:, :, :, :]], 1, **('dim',))
            return z
        if None(self.latent_norm_out, nn.BatchNorm2d):
            raise NotImplementedError('BatchNorm2d not supported')
        return z

    
    def _unnormalize_latent_channels(self = None, z = None):
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            running_mean = self.latent_norm_out.running_mean.view(1, -1, 1, 1, 1)
            running_var = self.latent_norm_out.running_var.view(1, -1, 1, 1, 1)
            eps = self.latent_norm_out.eps
            z = z * torch.sqrt(running_var + eps) + running_mean
            return z
        if None(self.latent_norm_out, nn.BatchNorm3d):
            raise NotImplementedError('BatchNorm2d not supported')
        return z

    
    def _encode(self = None, x = None):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        moments = self._normalize_latent_channels(moments)
        return moments

    
    def _decode(self = None, z = None, target_shape = None, timestep = (None, None)):
        z = self._unnormalize_latent_channels(z)
        z = self.post_quant_conv(z)
        if 'timestep' in self.decoder_params:
            dec = self.decoder(z, target_shape, timestep, **('target_shape', 'timestep'))
            return dec
        dec = self.decoder(z, target_shape, **('target_shape',))
        return dec

    
    def decode(self = None, z = None, return_dict = None, target_shape = (True, None, None), timestep = ('z', torch.FloatTensor, 'return_dict', bool, 'timestep', Optional[torch.Tensor], 'return', Union[(DecoderOutput, torch.FloatTensor)])):
        pass
    # WARNING: Decompyle incomplete

    
    def forward(self = None, sample = None, sample_posterior = None, return_dict = (False, True, None), generator = ('sample', torch.FloatTensor, 'sample_posterior', bool, 'return_dict', bool, 'generator', Optional[torch.Generator], 'return', Union[(DecoderOutput, torch.FloatTensor)])):
        '''
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                Generator used to sample from the posterior.
        '''
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator, **('generator',))
        else:
            z = posterior.mode()
        dec = self.decode(z, sample.shape, **('target_shape',)).sample
        if not return_dict:
            return (dec,)
        return AutoencoderKLOutput(sample=dec)


