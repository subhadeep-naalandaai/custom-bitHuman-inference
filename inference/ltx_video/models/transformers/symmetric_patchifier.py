# Source Generated with Decompyle++
# File: symmetric_patchifier.pyc (Python 3.10)

from abc import ABC, abstractmethod
from typing import Tuple
import torch
from diffusers.configuration_utils import ConfigMixin
from einops import rearrange
from torch import Tensor

class Patchifier(ABC, ConfigMixin):
    
    def __init__(self = None, patch_size = None):
        super().__init__()
        self._patch_size = (1, patch_size, patch_size)

    
    def patchify(self = None, latents = None):
        raise NotImplementedError('Patchify method not implemented')

    patchify = abstractmethod(patchify)
    
    def unpatchify(self, latents = None, output_height = None, output_width = abstractmethod, out_channels = ('latents', Tensor, 'output_height', int, 'output_width', int, 'out_channels', int, 'return', Tuple[(Tensor, Tensor)])):
        pass

    unpatchify = abstractmethod(unpatchify)
    
    def patch_size(self):
        return self._patch_size

    patch_size = property(patch_size)
    
    def get_latent_coords(self, latent_num_frames, latent_height, latent_width, batch_size, device):
        '''
        Return a tensor of shape [batch_size, 3, num_patches] containing the
            top-left corner latent coordinates of each latent patch.
        The tensor is repeated for each batch element.
        '''
        latent_sample_coords = torch.meshgrid(torch.arange(0, latent_num_frames, self._patch_size[0], device, **('device',)), torch.arange(0, latent_height, self._patch_size[1], device, **('device',)), torch.arange(0, latent_width, self._patch_size[2], device, **('device',)))
        latent_sample_coords = torch.stack(latent_sample_coords, 0, **('dim',))
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = rearrange(latent_coords, 'b c f h w -> b c (f h w)', batch_size, **('b',))
        return latent_coords



class SymmetricPatchifier(Patchifier):
    
    def patchify(self = None, latents = None):
        (b, _, f, h, w) = latents.shape
        latent_coords = self.get_latent_coords(f, h, w, b, latents.device)
        latents = rearrange(latents, 'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)', self._patch_size[0], self._patch_size[1], self._patch_size[2], **('p1', 'p2', 'p3'))
        return (latents, latent_coords)

    
    def unpatchify(self, latents = None, output_height = None, output_width = None, out_channels = ('latents', Tensor, 'output_height', int, 'output_width', int, 'out_channels', int, 'return', Tuple[(Tensor, Tensor)])):
        output_height = output_height // self._patch_size[1]
        output_width = output_width // self._patch_size[2]
        latents = rearrange(latents, 'b (f h w) (c p q) -> b c f (h p) (w q)', output_height, output_width, self._patch_size[1], self._patch_size[2], **('h', 'w', 'p', 'q'))
        return latents


