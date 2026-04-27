# Source Generated with Decompyle++
# File: embeddings.pyc (Python 3.10)

import math
import numpy as np
import torch
from einops import rearrange
from torch import nn

def get_timestep_embedding(timesteps, embedding_dim = None, flip_sin_to_cos = None, downscale_freq_shift = None, scale = (False, 1, 1, 10000), max_period = ('timesteps', torch.Tensor, 'embedding_dim', int, 'flip_sin_to_cos', bool, 'downscale_freq_shift', float, 'scale', float, 'max_period', int)):
    '''
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    '''
    pass
# WARNING: Decompyle incomplete


def get_3d_sincos_pos_embed(embed_dim, grid, w, h, f):
    '''
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    '''
    grid = rearrange(grid, 'c (f h w) -> c f h w', h, w, **('h', 'w'))
    grid = rearrange(grid, 'c f h w -> c h w f', h, w, **('h', 'w'))
    grid = grid.reshape([
        3,
        1,
        w,
        h,
        f])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.transpose(1, 0, 2, 3)
    return rearrange(pos_embed, 'h w f c -> (f h w) c')


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError('embed_dim must be divisible by 3')
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    emb = np.concatenate([
        emb_h,
        emb_w,
        emb_f], -1, **('axis',))
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    '''
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    '''
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be divisible by 2')
    omega = np.arange(embed_dim // 2, np.float64, **('dtype',))
    omega /= embed_dim / 2
    omega = 1 / 10000 ** omega
    pos_shape = pos.shape
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    out = out.reshape(out.reshape[-1])[0]
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([
        emb_sin,
        emb_cos], -1, **('axis',))
    return emb


class SinusoidalPositionalEmbedding(nn.Module):
    '''Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    '''
    
    def __init__(self = None, embed_dim = None, max_seq_length = None):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        (_, seq_length, _) = x.shape
        x = x + self.pe[:, :seq_length]
        return x


