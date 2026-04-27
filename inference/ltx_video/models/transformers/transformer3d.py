# Source Generated with Decompyle++
# File: transformer3d.pyc (Python 3.10)

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import json
import glob
from pathlib import Path
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils import logging
from torch import nn
from safetensors import safe_open
from inference.ltx_video.models.transformers.attention import BasicTransformerBlock
from inference.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from inference.ltx_video.utils.diffusers_config_mapping import diffusers_and_ours_config_mapping, make_hashable_key, TRANSFORMER_KEYS_RENAME_DICT
logger = logging.get_logger(__name__)
@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor = None

class Transformer3DModel(ConfigMixin, ModelMixin):
    _supports_gradient_checkpointing = True
    
    def __init__(self, num_attention_heads, attention_head_dim, in_channels, out_channels, num_layers, dropout, norm_num_groups, cross_attention_dim, attention_bias, num_vector_embeds, activation_fn, num_embeds_ada_norm, use_linear_projection, only_cross_attention, double_self_attention, upcast_attention, adaptive_norm = None, standardization_norm = None, norm_elementwise_affine = None, norm_eps = None, attention_type = None, caption_channels = None, use_tpu_flash_attention = None, qk_norm = None, positional_embedding_type = None, positional_embedding_theta = None, positional_embedding_max_pos = register_to_config, timestep_scale_multiplier = (16, 88, None, None, 1, 0, 32, None, False, None, 'geglu', None, False, False, False, False, 'single_scale_shift', 'layer_norm', True, 1e-05, 'default', None, False, None, 'rope', None, None, None, False), causal_temporal_positioning = (('num_attention_heads', int, 'attention_head_dim', int, 'in_channels', Optional[int], 'out_channels', Optional[int], 'num_layers', int, 'dropout', float, 'norm_num_groups', int, 'cross_attention_dim', Optional[int], 'attention_bias', bool, 'num_vector_embeds', Optional[int], 'activation_fn', str, 'num_embeds_ada_norm', Optional[int], 'use_linear_projection', bool, 'only_cross_attention', bool, 'double_self_attention', bool, 'upcast_attention', bool, 'adaptive_norm', str, 'standardization_norm', str, 'norm_elementwise_affine', bool, 'norm_eps', float, 'attention_type', str, 'caption_channels', int, 'use_tpu_flash_attention', bool, 'qk_norm', Optional[str], 'positional_embedding_type', str, 'positional_embedding_theta', Optional[float], 'positional_embedding_max_pos', Optional[List[int]], 'timestep_scale_multiplier', Optional[float], 'causal_temporal_positioning', bool),)):
        super().__init__()
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.patchify_proj = nn.Linear(in_channels, inner_dim, True, **('bias',))
        self.positional_embedding_type = positional_embedding_type
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.use_rope = self.positional_embedding_type == 'rope'
        self.timestep_scale_multiplier = timestep_scale_multiplier
        if self.positional_embedding_type == 'absolute':
            raise ValueError('Absolute positional embedding is no longer supported')
        if self.positional_embedding_type == 'rope':
            if positional_embedding_theta is None:
                raise ValueError('If `positional_embedding_type` type is rope, `positional_embedding_theta` must also be defined')
            if positional_embedding_max_pos is None:
                raise ValueError('If `positional_embedding_type` type is rope, `positional_embedding_max_pos` must also be defined')
        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
            inner_dim, num_attention_heads, attention_head_dim,
            dropout=dropout, cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn, attention_bias=attention_bias,
            norm_type=norm_type, positional_embedding_type=positional_embedding_type,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            skip_layer_strategy=skip_layer_strategy,
        ) for _ in range(num_layers)])
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, False, 1e-06, **('elementwise_affine', 'eps'))
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.proj_out = nn.Linear(inner_dim, self.out_channels)
        self.adaln_single = AdaLayerNormSingle(inner_dim, False, **('use_additional_conditions',))
        if adaptive_norm == 'single_scale':
            self.adaln_single.linear = nn.Linear(inner_dim, 4 * inner_dim, True, **('bias',))
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(caption_channels, inner_dim, **('in_features', 'hidden_size'))
        self.gradient_checkpointing = False

    __init__ = register_to_config(__init__)
    
    def set_use_tpu_flash_attention(self):
        '''
        Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
        attention kernel.
        '''
        logger.info('ENABLE TPU FLASH ATTENTION -> TRUE')
        self.use_tpu_flash_attention = True
        for block in self.transformer_blocks:
            block.set_use_tpu_flash_attention()

    
    def create_skip_layer_mask(self = None, batch_size = None, num_conds = None, ptb_index = (None,), skip_block_list = ('batch_size', int, 'num_conds', int, 'ptb_index', int, 'skip_block_list', Optional[List[int]])):
        if skip_block_list is None or len(skip_block_list) == 0:
            return None
        num_layers = len(self.transformer_blocks)
        mask = torch.ones((num_layers, batch_size * num_conds), self.device, self.dtype, **('device', 'dtype'))
        for block_idx in skip_block_list:
            mask[block_idx, ptb_index::num_conds] = 0
        return mask

    
    def _set_gradient_checkpointing(self, module, value = (False,)):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value
            return None

    
    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack([indices_grid[:, i] / self.positional_embedding_max_pos[i] for i in range(3)], dim=-1)
        return fractional_positions

    
    def precompute_freqs_cis(self, indices_grid, spacing = ('exp',)):
        dtype = torch.float32
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        fractional_positions = self.get_fractional_positions(indices_grid)
        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == 'exp':
            indices = theta ** torch.linspace(math.log(start, theta), math.log(end, theta), dim // 6, device, dtype, **('device', 'dtype'))
            indices = indices.to(dtype, **('dtype',))
        elif spacing == 'exp_2':
            indices = 1 / theta ** (torch.arange(0, dim, 6, device, **('device',)) / dim)
            indices = indices.to(dtype, **('dtype',))
        elif spacing == 'linear':
            indices = torch.linspace(start, end, dim // 6, device, dtype, **('device', 'dtype'))
        elif spacing == 'sqrt':
            indices = torch.linspace(start ** 2, end ** 2, dim // 6, device, dtype, **('device', 'dtype')).sqrt()
        indices = indices * math.pi / 2
        if spacing == 'exp_2':
            freqs = (indices * fractional_positions.unsqueeze(-1)).transpose(-1, -2).flatten(2)
        else:
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
        cos_freq = freqs.cos().repeat_interleave(2, -1, **('dim',))
        sin_freq = freqs.sin().repeat_interleave(2, -1, **('dim',))
        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, :dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, :dim % 6])
            cos_freq = torch.cat([
                cos_padding,
                cos_freq], -1, **('dim',))
            sin_freq = torch.cat([
                sin_padding,
                sin_freq], -1, **('dim',))
        return (cos_freq.to(self.dtype), sin_freq.to(self.dtype))

    
    def load_state_dict(self = None, state_dict = None, *args, **kwargs):
        if any(key.startswith('model.diffusion_model.') for key in state_dict.keys()):
            state_dict = {k[len('model.diffusion_model.'):]: v for k, v in state_dict.items() if k.startswith('model.diffusion_model.')}
        super().load_state_dict(state_dict, *args, **kwargs)

    
    def from_pretrained(cls = None, pretrained_model_path = None, *args, **kwargs):
        pretrained_model_path = Path(pretrained_model_path)
    # WARNING: Decompyle incomplete

    from_pretrained = classmethod(from_pretrained)
    
    def forward(self, hidden_states, indices_grid, encoder_hidden_states, timestep, class_labels, cross_attention_kwargs, attention_mask = None, encoder_attention_mask = None, skip_layer_mask = None, skip_layer_strategy = (None, None, None, None, None, None, None, None, True), return_dict = ('hidden_states', torch.Tensor, 'indices_grid', torch.Tensor, 'encoder_hidden_states', Optional[torch.Tensor], 'timestep', Optional[torch.LongTensor], 'class_labels', Optional[torch.LongTensor], 'cross_attention_kwargs', Dict[(str, Any)], 'attention_mask', Optional[torch.Tensor], 'encoder_attention_mask', Optional[torch.Tensor], 'skip_layer_mask', Optional[torch.Tensor], 'skip_layer_strategy', Optional[SkipLayerStrategy], 'return_dict', bool)):
        '''
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            indices_grid (`torch.LongTensor` of shape `(batch size, 3, num latent pixels)`):
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            skip_layer_mask ( `torch.Tensor`, *optional*):
                A mask of shape `(num_layers, batch)` that indicates which layers to skip. `0` at position
                `layer, batch_idx` indicates that the layer should be skipped for the corresponding batch index.
            skip_layer_strategy ( `SkipLayerStrategy`, *optional*, defaults to `None`):
                Controls which layers are skipped when calculating a perturbed latent for spatiotemporal guidance.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        '''
        if not self.use_tpu_flash_attention:
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000
                attention_mask = attention_mask.unsqueeze(1)
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        hidden_states = self.patchify_proj(hidden_states)
        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep
        freqs_cis = self.precompute_freqs_cis(indices_grid)
        batch_size = hidden_states.shape[0]
        (timestep, embedded_timestep) = self.adaln_single(timestep.flatten(), {
            'resolution': None,
            'aspect_ratio': None }, batch_size, hidden_states.dtype, **('batch_size', 'hidden_dtype'))
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    # WARNING: Decompyle incomplete


