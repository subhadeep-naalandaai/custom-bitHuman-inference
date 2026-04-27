# Source Generated with Decompyle++
# File: attention.pyc (Python 3.10)

import inspect
from importlib import import_module
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional
F = functional
nn
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.attention_processor import LoRAAttnAddedKVProcessor, LoRAAttnProcessor, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor, SpatialNorm
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import RMSNorm
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange
from torch import nn
from inference.ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
# WARNING: Decompyle incomplete
