# Source Generated with Decompyle++
# File: vae_encode.pyc (Python 3.10)

from typing import Tuple
import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torch import Tensor
from inference.ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from inference.ltx_video.models.autoencoders.video_autoencoder import Downsample3D, VideoAutoencoder
# WARNING: Decompyle incomplete
