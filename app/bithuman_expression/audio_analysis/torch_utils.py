"""
bithuman_expression/audio_analysis/torch_utils.py
Python replacement for torch_utils.cpython-310-x86_64-linux-gnu.so

Exposes:
  get_mask_from_lengths(lengths, max_len=None) -> torch.BoolTensor
  linear_interpolation(features, seq_len)      -> torch.Tensor
"""

import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths: torch.Tensor, max_len: int = None) -> torch.BoolTensor:
    """
    Build a boolean mask from a batch of sequence lengths.

    Args:
        lengths: (B,) integer tensor — number of valid frames per sample
        max_len: if None, uses lengths.max()

    Returns:
        mask: (B, max_len) bool tensor — True at valid positions
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def linear_interpolation(features: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Linearly interpolate a sequence of feature vectors to a target length.

    Args:
        features: (B, T, C) float tensor
        seq_len:  target sequence length

    Returns:
        (B, seq_len, C) float tensor
    """
    # F.interpolate expects (B, C, T)
    x = features.transpose(1, 2).float()          # (B, C, T)
    out = F.interpolate(x, size=seq_len, mode="linear", align_corners=False)
    return out.transpose(1, 2).to(features.dtype)  # (B, seq_len, C)
