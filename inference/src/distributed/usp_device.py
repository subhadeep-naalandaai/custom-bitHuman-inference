"""
bithuman_expression/src/distributed/usp_device.py
Python replacement for usp_device.cpython-310-x86_64-linux-gnu.so

Unified Sequence Parallelism (USP) device utilities.
Used to determine optimal parallel degrees and device groups
when running inference across multiple GPUs.

Exposes:
  get_device(ulysses_degree, ring_degree) -> torch.device or ProcessGroup
  get_parallel_degree(world_size, num_heads) -> int
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def get_parallel_degree(world_size: int, num_heads: int) -> int:
    """
    Compute the optimal sequence-parallel degree given world_size GPUs
    and num_heads attention heads.

    The parallel degree must evenly divide both world_size and num_heads
    so that each GPU handles an integer number of heads and sequence chunks.

    Args:
        world_size: total number of processes / GPUs
        num_heads:  number of attention heads in the model

    Returns:
        Largest valid parallel degree (<= world_size) that divides num_heads.
        Returns 1 when no parallelism is possible or world_size == 1.
    """
    if world_size <= 1:
        return 1
    # Find the largest factor of world_size that also divides num_heads
    for degree in range(world_size, 0, -1):
        if world_size % degree == 0 and num_heads % degree == 0:
            return degree
    return 1


def get_device(ulysses_degree: int, ring_degree: int):
    """
    Return the appropriate distributed device / process group for the
    given USP configuration (Ulysses + Ring-Attention degrees).

    Args:
        ulysses_degree: sequence-parallel degree via DeepSpeed Ulysses method
        ring_degree:    sequence-parallel degree via Ring-Attention method

    Returns:
        torch.device for the current rank, or None when not in distributed mode.
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            device_id = rank % torch.cuda.device_count()
            return torch.device(f"cuda:{device_id}")
    except Exception:
        pass

    # Single-GPU / non-distributed fallback
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
