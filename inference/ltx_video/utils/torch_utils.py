# Source Generated with Decompyle++
# File: torch_utils.pyc (Python 3.10)

import torch
from torch import nn

def append_dims(x = None, target_dims = None):
    '''Appends dimensions to the end of a tensor until it has target_dims dimensions.'''
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'''input has {x.ndim} dims but target_dims is {target_dims}, which is less''')
    if dims_to_append == 0:
        return x
    return None[(...,) + (None,) * dims_to_append]


class Identity(nn.Module):
    '''A placeholder identity operator that is argument-insensitive.'''
    
    def __init__(self = None, *args, **kwargs):
        super().__init__()

    
    def forward(self = None, x = None, *args, **kwargs):
        return x


