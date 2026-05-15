# Source Generated with Decompyle++
# File: pixel_norm.pyc (Python 3.10)

import torch
from torch import nn

class PixelNorm(nn.Module):
    
    def __init__(self = None, dim = None, eps = None):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, self.dim, True, **('dim', 'keepdim')) + self.eps)


