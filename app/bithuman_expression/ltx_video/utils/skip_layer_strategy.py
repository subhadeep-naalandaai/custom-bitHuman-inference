# Source Generated with Decompyle++
# File: skip_layer_strategy.pyc (Python 3.10)

from enum import Enum, auto

class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto()

