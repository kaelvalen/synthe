"""
SYNTHE Layers — Mechanistic Building Blocks

Four orthogonal kernels:
- DeltaLayer: Associative memory with active overwriting
- MomentumLayer: Fast local pattern capture via gated EMA
- KalmanLayer: Uncertainty-aware state estimation
- AttentionProbe: Emergency sliding-window attention

Base interface:
- SyntheLayer: Abstract base for all layers
- LayerState: Container for persistent layer state
"""

from .base import SyntheLayer, LayerState
from .delta import DeltaLayer, ChunkedDeltaLayer
from .momentum import MomentumLayer, ParallelMomentumLayer
from .kalman import KalmanLayer
from .attention import AttentionProbe

__all__ = [
    "SyntheLayer",
    "LayerState",
    "DeltaLayer",
    "ChunkedDeltaLayer",
    "MomentumLayer",
    "ParallelMomentumLayer",
    "KalmanLayer",
    "AttentionProbe",
]
