"""
SYNTHE Layers — Mechanistic Building Blocks

Named after the scientists whose ideas they embody:

- HopfieldCore (ex-DeltaLayer): Associative memory with active overwriting
  → John Hopfield — energy-based associative memory networks

- JordanLayer (ex-MomentumLayer): Fast local pattern capture via gated EMA
  → Michael I. Jordan — recurrent network architectures

- WienerCore (ex-KalmanLayer): Uncertainty-aware state estimation
  → Norbert Wiener — cybernetics and optimal filtering

- TuringGate: Epistemic confidence aggregation and routing
  → Alan Turing — a system that knows when it doesn't know

- AttentionProbe: Emergency sliding-window attention (unchanged)

Base interface:
- SyntheLayer: Abstract base for all layers
- LayerState: Container for persistent layer state

Backward-compatible aliases: DeltaLayer, ChunkedDeltaLayer, 
MomentumLayer, ParallelMomentumLayer, KalmanLayer
"""

from .base import SyntheLayer, LayerState
from .hopfield import HopfieldCore, ChunkedHopfieldCore
from .jordan import JordanLayer, ParallelJordanLayer
from .wiener import WienerCore
from .turing import TuringGate
from .attention import AttentionProbe

# === Backward-compatible aliases ===
DeltaLayer = HopfieldCore
ChunkedDeltaLayer = ChunkedHopfieldCore
MomentumLayer = JordanLayer
ParallelMomentumLayer = ParallelJordanLayer
KalmanLayer = WienerCore

__all__ = [
    # Base
    "SyntheLayer",
    "LayerState",
    # Named layers
    "HopfieldCore",
    "ChunkedHopfieldCore",
    "JordanLayer",
    "ParallelJordanLayer",
    "WienerCore",
    "TuringGate",
    "AttentionProbe",
    # Backward-compatible aliases
    "DeltaLayer",
    "ChunkedDeltaLayer",
    "MomentumLayer",
    "ParallelMomentumLayer",
    "KalmanLayer",
]
