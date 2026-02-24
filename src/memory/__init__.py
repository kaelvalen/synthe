"""
SYNTHE Memory System — Hierarchical temporal memory orchestration

Tiers named after their intellectual lineage:
- Hebb Layer (Tier 1): Token-level fast memory
- Elman Module (Tier 2): Sentence-level delta memory
- Shannon Module (Tier 3): Discourse-level Kalman estimation
"""

from .hub import TemporalMemoryHub, MemoryState, TierCompressor, TierBroadcaster

__all__ = ["TemporalMemoryHub", "MemoryState", "TierCompressor", "TierBroadcaster"]
