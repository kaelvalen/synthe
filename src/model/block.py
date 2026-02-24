"""
SYNTHE Block — The Composable Unit

One SYNTHE block = Jordan → Hopfield → Wiener → (optional Probe)
with residual connections, memory hub interaction, and depth routing.

Naming convention:
    Jordan Layer   (ex-Momentum)  — Michael I. Jordan
    Hopfield Core  (ex-Delta)     — John Hopfield
    Wiener Core    (ex-Kalman)    — Norbert Wiener
    Turing Gate    (confidence)   — Alan Turing

Multiple blocks are stacked to form the full model.
Each block maintains its own state across tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..layers import (
    JordanLayer,
    HopfieldCore,
    WienerCore,
    AttentionProbe,
    LayerState,
)


@dataclass
class BlockState:
    """Combined state for all layers in a block."""
    jordan_state: LayerState    # ex-momentum_state
    hopfield_state: LayerState  # ex-delta_state
    wiener_state: LayerState    # ex-kalman_state
    probe_state: LayerState

    def detach(self) -> "BlockState":
        return BlockState(
            jordan_state=self.jordan_state.detach(),
            hopfield_state=self.hopfield_state.detach(),
            wiener_state=self.wiener_state.detach(),
            probe_state=self.probe_state.detach(),
        )


class FeedForward(nn.Module):
    """Standard SwiGLU FFN — same as in Transformers/Mamba."""

    def __init__(self, d_model: int, expand_factor: float = 2.67):
        super().__init__()
        hidden = int(d_model * expand_factor)
        # Round to multiple of 64 for hardware efficiency
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class SyntheBlock(nn.Module):
    """
    One SYNTHE processing block.
    
    Flow:
        x → [Jordan] → +residual → [Hopfield] → +residual → 
        [Wiener] → +residual → [FFN] → +residual → 
        (if low confidence: [Attention Probe] → +residual)
        → output
    
    Args:
        d_model: Model dimension
        state_dim: State size for Hopfield and Jordan layers
        wiener_state_dim: State size for Wiener Core (typically smaller)
        n_heads: Number of heads for all multi-head layers
        ffn_expand: FFN expansion factor
        probe_window: Attention probe window size
        probe_threshold: Confidence threshold to activate probe
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 128,
        wiener_state_dim: int = 64,
        n_heads: int = 4,
        ffn_expand: float = 2.67,
        probe_window: int = 256,
        probe_threshold: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Layer primitives (named after their intellectual lineage)
        self.jordan = JordanLayer(
            d_model=d_model, state_dim=state_dim, n_heads=n_heads
        )
        self.hopfield = HopfieldCore(
            d_model=d_model, state_dim=state_dim, n_heads=n_heads
        )
        self.wiener = WienerCore(
            d_model=d_model, state_dim=wiener_state_dim, n_heads=n_heads
        )
        self.probe = AttentionProbe(
            d_model=d_model, window_size=probe_window,
            n_heads=n_heads, confidence_threshold=probe_threshold,
        )

        # FFN
        self.ffn = FeedForward(d_model, ffn_expand)
        self.ffn_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def init_state(self, batch_size: int, device: torch.device) -> BlockState:
        return BlockState(
            jordan_state=self.jordan.init_state(batch_size, device),
            hopfield_state=self.hopfield.init_state(batch_size, device),
            wiener_state=self.wiener.init_state(batch_size, device),
            probe_state=self.probe.init_state(batch_size, device),
        )

    def forward(
        self,
        x: torch.Tensor,                        # (B, T, D)
        state: Optional[BlockState] = None,
        compute_budget: Optional[torch.Tensor] = None,  # (B, T) from depth router
    ) -> Tuple[torch.Tensor, BlockState, Dict[str, Any]]:
        """
        Returns:
            output: (B, T, D)
            new_state: Updated BlockState
            info: Dict with confidence, surprise, etc.
        """
        B, T, D = x.shape

        if state is None:
            state = self.init_state(B, x.device)

        # === Jordan Layer (ex-Momentum) ===
        jordan_out, jordan_state = self.jordan(x, state.jordan_state)
        x = x + self.dropout(jordan_out)

        # === Hopfield Core (ex-Delta) ===
        hopfield_out, hopfield_state = self.hopfield(x, state.hopfield_state)
        x = x + self.dropout(hopfield_out)

        # === Wiener Core (ex-Kalman) ===
        wiener_out, wiener_state = self.wiener(x, state.wiener_state)
        x = x + self.dropout(wiener_out)

        # === FFN ===
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        # === Conditional Attention Probe ===
        # Turing Gate signal: Wiener confidence drives probe activation
        wiener_confidence = wiener_state.confidence  # (B,)
        probe_out, probe_state = self.probe(
            x, state=state.probe_state, confidence=wiener_confidence
        )
        x = x + probe_out  # probe_out is zero when not activated

        # === Apply compute budget (soft masking for depth routing) ===
        if compute_budget is not None:
            # compute_budget: (B, T) in [0, 1]
            # Scale output — low budget tokens contribute less
            x = x * compute_budget.unsqueeze(-1)

        new_state = BlockState(
            jordan_state=jordan_state,
            hopfield_state=hopfield_state,
            wiener_state=wiener_state,
            probe_state=probe_state,
        )

        info = {
            "wiener_confidence": wiener_confidence.mean().item(),
            "hopfield_surprise": 1.0 - hopfield_state.confidence.mean().item(),
            "probe_activated": (wiener_confidence < self.probe.confidence_threshold).float().mean().item(),
        }

        return x, new_state, info
