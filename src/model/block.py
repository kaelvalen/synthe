"""
SYNTHE Block — The Composable Unit

One SYNTHE block = Momentum → Delta → Kalman → (optional Probe)
with residual connections, memory hub interaction, and depth routing.

Multiple blocks are stacked to form the full model.
Each block maintains its own state across tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..layers import (
    MomentumLayer,
    DeltaLayer,
    KalmanLayer,
    AttentionProbe,
    LayerState,
)


@dataclass
class BlockState:
    """Combined state for all layers in a block."""
    momentum_state: LayerState
    delta_state: LayerState
    kalman_state: LayerState
    probe_state: LayerState

    def detach(self) -> "BlockState":
        return BlockState(
            momentum_state=self.momentum_state.detach(),
            delta_state=self.delta_state.detach(),
            kalman_state=self.kalman_state.detach(),
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
        x → [Momentum] → +residual → [Delta] → +residual → 
        [Kalman] → +residual → [FFN] → +residual → 
        (if low confidence: [Attention Probe] → +residual)
        → output
    
    Args:
        d_model: Model dimension
        state_dim: State size for Delta and Momentum layers
        kalman_state_dim: State size for Kalman layer (typically smaller)
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
        kalman_state_dim: int = 64,
        n_heads: int = 4,
        ffn_expand: float = 2.67,
        probe_window: int = 256,
        probe_threshold: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Layer primitives
        self.momentum = MomentumLayer(
            d_model=d_model, state_dim=state_dim, n_heads=n_heads
        )
        self.delta = DeltaLayer(
            d_model=d_model, state_dim=state_dim, n_heads=n_heads
        )
        self.kalman = KalmanLayer(
            d_model=d_model, state_dim=kalman_state_dim, n_heads=n_heads
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
            momentum_state=self.momentum.init_state(batch_size, device),
            delta_state=self.delta.init_state(batch_size, device),
            kalman_state=self.kalman.init_state(batch_size, device),
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

        # === Momentum Layer ===
        momentum_out, momentum_state = self.momentum(x, state.momentum_state)
        x = x + self.dropout(momentum_out)

        # === Delta Layer ===
        delta_out, delta_state = self.delta(x, state.delta_state)
        x = x + self.dropout(delta_out)

        # === Kalman Layer ===
        kalman_out, kalman_state = self.kalman(x, state.kalman_state)
        x = x + self.dropout(kalman_out)

        # === FFN ===
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(ffn_out)

        # === Conditional Attention Probe ===
        kalman_confidence = kalman_state.confidence  # (B,)
        probe_out, probe_state = self.probe(
            x, state=state.probe_state, confidence=kalman_confidence
        )
        x = x + probe_out  # probe_out is zero when not activated

        # === Apply compute budget (soft masking for depth routing) ===
        if compute_budget is not None:
            # compute_budget: (B, T) in [0, 1]
            # Scale output — low budget tokens contribute less
            x = x * compute_budget.unsqueeze(-1)

        new_state = BlockState(
            momentum_state=momentum_state,
            delta_state=delta_state,
            kalman_state=kalman_state,
            probe_state=probe_state,
        )

        info = {
            "kalman_confidence": kalman_confidence.mean().item(),
            "delta_surprise": 1.0 - delta_state.confidence.mean().item(),
            "probe_activated": (kalman_confidence < self.probe.confidence_threshold).float().mean().item(),
        }

        return x, new_state, info
