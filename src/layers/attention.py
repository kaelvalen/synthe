"""
SYNTHE Attention Probe — Emergency Precision Recall

NOT a full attention layer. Small, local, sliding-window,
activated ONLY when confidence is below threshold.

O(w) per token where w = window_size (default 256).
Most tokens never trigger this — estimated 10-20% activation rate.

This is pragmatic: Delta layer handles 90% of recall.
Attention probe is the safety net for edge cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .base import SyntheLayer, LayerState


class AttentionProbe(SyntheLayer):
    """
    Sparse sliding-window attention, conditionally activated.
    
    Only computes attention when the incoming confidence signal
    (from Kalman layer or depth router) falls below threshold.
    Otherwise, passes input through with zero compute.
    
    Args:
        d_model: Input/output dimension
        window_size: Local attention window (default 256)
        n_heads: Number of attention heads
        confidence_threshold: Below this → activate attention
        always_on: If True, ignore confidence and always compute (for ablation)
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 0,  # Not used, kept for interface compatibility
        window_size: int = 256,
        n_heads: int = 4,
        confidence_threshold: float = 0.3,
        always_on: bool = False,
    ):
        super().__init__(d_model=d_model, state_dim=window_size)

        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.confidence_threshold = confidence_threshold
        self.always_on = always_on
        self.scale = self.head_dim ** -0.5

        assert d_model % n_heads == 0

        # QKV projection (fused for efficiency)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Buffer for caching KV within window
        self.register_buffer("_kv_cache", None, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W_qkv.weight, gain=0.5)
        nn.init.xavier_normal_(self.W_out.weight, gain=0.1)  # Small init — start as near-identity

    def init_state(self, batch_size: int, device: torch.device) -> LayerState:
        """State: KV cache for sliding window."""
        # Cache: (batch, window_size, 2, n_heads, head_dim) — K and V
        cache = torch.zeros(
            batch_size, self.window_size, 2, self.n_heads, self.head_dim,
            device=device,
        )
        confidence = torch.ones(batch_size, device=device)
        return LayerState(state=cache, confidence=confidence, step=0)

    def _sliding_window_attention(
        self,
        x: torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:
        """Compute local sliding-window attention."""
        B, T, D = x.shape

        x_norm = self.norm(x)
        qkv = self.W_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        # Create causal sliding window mask
        # Each position attends to [max(0, t-w+1), t] 
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        # Window mask: block anything beyond window_size
        window_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=-(self.window_size),
        )

        full_mask = causal_mask | window_mask
        attn_scores = attn_scores.masked_fill(full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum("bhij,bhjd->bhid", attn_weights, v)
        output = rearrange(output, "b h t d -> b t (h d)")

        return self.W_out(output)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
        confidence: Optional[torch.Tensor] = None,  # External confidence signal
    ) -> Tuple[torch.Tensor, LayerState]:
        B, T, D = x.shape
        state = self._ensure_state(state, B, x.device)

        if self.always_on:
            output = self._sliding_window_attention(x)
            new_state = LayerState(
                state=state.state,
                confidence=torch.ones(B, device=x.device),
                step=state.step + T,
            )
            return output, new_state

        # Check confidence — only activate if below threshold
        if confidence is not None:
            # Per-batch activation mask
            activate = confidence < self.confidence_threshold  # (B,)

            if not activate.any():
                # Nobody needs attention — skip entirely
                new_state = LayerState(
                    state=state.state,
                    confidence=state.confidence,
                    step=state.step + T,
                )
                return torch.zeros_like(x), new_state

            if activate.all():
                # Everyone needs attention
                output = self._sliding_window_attention(x)
            else:
                # Mixed: only compute for uncertain samples
                output = torch.zeros_like(x)
                active_idx = activate.nonzero(as_tuple=True)[0]
                active_x = x[active_idx]
                active_out = self._sliding_window_attention(active_x)
                output[active_idx] = active_out
        else:
            # No confidence signal — always compute
            output = self._sliding_window_attention(x)

        new_state = LayerState(
            state=state.state,
            confidence=torch.ones(B, device=x.device),
            step=state.step + T,
        )

        return output, new_state