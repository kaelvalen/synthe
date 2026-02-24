"""
SYNTHE Jordan Layer — Fast Temporal Pattern Capture

Named after Michael I. Jordan, pioneer of recurrent neural networks
and early architectures for temporal sequence processing.

Gated exponential moving average with input-dependent decay.
This is SYNTHE's "reflexes" — captures local syntax patterns,
bigrams, trigrams, and short-range dependencies at O(1) per token.

Math:
    α_t = sigmoid(W_α · x_t)           # forget gate (input-dependent)
    k_t = W_k · x_t                    # key
    v_t = W_v · x_t                    # value  
    s_t = α_t ⊙ s_{t-1} + (1 - α_t) ⊙ (k_t ⊗ v_t)   # state update
    y_t = (W_q · x_t) · s_t           # query state

Inspired by: RWKV-7 WKV mechanism, mLSTM exponential gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .base import SyntheLayer, LayerState


class JordanLayer(SyntheLayer):
    """
    Jordan Layer — Gated EMA layer with multi-head parallel state tracking.
    
    Named after Michael I. Jordan: fast local pattern capture via
    recurrent temporal processing.
    
    Fastest SYNTHE layer — pure element-wise ops, no matrix multiply
    in the recurrence. Ideal for local pattern capture.
    
    Args:
        d_model: Input/output dimension
        state_dim: Dimension of each head's state (key_dim × value_dim)
        n_heads: Number of parallel momentum heads  
        expand_factor: FFN-style expansion in value projection
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        n_heads: int = 4,
        expand_factor: float = 1.0,
    ):
        super().__init__(d_model=d_model, state_dim=state_dim)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.expand_factor = expand_factor
        self.v_dim = int(self.head_dim * expand_factor)

        assert d_model % n_heads == 0

        # Projections
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, self.n_heads * self.v_dim, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # Input-dependent forget gate
        self.W_alpha = nn.Linear(d_model, n_heads, bias=True)

        # Output projection (maps back from expanded v_dim if needed)
        self.W_out = nn.Linear(self.n_heads * self.v_dim, d_model, bias=False)

        # Norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_k, self.W_v, self.W_q, self.W_out]:
            nn.init.xavier_normal_(module.weight, gain=0.5)
        # Initialize alpha bias positive → start with moderate retention
        nn.init.constant_(self.W_alpha.bias, 1.0)

    def init_state(self, batch_size: int, device: torch.device) -> LayerState:
        """State: (batch, n_heads, head_dim, v_dim) — outer product accumulator."""
        state = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.v_dim,
            device=device, dtype=torch.float32,
        )
        confidence = torch.ones(batch_size, device=device)
        return LayerState(state=state, confidence=confidence, step=0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
    ) -> Tuple[torch.Tensor, LayerState]:
        B, T, D = x.shape
        state = self._ensure_state(state, B, x.device)

        x_norm = self.norm(x)

        # Projections
        k = rearrange(self.W_k(x_norm), "b t (h d) -> b t h d", h=self.n_heads)
        v = rearrange(self.W_v(x_norm), "b t (h d) -> b t h d", h=self.n_heads)
        q = rearrange(self.W_q(x_norm), "b t (h d) -> b t h d", h=self.n_heads)

        # Normalize keys
        k = F.normalize(k, dim=-1)

        # Input-dependent forget gate
        alpha = torch.sigmoid(self.W_alpha(x_norm))  # (B, T, n_heads)

        # Sequential scan
        S = state.state  # (B, h, head_dim, v_dim)
        outputs = []

        for t in range(T):
            k_t = k[:, t]          # (B, h, head_dim)
            v_t = v[:, t]          # (B, h, v_dim)
            q_t = q[:, t]          # (B, h, head_dim)
            alpha_t = alpha[:, t]  # (B, h)

            # Gated EMA: s_t = α * s_{t-1} + (1 - α) * (k ⊗ v)
            kv_outer = torch.einsum("bhi,bhj->bhij", k_t, v_t)  # (B, h, hd, vd)
            a = alpha_t.unsqueeze(-1).unsqueeze(-1)  # (B, h, 1, 1)
            S = a * S + (1 - a) * kv_outer

            # Query: y = q · S → (B, h, v_dim)
            y_t = torch.einsum("bhi,bhij->bhj", q_t, S)
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # (B, T, h, v_dim)
        output = rearrange(output, "b t h d -> b t (h d)")
        output = self.W_out(output)

        # Confidence: based on gate stability
        # If alpha is consistently high or low → confident; oscillating → uncertain
        alpha_var = alpha.var(dim=1, unbiased=False).mean(dim=-1)  # (B,)
        confidence = torch.exp(-alpha_var)

        new_state = LayerState(
            state=S.detach() if not self.training else S,
            confidence=confidence.detach(),
            step=state.step + T,
        )

        return output, new_state


class ParallelJordanLayer(JordanLayer):
    """
    Jordan Layer with parallel scan for training efficiency.
    
    The gated EMA recurrence s_t = α_t * s_{t-1} + (1-α_t) * x_t
    can be computed via parallel prefix scan in O(T log T) work
    with O(log T) depth, vs O(T) sequential.
    
    Uses the associative scan trick:
        (a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
    """

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
    ) -> Tuple[torch.Tensor, LayerState]:
        B, T, D = x.shape

        # For short sequences or inference, use sequential
        if T <= 32 or not self.training:
            return super().forward(x, state)

        state = self._ensure_state(state, B, x.device)
        x_norm = self.norm(x)

        k = rearrange(self.W_k(x_norm), "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(self.W_v(x_norm), "b t (h d) -> b h t d", h=self.n_heads)
        q = rearrange(self.W_q(x_norm), "b t (h d) -> b h t d", h=self.n_heads)

        k = F.normalize(k, dim=-1)
        alpha = torch.sigmoid(self.W_alpha(x_norm))  # (B, T, h)
        alpha = rearrange(alpha, "b t h -> b h t")

        # Compute kv outer products for all timesteps
        kv = torch.einsum("bhti,bhtj->bhtij", k, v)  # (B, h, T, hd, vd)

        # Parallel prefix scan for gated EMA
        # For now, fall back to sequential — parallel scan requires custom CUDA
        # TODO: Implement triton kernel for parallel scan
        S = state.state
        all_S = []

        for t in range(T):
            a = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)
            S = a * S + (1 - a) * kv[:, :, t]
            all_S.append(S)

        all_S = torch.stack(all_S, dim=2)  # (B, h, T, hd, vd)

        # Query all states
        output = torch.einsum("bhti,bhtij->bhtj", q, all_S)  # (B, h, T, vd)
        output = rearrange(output, "b h t d -> b t (h d)")
        output = self.W_out(output)

        alpha_var = alpha.var(dim=-1, unbiased=False).mean(dim=-1)
        confidence = torch.exp(-alpha_var)

        new_state = LayerState(
            state=S.detach() if not self.training else S,
            confidence=confidence.detach(),
            step=state.step + T,
        )

        return output, new_state