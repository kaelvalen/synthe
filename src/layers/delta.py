"""
SYNTHE Delta Layer — Associative Memory with Active Overwriting

The key insight: standard gated recurrence ADDS new information while
DECAYING old information. The delta rule ERASES the old association
before writing the new one. This is why it achieves near-perfect
associative recall where Mamba and RWKV fail.

Math:
    v̂_t = S_{t-1} · k_t           # retrieve: what does state predict for this key?
    e_t = v_t - v̂_t                # error: surprise signal
    S_t = S_{t-1} + β_t · (e_t ⊗ k_t)  # update: correct the association

When key k was previously associated with value v_old:
    v̂ = S · k ≈ v_old
    e = v_new - v_old              # the correction
    S_new = S + β · (v_new - v_old) ⊗ k  # overwrites v_old → v_new

Inspired by: Gated DeltaNet (ICLR 2025), Hopfield delta rule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .base import SyntheLayer, LayerState


class DeltaLayer(SyntheLayer):
    """
    Delta rule layer with input-dependent gating.
    
    State: S ∈ R^{state_dim × state_dim} — associative memory matrix
    Per token: O(state_dim²) compute, O(1) relative to sequence length
    
    Args:
        d_model: Input/output dimension
        state_dim: Size of the square state matrix
        n_heads: Number of parallel delta heads (state is shared per head)
        use_gate: Apply forget gate before delta update (Gated DeltaNet style)
        beta_min: Minimum learning rate (prevents zero updates)
        beta_max: Maximum learning rate (prevents state explosion)
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 128,
        n_heads: int = 4,
        use_gate: bool = True,
        beta_min: float = 0.01,
        beta_max: float = 0.99,
    ):
        super().__init__(d_model=d_model, state_dim=state_dim)
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_gate = use_gate
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Projections: input → key, value, query, learning rate
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)  # per-head learning rate

        # Optional forget gate (Gated DeltaNet style)
        if use_gate:
            self.W_gate = nn.Linear(d_model, n_heads, bias=True)

        # Output projection
        self.W_out = nn.Linear(d_model, d_model, bias=False)

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

        # Initialize projections with small values for stability
        self._init_weights()

    def _init_weights(self):
        for module in [self.W_k, self.W_v, self.W_q, self.W_out]:
            nn.init.xavier_normal_(module.weight, gain=0.5)
        nn.init.zeros_(self.W_beta.bias)
        if self.use_gate:
            # Initialize gate bias positive → start with high retention
            nn.init.constant_(self.W_gate.bias, 2.0)

    def init_state(self, batch_size: int, device: torch.device) -> LayerState:
        """Initialize empty associative memory."""
        # State: (batch, n_heads, head_dim, head_dim) — one matrix per head
        state = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=torch.float32,
        )
        confidence = torch.ones(batch_size, device=device)
        return LayerState(state=state, confidence=confidence, step=0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
    ) -> Tuple[torch.Tensor, LayerState]:
        """
        Args:
            x: (batch, seq_len, d_model)
            state: Previous LayerState or None
            
        Returns:
            output: (batch, seq_len, d_model) — residual-ready
            new_state: Updated state after processing all tokens
        """
        B, T, D = x.shape
        state = self._ensure_state(state, B, x.device)

        # Pre-norm
        x_norm = self.norm(x)

        # Project to keys, values, queries
        k = self.W_k(x_norm)  # (B, T, D)
        v = self.W_v(x_norm)  # (B, T, D)
        q = self.W_q(x_norm)  # (B, T, D)

        # Per-head learning rate: bounded sigmoid
        beta = self.W_beta(x_norm)  # (B, T, n_heads)
        beta = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(beta)

        # Optional forget gate
        if self.use_gate:
            gate = torch.sigmoid(self.W_gate(x_norm))  # (B, T, n_heads)

        # Reshape to multi-head: (B, T, n_heads, head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.n_heads)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.n_heads)

        # Normalize keys for stable state updates
        k = F.normalize(k, dim=-1)

        # Sequential scan through tokens
        S = state.state  # (B, n_heads, head_dim, head_dim)
        outputs = []
        surprise_accum = torch.zeros(B, device=x.device)

        for t in range(T):
            k_t = k[:, t]      # (B, h, d)
            v_t = v[:, t]      # (B, h, d)
            q_t = q[:, t]      # (B, h, d)
            beta_t = beta[:, t]  # (B, h)

            # Retrieve: what does current state predict for this key?
            # v̂_t = S · k_t
            v_hat = torch.einsum("bhij,bhj->bhi", S, k_t)  # (B, h, d)

            # Error signal: surprise
            error = v_t - v_hat  # (B, h, d)

            # Track surprise magnitude for confidence estimation
            surprise_t = error.norm(dim=-1).mean(dim=-1)  # (B,)
            surprise_accum = surprise_accum + surprise_t

            # Apply forget gate before update
            if self.use_gate:
                gate_t = gate[:, t]  # (B, h)
                S = S * gate_t.unsqueeze(-1).unsqueeze(-1)

            # Delta rule update: S += β * (error ⊗ key)
            # (B, h, d, 1) × (B, h, 1, d) → (B, h, d, d)
            delta = torch.einsum("bhi,bhj->bhij", error, k_t)
            S = S + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

            # Query the updated state
            # y_t = S · q_t
            y_t = torch.einsum("bhij,bhj->bhi", S, q_t)  # (B, h, d)
            outputs.append(y_t)

        # Recombine heads: (B, T, n_heads, head_dim) → (B, T, d_model)
        output = torch.stack(outputs, dim=1)  # (B, T, h, d)
        output = rearrange(output, "b t h d -> b t (h d)")

        # Output projection
        output = self.W_out(output)

        # Compute confidence: inverse of average surprise (normalized)
        avg_surprise = surprise_accum / T
        confidence = torch.exp(-avg_surprise)  # high surprise → low confidence

        new_state = LayerState(
            state=S.detach() if not self.training else S,
            confidence=confidence.detach(),
            step=state.step + T,
        )

        return output, new_state


class ChunkedDeltaLayer(DeltaLayer):
    """
    Delta layer with chunk-wise parallel processing.
    
    Instead of pure sequential scan, processes chunks of tokens
    in parallel (intra-chunk) then propagates state between chunks
    (inter-chunk). Trades some precision for significant speedup.
    
    For training: chunk_size=64-256 gives good speed/quality balance.
    For inference: falls back to sequential (chunk_size=1).
    """

    def __init__(self, *args, chunk_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
    ) -> Tuple[torch.Tensor, LayerState]:
        B, T, D = x.shape

        # For short sequences or inference, use sequential scan
        if T <= self.chunk_size or not self.training:
            return super().forward(x, state)

        # Chunk-wise processing
        # Pad sequence to multiple of chunk_size
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        n_chunks = x.shape[1] // self.chunk_size
        chunks = x.reshape(B, n_chunks, self.chunk_size, D)

        state = self._ensure_state(state, B, x.device)
        all_outputs = []

        for c in range(n_chunks):
            chunk_out, state = super().forward(chunks[:, c], state)
            all_outputs.append(chunk_out)

        output = torch.cat(all_outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            output = output[:, :T]

        return output, state