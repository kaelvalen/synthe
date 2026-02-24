"""
SYNTHE Temporal Memory Hub — Hierarchical Multi-Scale Memory

Three tiers, named after the scientists whose ideas they embody:

  Hebb Layer (Tier 1, Token):      Updates every token. Fast, high-bandwidth.
    → Donald Hebb — "fire together, wire together"

  Elman Module (Tier 2, Sentence):  Updates every ~16 tokens. Delta-rule overwrite.
    → Jeffrey Elman — recurrent temporal memory pioneer

  Shannon Module (Tier 3, Discourse): Updates every ~128 tokens. Kalman estimation.
    → Claude Shannon — information theory and entropy-based compression

Information flows bidirectionally:
  UP:   Hebb → compressed → Elman → compressed → Shannon
  DOWN: Shannon → context prior → Elman → bias → Hebb

Consolidation: persistent Hebb patterns are promoted to Elman.
Forgetting: Delta rule in Elman actively erases stale associations.

Inspired by hippocampal-neocortical memory consolidation
and BWR-DNC's multi-scale compression hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class MemoryState:
    """State for the entire memory hub."""
    tier1: torch.Tensor      # (B, state_dim1)
    tier2: torch.Tensor      # (B, state_dim2)  
    tier3: torch.Tensor      # (B, state_dim3)
    tier2_var: torch.Tensor   # (B, state_dim3) — Tier 3 uncertainty
    token_count: int          # Total tokens processed
    consolidation_buffer: torch.Tensor  # (B, buffer_size, d_model) — recent Tier 1 outputs

    def detach(self) -> "MemoryState":
        return MemoryState(
            tier1=self.tier1.detach(),
            tier2=self.tier2.detach(),
            tier3=self.tier3.detach(),
            tier2_var=self.tier2_var.detach(),
            token_count=self.token_count,
            consolidation_buffer=self.consolidation_buffer.detach(),
        )


class TierCompressor(nn.Module):
    """Compresses lower-tier signals for upper-tier consumption."""

    def __init__(self, d_input: int, d_output: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_output),
            nn.GELU(),
            nn.Linear(d_output, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TierBroadcaster(nn.Module):
    """Projects upper-tier context priors down to lower tiers."""

    def __init__(self, d_upper: int, d_lower: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_upper, d_lower),
            nn.Tanh(),  # Bounded output — bias, not override
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalMemoryHub(nn.Module):
    """
    Three-tier hierarchical memory with bidirectional flow.
    
    Tiers:
        Hebb Layer (tier1):     Token-level, fast GRU-style gated accumulator
        Elman Module (tier2):   Sentence-level, delta-rule overwrite
        Shannon Module (tier3): Discourse-level, Kalman state estimation
    
    Args:
        d_model: Model dimension (input/output)
        hebb_dim: Hebb Layer (Tier 1) state dimension (fast, token-level)
        elman_dim: Elman Module (Tier 2) state dimension (medium, sentence-level)
        shannon_dim: Shannon Module (Tier 3) state dimension (slow, discourse-level)
        elman_interval: Update Elman Module every N tokens
        shannon_interval: Update Shannon Module every N tokens
        consolidation_buffer_size: How many recent outputs to track for promotion
    """

    def __init__(
        self,
        d_model: int,
        tier1_dim: int = 256,
        tier2_dim: int = 128,
        tier3_dim: int = 64,
        tier2_interval: int = 16,
        tier3_interval: int = 128,
        consolidation_buffer_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.tier1_dim = tier1_dim
        self.tier2_dim = tier2_dim
        self.tier3_dim = tier3_dim
        self.tier2_interval = tier2_interval
        self.tier3_interval = tier3_interval
        self.buffer_size = consolidation_buffer_size

        # === Hebb Layer (Tier 1): Token-level gated accumulator ===
        self.tier1_gate = nn.Linear(d_model + tier1_dim, tier1_dim)
        self.tier1_candidate = nn.Linear(d_model + tier1_dim, tier1_dim)
        self.tier1_read = nn.Linear(tier1_dim, d_model)

        # === Elman Module (Tier 2): Sentence-level delta update ===
        self.tier2_key = nn.Linear(d_model, tier2_dim)
        self.tier2_value = nn.Linear(d_model, tier2_dim)
        self.tier2_beta = nn.Linear(d_model, 1)  # Learning rate
        self.tier2_read = nn.Linear(tier2_dim, d_model)

        # === Shannon Module (Tier 3): Discourse-level Kalman update ===
        self.tier3_obs = nn.Linear(d_model, tier3_dim)
        self.tier3_obs_noise = nn.Linear(d_model, tier3_dim)
        self.tier3_read = nn.Linear(tier3_dim * 2, d_model)  # mean + variance

        # === Inter-tier communication ===
        # Upward compression: Hebb → Elman → Shannon
        self.compress_1to2 = TierCompressor(tier1_dim, tier2_dim)
        self.compress_2to3 = TierCompressor(tier2_dim, tier3_dim)

        # Downward broadcasting: Shannon → Elman → Hebb
        self.broadcast_3to2 = TierBroadcaster(tier3_dim, tier2_dim)
        self.broadcast_2to1 = TierBroadcaster(tier2_dim, tier1_dim)

        # === Consolidation gate: decides what to promote from Hebb → Elman ===
        self.consolidation_gate = nn.Linear(d_model, 1)

        # === Output fusion: combine all tier readouts ===
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        return MemoryState(
            tier1=torch.zeros(batch_size, self.tier1_dim, device=device),
            tier2=torch.zeros(batch_size, self.tier2_dim, device=device),
            tier3=torch.zeros(batch_size, self.tier3_dim, device=device),
            tier2_var=torch.ones(batch_size, self.tier3_dim, device=device),
            token_count=0,
            consolidation_buffer=torch.zeros(
                batch_size, self.buffer_size, self.d_model, device=device
            ),
        )

    def _update_tier1(
        self, x_t: torch.Tensor, tier1: torch.Tensor, bias_from_tier2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GRU-style gated update for Hebb Layer (token-level memory)."""
        # Apply bias from higher tier
        tier1_biased = tier1 + bias_from_tier2

        combined = torch.cat([x_t, tier1_biased], dim=-1)
        gate = torch.sigmoid(self.tier1_gate(combined))
        candidate = torch.tanh(self.tier1_candidate(combined))
        tier1_new = gate * tier1_biased + (1 - gate) * candidate

        readout = self.tier1_read(tier1_new)
        return tier1_new, readout

    def _update_tier2(
        self, 
        summary: torch.Tensor,  # Compressed from Hebb Layer
        tier2: torch.Tensor,
        bias_from_tier3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delta-rule update for Elman Module (sentence-level memory)."""
        tier2_biased = tier2 + bias_from_tier3

        k = self.tier2_key(summary)
        v = self.tier2_value(summary)
        beta = torch.sigmoid(self.tier2_beta(summary))  # (B, 1)

        # Delta rule: retrieve → compute error → update
        v_hat = tier2_biased  # Simple: treat entire state as the "retrieved" value
        error = v - v_hat
        tier2_new = tier2_biased + beta * error

        readout = self.tier2_read(tier2_new)
        return tier2_new, readout

    def _update_tier3(
        self,
        summary: torch.Tensor,  # Compressed from Elman Module
        tier3: torch.Tensor,
        tier3_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Diagonal Kalman update for Shannon Module (discourse-level memory)."""
        z = self.tier3_obs(summary)
        r = F.softplus(self.tier3_obs_noise(summary)) + 0.01

        # Predict: uncertainty grows
        sigma_pred = 0.99 * tier3_var + 0.01

        # Kalman gain
        K = sigma_pred / (sigma_pred + r + 1e-8)

        # Update
        tier3_new = tier3 + K * (z - tier3)
        var_new = (1.0 - K) * sigma_pred

        # Readout includes uncertainty
        readout = self.tier3_read(torch.cat([tier3_new, var_new], dim=-1))

        return tier3_new, var_new, readout

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) — input from layer stack
        state: Optional[MemoryState] = None,
    ) -> Tuple[torch.Tensor, MemoryState]:
        B, T, D = x.shape

        if state is None:
            state = self.init_state(B, x.device)

        tier1 = state.tier1
        tier2 = state.tier2
        tier3 = state.tier3
        tier3_var = state.tier2_var
        token_count = state.token_count
        cons_buffer = state.consolidation_buffer

        outputs = []

        for t in range(T):
            x_t = x[:, t]  # (B, D)
            token_count += 1

            # --- Downward broadcasts (context priors) ---
            # Shannon → Elman → Hebb
            bias_2to1 = self.broadcast_2to1(tier2)  # (B, tier1_dim)
            bias_3to2 = self.broadcast_3to2(tier3)  # (B, tier2_dim)

            # --- Hebb Layer: every token ---
            tier1, read1 = self._update_tier1(x_t, tier1, bias_2to1)

            # --- Elman Module: every elman_interval tokens ---
            if token_count % self.tier2_interval == 0:
                # Compress Hebb signal for Elman
                compressed_1 = self.compress_1to2(tier1)
                # Use mean of consolidation buffer as summary
                buffer_mean = cons_buffer.mean(dim=1)  # (B, D)
                tier2, read2_raw = self._update_tier2(buffer_mean, tier2, bias_3to2)
            
            read2 = self.tier2_read(tier2)  # Always readable

            # --- Shannon Module: every shannon_interval tokens ---
            if token_count % self.tier3_interval == 0:
                compressed_2 = self.compress_2to3(tier2)
                tier3, tier3_var, read3_raw = self._update_tier3(
                    self.tier2_read(tier2), tier3, tier3_var
                )

            read3 = self.tier3_read(torch.cat([tier3, tier3_var], dim=-1))

            # --- Fusion: combine all tier readouts ---
            fused = self.fusion(torch.cat([read1, read2, read3], dim=-1))
            outputs.append(fused)

            # --- Update consolidation buffer (ring buffer) ---
            idx = (token_count - 1) % self.buffer_size
            cons_buffer[:, idx] = x_t.detach()

        output = torch.stack(outputs, dim=1)  # (B, T, D)
        output = self.norm(output)

        new_state = MemoryState(
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            tier2_var=tier3_var,
            token_count=token_count,
            consolidation_buffer=cons_buffer,
        )

        return output, new_state

    def get_confidence(self, state: MemoryState) -> torch.Tensor:
        """Global confidence (Turing Gate signal) from Shannon Module variance."""
        avg_var = state.tier2_var.mean(dim=-1)
        return torch.exp(-avg_var).clamp(0, 1)
