"""
SYNTHE Kalman Layer — Uncertainty-Aware State Estimation

Novel primitive: no existing sequence model tracks uncertainty.
Uses diagonal Kalman filtering — each state dimension is independent,
so no matrix inverse needed. O(d) per token.

Math (diagonal simplified):
    σ²_pred = α · σ²_{t-1} + q_t          # uncertainty grows (process noise)
    K_t = σ²_pred / (σ²_pred + r_t)       # Kalman gain: trust observation vs prior
    μ_t = μ_{t-1} + K_t · (z_t - μ_{t-1}) # state update
    σ²_t = (1 - K_t) · σ²_pred            # uncertainty shrinks after observation

The Kalman gain K_t is the key:
    - When σ² is high (uncertain) → K_t ≈ 1 → trust the new observation
    - When σ² is low (confident) → K_t ≈ 0 → trust the existing state
    - When r_t is high (noisy input) → K_t ≈ 0 → ignore the observation

This naturally implements "selective attention" — the state only updates
when it's uncertain OR when the observation is trustworthy.

Confidence signal is derived from the Innovation-to-Observation Ratio (IOR):
    - IOR = |z_t - μ_{t-1}| / |z_t|  (prediction error ÷ observation scale)
    - Pattern data → μ converges → IOR → 0 → HIGH confidence
    - Random/noise → μ can't predict → IOR ≈ 1+ → LOW confidence
This drives the attention probe: fire only when prediction fails.

Inspired by: Kalman filtering, KalmanNet, Bayesian RNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .base import SyntheLayer, LayerState


@torch.jit.script
def diagonal_kalman_step(
    mu: torch.Tensor,        # (B, D) — state mean
    sigma_sq: torch.Tensor,  # (B, D) — state variance
    z: torch.Tensor,         # (B, D) — observation
    alpha: torch.Tensor,     # (B, D) — state transition (decay)
    q: torch.Tensor,         # (B, D) — process noise
    r: torch.Tensor,         # (B, D) — observation noise
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single diagonal Kalman filter step. JIT-compiled for speed."""
    # Predict
    sigma_pred = alpha * sigma_sq + q

    # Kalman gain
    K = sigma_pred / (sigma_pred + r + 1e-8)

    # Update
    innovation = z - mu
    mu_new = mu + K * innovation
    sigma_new = (1.0 - K) * sigma_pred

    return mu_new, sigma_new, K


class KalmanLayer(SyntheLayer):
    """
    Diagonal Kalman filter as a sequence model layer.
    
    State: μ ∈ R^{state_dim}, σ² ∈ R^{state_dim}
    Per token: O(state_dim) — purely element-wise ops
    
    The confidence output is directly derived from σ² — 
    this drives SYNTHE's depth router and attention probe activation.
    
    Args:
        d_model: Input/output dimension
        state_dim: Kalman state dimension (capped at max_state_dim)
        n_heads: Number of independent Kalman filters
        learn_dynamics: If True, learn transition matrix A; else fixed decay
        min_obs_noise: Minimum observation noise (prevents overconfidence)
        max_state_dim: Hard cap on state_dim to prevent param bloat at scale
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        n_heads: int = 4,
        learn_dynamics: bool = True,
        min_obs_noise: float = 0.01,
        max_state_dim: int = 96,
    ):
        # Cap state_dim — Kalman doesn't need large state for uncertainty tracking
        state_dim = min(state_dim, max_state_dim)
        super().__init__(d_model=d_model, state_dim=state_dim)

        self.n_heads = n_heads
        self.head_dim = state_dim  # Each head has full state_dim
        self.learn_dynamics = learn_dynamics
        self.min_obs_noise = min_obs_noise

        # Project input → observation
        self.W_obs = nn.Linear(d_model, n_heads * state_dim, bias=False)

        # Fused noise projection: single projection → split into (r, q)
        # Halves param count vs separate W_r + W_q
        self.W_noise = nn.Linear(d_model, n_heads * state_dim * 2, bias=True)

        # State transition: either learned or fixed
        if learn_dynamics:
            self.W_alpha = nn.Linear(d_model, n_heads * state_dim, bias=True)
        else:
            # Fixed decay ≈ 0.95
            self.register_buffer(
                "alpha_fixed",
                torch.ones(1, n_heads, state_dim) * 0.95,
            )

        # Read from state: project (μ, σ²) → output
        self.W_read = nn.Linear(n_heads * state_dim * 2, d_model, bias=False)

        # Norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.W_obs.weight, gain=0.5)
        nn.init.xavier_normal_(self.W_read.weight, gain=0.5)
        # Fused noise bias init: first half = r (observation noise, cautious),
        # second half = q (process noise, slow growth)
        n = self.n_heads * self.state_dim
        nn.init.constant_(self.W_noise.bias[:n], 1.0)    # r: cautious start
        nn.init.constant_(self.W_noise.bias[n:], -2.0)   # q: slow uncertainty growth

    def init_state(self, batch_size: int, device: torch.device) -> LayerState:
        """Initialize with zero mean and high uncertainty."""
        # μ: (B, n_heads, state_dim) — zero mean
        mu = torch.zeros(batch_size, self.n_heads, self.state_dim, device=device)
        # σ²: (B, n_heads, state_dim) — high initial uncertainty
        # Start at 5.0 so the filter genuinely begins uncertain
        sigma_sq = torch.full(
            (batch_size, self.n_heads, self.state_dim), 5.0, device=device
        )
        # Pack both into state
        state = torch.stack([mu, sigma_sq], dim=-1)  # (B, h, sd, 2)
        confidence = torch.zeros(batch_size, device=device)  # Start uncertain
        return LayerState(state=state, confidence=confidence, step=0)

    def _unpack_state(self, state: torch.Tensor):
        """Unpack (B, h, sd, 2) → mu (B, h, sd), sigma_sq (B, h, sd)."""
        return state[..., 0], state[..., 1]

    def _pack_state(self, mu: torch.Tensor, sigma_sq: torch.Tensor):
        """Pack mu, sigma_sq → (B, h, sd, 2)."""
        return torch.stack([mu, sigma_sq], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[LayerState] = None,
    ) -> Tuple[torch.Tensor, LayerState]:
        B, T, D = x.shape
        state = self._ensure_state(state, B, x.device)

        x_norm = self.norm(x)

        # Observations
        z = rearrange(
            self.W_obs(x_norm), "b t (h d) -> b t h d", h=self.n_heads
        )

        # Fused noise projection → split into observation noise (r) and process noise (q)
        noise = self.W_noise(x_norm)  # (B, T, 2 * n_heads * state_dim)
        r_raw, q_raw = noise.chunk(2, dim=-1)
        r = rearrange(r_raw, "b t (h d) -> b t h d", h=self.n_heads)
        r = F.softplus(r) + self.min_obs_noise  # Ensure positive
        q = rearrange(q_raw, "b t (h d) -> b t h d", h=self.n_heads)
        q = F.softplus(q) + 1e-6

        # State transition
        if self.learn_dynamics:
            alpha = rearrange(
                self.W_alpha(x_norm), "b t (h d) -> b t h d", h=self.n_heads
            )
            alpha = torch.sigmoid(alpha)  # Bounded [0, 1]
        else:
            alpha = self.alpha_fixed.expand(B, T, -1, -1)

        # Unpack state
        mu, sigma_sq = self._unpack_state(state.state)

        outputs = []
        innov_ratios = []

        for t in range(T):
            # Innovation: prediction error BEFORE Kalman update
            # This measures how well the filter is predicting — the key
            # signal that differentiates predictable vs random input.
            innovation = z[:, t] - mu  # (B, h, d)
            innov_mag = innovation.abs().mean(dim=(-1, -2))  # (B,)
            obs_mag = z[:, t].abs().mean(dim=(-1, -2))       # (B,)
            innov_ratios.append(innov_mag / (obs_mag + 1e-8))

            mu, sigma_sq, K = diagonal_kalman_step(
                mu=mu,
                sigma_sq=sigma_sq,
                z=z[:, t],
                alpha=alpha[:, t],
                q=q[:, t],
                r=r[:, t],
            )

            # Read: concatenate mean and variance
            mu_flat = rearrange(mu, "b h d -> b (h d)")
            sigma_flat = rearrange(sigma_sq, "b h d -> b (h d)")
            read_input = torch.cat([mu_flat, sigma_flat], dim=-1)
            y_t = self.W_read(read_input)
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # (B, T, D)
        ior_history = torch.stack(innov_ratios, dim=1)  # (B, T)

        # Confidence from prediction quality (innovation-to-observation ratio).
        # Use last quarter to skip initial transient where mu=0.
        #   IOR ≈ 0 → filter predicts perfectly → HIGH confidence
        #   IOR ≈ 1+ → filter can't predict → LOW confidence
        # Repeated pattern → IOR→0 → conf→1.0 (probe OFF)
        # Random/noise → IOR≈1.4 → conf≈0.25 (probe ON)
        recent = max(1, T // 4)
        avg_ior = ior_history[:, -recent:].mean(dim=1)  # (B,)
        confidence = torch.exp(-avg_ior).clamp(0, 1)

        new_state = LayerState(
            state=self._pack_state(mu, sigma_sq).detach() 
                  if not self.training else self._pack_state(mu, sigma_sq),
            confidence=confidence.detach(),
            step=state.step + T,
        )

        return output, new_state