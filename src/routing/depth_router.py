"""
SYNTHE Depth Router — Dynamic Computation Allocation

Not every token deserves the same compute. The depth router
estimates input difficulty and decides how many blocks to process.

Easy tokens (function words, predictable) → early exit after 2-3 blocks
Hard tokens (rare, ambiguous, critical)  → full depth + memory query

This is Mixture-of-Depths but continuous, not binary.
Expected savings: 30-50% FLOPs on average text.

The router uses two signals:
1. Its own difficulty estimate (tiny MLP)
2. Confidence from Kalman layer / Memory Hub (external signal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthRouter(nn.Module):
    """
    Lightweight difficulty estimator + routing decision.
    
    For each token, outputs:
        - continue_prob: probability of processing next block
        - should_probe: whether to activate attention probe
    
    During training: uses Gumbel-softmax for differentiable routing.
    During inference: hard thresholding.
    
    Args:
        d_model: Input dimension
        hidden_dim: Router MLP hidden size (tiny — <1% of model params)
        n_blocks: Total number of SYNTHE blocks
        min_depth: Minimum blocks every token must pass through
        exit_threshold: Below this continue_prob → early exit
        probe_threshold: Below this confidence → trigger attention probe
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        n_blocks: int = 12,
        min_depth: int = 2,
        exit_threshold: float = 0.5,
        probe_threshold: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.min_depth = min_depth
        self.exit_threshold = exit_threshold
        self.probe_threshold = probe_threshold
        self.temperature = temperature

        # Tiny MLP: difficulty estimator
        self.estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Confidence integrator: combines router estimate with external signals
        self.confidence_fuser = nn.Linear(3, 1, bias=True)  # [router, kalman_conf, memory_conf]

        self._init_weights()

    def _init_weights(self):
        # Initialize to "continue by default" — positive bias
        nn.init.zeros_(self.estimator[0].weight)
        nn.init.zeros_(self.estimator[2].weight)
        nn.init.constant_(self.estimator[2].bias, 1.0)  # Start with high continue prob

    def forward(
        self,
        x: torch.Tensor,             # (B, T, D) — current hidden state
        block_idx: int,               # Which block we're at (0-indexed)
        kalman_confidence: Optional[torch.Tensor] = None,  # (B,) from Kalman layer
        memory_confidence: Optional[torch.Tensor] = None,  # (B,) from Memory Hub
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            continue_mask: (B, T) — 1.0 = continue, 0.0 = exit
            probe_mask: (B,) — True = activate attention probe
            compute_budget: (B, T) — soft routing weight [0, 1]
        """
        B, T, D = x.shape

        # Always process minimum depth
        if block_idx < self.min_depth:
            return (
                torch.ones(B, T, device=x.device),
                torch.zeros(B, dtype=torch.bool, device=x.device),
                torch.ones(B, T, device=x.device),
            )

        # Router's own difficulty estimate
        router_score = self.estimator(x).squeeze(-1)  # (B, T)
        router_confidence = torch.sigmoid(router_score)  # (B, T)

        # Fuse with external confidence signals
        router_mean = router_confidence.mean(dim=1)  # (B,)
        
        if kalman_confidence is None:
            kalman_confidence = torch.ones(B, device=x.device)
        if memory_confidence is None:
            memory_confidence = torch.ones(B, device=x.device)

        fuse_input = torch.stack([
            router_mean, kalman_confidence, memory_confidence
        ], dim=-1)  # (B, 3)
        
        fused_confidence = torch.sigmoid(
            self.confidence_fuser(fuse_input).squeeze(-1)
        )  # (B,)

        # Continue decision
        if self.training:
            # Soft routing during training (differentiable)
            # Add noise for exploration
            noise = torch.randn_like(router_confidence) * self.temperature * 0.1
            compute_budget = torch.sigmoid((router_confidence + noise - self.exit_threshold) / self.temperature)
            continue_mask = compute_budget
        else:
            # Hard routing during inference
            compute_budget = router_confidence
            continue_mask = (router_confidence > self.exit_threshold).float()

        # Probe activation: based on fused confidence
        probe_mask = fused_confidence < self.probe_threshold

        return continue_mask, probe_mask, compute_budget

    def get_utilization(self, compute_budgets: list[torch.Tensor]) -> dict:
        """Compute routing statistics for logging."""
        if not compute_budgets:
            return {}
        
        all_budgets = torch.stack(compute_budgets, dim=0)  # (n_blocks, B, T)
        
        return {
            "mean_depth_fraction": all_budgets.mean().item(),
            "early_exit_rate": (all_budgets < 0.5).float().mean().item(),
            "full_depth_rate": (all_budgets > 0.5).all(dim=0).float().mean().item(),
            "flop_savings_estimate": 1.0 - all_budgets.mean().item(),
        }
