"""
SYNTHE Turing Gate — Epistemic Confidence System

Named after Alan Turing, whose work on machine intelligence and
computability theory underlies the concept of a system that
*knows when it doesn't know*.

The Turing Gate is SYNTHE's epistemic confidence mechanism:
it aggregates uncertainty signals from all computational layers
and makes the binary decision — compute more, or trust the state.

Signals consumed:
    - Wiener Core (Kalman) IOR → prediction quality
    - Hopfield Core (Delta) surprise → associative memory errors
    - Jordan Layer gate variance → temporal stability
    - Shannon Module (Tier 3) variance → discourse-level uncertainty

Signals produced:
    - continue_prob: should this token go deeper?
    - probe_mask: should the Attention Probe fire?
    - confidence: scalar [0, 1] for downstream routing

This is not a tribute — it is lineage. A system that evaluates
its own epistemic state is performing a computation Turing would
recognize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TuringGate(nn.Module):
    """
    Epistemic confidence aggregator and gating mechanism.
    
    Fuses uncertainty signals from multiple SYNTHE components
    into a single confidence scalar per sample, then makes
    routing decisions (continue/exit/probe).
    
    Args:
        n_signals: Number of incoming confidence signals (default 3)
        probe_threshold: Below this confidence → activate attention probe
        exit_threshold: Above this confidence → allow early exit
    """

    def __init__(
        self,
        n_signals: int = 3,
        probe_threshold: float = 0.35,
        exit_threshold: float = 0.5,
    ):
        super().__init__()
        self.probe_threshold = probe_threshold
        self.exit_threshold = exit_threshold

        # Learned fusion of confidence signals
        self.fuser = nn.Linear(n_signals, 1, bias=True)

        # Initialize to trust all signals equally
        nn.init.constant_(self.fuser.weight, 1.0 / n_signals)
        nn.init.zeros_(self.fuser.bias)

    def forward(
        self,
        *confidence_signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse confidence signals and produce gating decisions.
        
        Args:
            confidence_signals: Variable number of (B,) tensors in [0, 1]
            
        Returns:
            confidence: (B,) — fused confidence in [0, 1]
            probe_mask: (B,) bool — True where probe should fire
            exit_mask: (B,) bool — True where early exit is allowed
        """
        # Stack and default missing signals to 0.5 (uncertain)
        signals = []
        for s in confidence_signals:
            if s is not None:
                signals.append(s)
            else:
                signals.append(torch.full_like(signals[0] if signals else torch.tensor([0.5]), 0.5))

        stacked = torch.stack(signals, dim=-1)  # (B, n_signals)
        confidence = torch.sigmoid(self.fuser(stacked).squeeze(-1))  # (B,)

        probe_mask = confidence < self.probe_threshold
        exit_mask = confidence > self.exit_threshold

        return confidence, probe_mask, exit_mask

    def should_probe(self, confidence: torch.Tensor) -> torch.Tensor:
        """Quick check: does this confidence level warrant a probe?"""
        return confidence < self.probe_threshold

    def should_exit(self, confidence: torch.Tensor) -> torch.Tensor:
        """Quick check: is confidence high enough for early exit?"""
        return confidence > self.exit_threshold
