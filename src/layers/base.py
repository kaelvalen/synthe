"""
SYNTHE Base Layer Interface

Every SYNTHE layer is an online learner with:
- A state that persists across tokens
- An update rule that modifies state per token
- A query mechanism that reads from state
- An optional reset mechanism

All layers share this interface for composability.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LayerState:
    """Container for layer state between tokens.
    
    Each layer defines its own state shape, but all states
    share this interface for the memory hub and routing system.
    """
    state: torch.Tensor          # Primary state matrix/vector
    confidence: torch.Tensor     # Uncertainty estimate [0, 1] — 1 = certain
    step: int = 0                # How many tokens this state has seen

    def detach(self) -> "LayerState":
        """Detach state from computation graph (for TBPTT)."""
        return LayerState(
            state=self.state.detach(),
            confidence=self.confidence.detach(),
            step=self.step,
        )


class SyntheLayer(nn.Module, ABC):
    """Abstract base for all SYNTHE layers.
    
    Contract:
        - init_state(batch_size) → LayerState
        - forward(x, state) → (output, new_state)
        - Each layer tracks its own confidence estimate
    """

    def __init__(self, d_model: int, state_dim: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

    @abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> LayerState:
        """Initialize fresh state for a new sequence."""
        ...

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,              # (batch, seq_len, d_model)
        state: Optional[LayerState],   # None → auto-init
    ) -> Tuple[torch.Tensor, LayerState]:
        """
        Process input and update state.
        
        Returns:
            output: (batch, seq_len, d_model)
            new_state: Updated LayerState with confidence
        """
        ...

    def _ensure_state(
        self, state: Optional[LayerState], batch_size: int, device: torch.device
    ) -> LayerState:
        """Helper: init state if None."""
        if state is None:
            return self.init_state(batch_size, device)
        return state
