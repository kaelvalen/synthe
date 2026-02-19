"""
SYNTHE — Full Model

The complete architecture:
    Embedding → [SYNTHE Block × N with Memory Hub + Depth Router] → LM Head

This is a causal language model that can be trained on standard
next-token prediction and evaluated on standard benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml
import math

from .block import SyntheBlock, BlockState
from ..memory.hub import TemporalMemoryHub, MemoryState
from ..routing.depth_router import DepthRouter


@dataclass
class SyntheConfig:
    """Full model configuration."""
    # Model dimensions
    d_model: int = 512
    n_blocks: int = 8
    vocab_size: int = 32000

    # Layer config
    state_dim: int = 128
    kalman_state_dim: int = 64
    n_heads: int = 8
    ffn_expand: float = 2.67

    # Memory hub
    tier1_dim: int = 256
    tier2_dim: int = 128
    tier3_dim: int = 64
    tier2_interval: int = 16
    tier3_interval: int = 128

    # Routing
    router_hidden: int = 64
    min_depth: int = 2
    exit_threshold: float = 0.5
    probe_threshold: float = 0.35
    probe_window: int = 256

    # Training
    dropout: float = 0.0
    max_seq_len: int = 2048
    tie_embeddings: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SyntheConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def tiny(cls) -> "SyntheConfig":
        """~15M params — for unit tests."""
        return cls(d_model=256, n_blocks=4, n_heads=4, state_dim=64, kalman_state_dim=32)

    @classmethod
    def small_60m(cls) -> "SyntheConfig":
        """~60M params — for Zoology benchmarks on 5060."""
        return cls(d_model=512, n_blocks=8, n_heads=8, state_dim=128, kalman_state_dim=64)

    @classmethod
    def medium_125m(cls) -> "SyntheConfig":
        """~125M params — for C4/SlimPajama pretraining."""
        return cls(d_model=768, n_blocks=12, n_heads=12, state_dim=192, kalman_state_dim=64)

    @classmethod
    def large_350m(cls) -> "SyntheConfig":
        """~350M params — for full benchmark suite."""
        return cls(d_model=1024, n_blocks=16, n_heads=16, state_dim=256, kalman_state_dim=96)


@dataclass
class SyntheModelState:
    """Complete model state for generation / streaming."""
    block_states: List[BlockState]
    memory_state: MemoryState
    step: int = 0

    def detach(self) -> "SyntheModelState":
        return SyntheModelState(
            block_states=[s.detach() for s in self.block_states],
            memory_state=self.memory_state.detach(),
            step=self.step,
        )


class Synthe(nn.Module):
    """
    SYNTHE: Synthesis of Online Learning Primitives
    for Hierarchical Temporal Engines.
    
    A causal language model where every layer is an online learner.
    """

    def __init__(self, config: SyntheConfig):
        super().__init__()
        self.config = config

        # === Embedding ===
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model)
        self.embed_drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Positional embedding (learnable)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # === SYNTHE Blocks ===
        self.blocks = nn.ModuleList([
            SyntheBlock(
                d_model=config.d_model,
                state_dim=config.state_dim,
                kalman_state_dim=config.kalman_state_dim,
                n_heads=config.n_heads,
                ffn_expand=config.ffn_expand,
                probe_window=config.probe_window,
                probe_threshold=config.probe_threshold,
                dropout=config.dropout,
            )
            for _ in range(config.n_blocks)
        ])

        # === Memory Hub ===
        self.memory_hub = TemporalMemoryHub(
            d_model=config.d_model,
            tier1_dim=config.tier1_dim,
            tier2_dim=config.tier2_dim,
            tier3_dim=config.tier3_dim,
            tier2_interval=config.tier2_interval,
            tier3_interval=config.tier3_interval,
        )

        # === Depth Router ===
        self.router = DepthRouter(
            d_model=config.d_model,
            hidden_dim=config.router_hidden,
            n_blocks=config.n_blocks,
            min_depth=config.min_depth,
            exit_threshold=config.exit_threshold,
            probe_threshold=config.probe_threshold,
        )

        # === Output Head ===
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize
        self.apply(self._init_weights)
        self._count_params()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._total_params = total
        self._trainable_params = trainable

    @property
    def num_params(self) -> int:
        return self._total_params

    @property
    def num_params_str(self) -> str:
        p = self._total_params
        if p >= 1e9:
            return f"{p/1e9:.1f}B"
        elif p >= 1e6:
            return f"{p/1e6:.1f}M"
        elif p >= 1e3:
            return f"{p/1e3:.1f}K"
        return str(p)

    def init_state(self, batch_size: int, device: torch.device) -> SyntheModelState:
        return SyntheModelState(
            block_states=[
                block.init_state(batch_size, device)
                for block in self.blocks
            ],
            memory_state=self.memory_hub.init_state(batch_size, device),
            step=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,                     # (B, T)
        state: Optional[SyntheModelState] = None,
        targets: Optional[torch.Tensor] = None,       # (B, T) for loss
        return_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Returns dict with:
            logits: (B, T, vocab_size)
            loss: scalar (if targets provided)
            state: SyntheModelState (if return_state=True)
            info: routing/confidence statistics
        """
        B, T = input_ids.shape
        device = input_ids.device

        if state is None:
            state = self.init_state(B, device)

        # === Embedding ===
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        # Offset positions by current step for streaming
        positions = (positions + state.step) % self.config.max_seq_len

        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.embed_norm(x)
        x = self.embed_drop(x)

        # === Process through blocks ===
        new_block_states = []
        compute_budgets = []
        block_infos = []

        # Get memory confidence for routing
        memory_confidence = self.memory_hub.get_confidence(state.memory_state)

        for i, block in enumerate(self.blocks):
            # Routing decision
            kalman_conf = state.block_states[i].kalman_state.confidence if i > 0 else None
            continue_mask, probe_mask, budget = self.router(
                x, block_idx=i,
                kalman_confidence=kalman_conf,
                memory_confidence=memory_confidence,
            )
            compute_budgets.append(budget)

            # Process block
            x, block_state, info = block(
                x, state=state.block_states[i],
                compute_budget=budget if i >= self.config.min_depth else None,
            )

            new_block_states.append(block_state)
            block_infos.append(info)

        # === Memory Hub ===
        memory_out, new_memory_state = self.memory_hub(x, state.memory_state)
        x = x + memory_out  # Additive memory contribution

        # === Output ===
        x = self.out_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits}

        # === Loss ===
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        # === State ===
        if return_state:
            result["state"] = SyntheModelState(
                block_states=new_block_states,
                memory_state=new_memory_state,
                step=state.step + T,
            )

        # === Info ===
        routing_stats = self.router.get_utilization(compute_budgets)
        avg_confidence = sum(i["kalman_confidence"] for i in block_infos) / len(block_infos)
        avg_surprise = sum(i["delta_surprise"] for i in block_infos) / len(block_infos)
        avg_probe = sum(i["probe_activated"] for i in block_infos) / len(block_infos)

        result["info"] = {
            "routing": routing_stats,
            "avg_kalman_confidence": avg_confidence,
            "avg_delta_surprise": avg_surprise,
            "avg_probe_activation": avg_probe,
            "num_params": self.num_params_str,
        }

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,    # (B, T) prompt
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with state caching."""
        self.eval()
        B, T = input_ids.shape

        # Process prompt
        result = self.forward(input_ids, return_state=True)
        state = result["state"]

        generated = [input_ids]

        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = result["logits"][:, -1, :]  # (B, vocab_size)

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token)

            # Forward single token with cached state
            result = self.forward(next_token, state=state, return_state=True)
            state = result["state"]

        return torch.cat(generated, dim=1)
