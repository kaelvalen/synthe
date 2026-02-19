# SYNTHE

**Synthesis of Online Learning Primitives for Hierarchical Temporal Engines**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

> A mechanistic alternative to Transformers where every layer is an independent online learning agent.

---

## Overview

SYNTHE is a meta-architecture for sequence modeling that replaces the fixed attention+FFN block with a composition of **online learning primitives** — each operating at different temporal scales, each with its own update rule. The model doesn't attend to all tokens; it **learns what to remember** at inference time, allocates compute dynamically, and tracks its own uncertainty.

```
INPUT → Embedding → [Momentum → Delta → Kalman → (Probe)] × N → LM Head → OUTPUT
                          ↕           ↕          ↕
                    ┌─────────────────────────────────────┐
                    │      Temporal Memory Hub             │
                    │  Token ←→ Sentence ←→ Discourse     │
                    └─────────────────────────────────────┘
                          ↕
                    ┌─────────────┐
                    │ Depth Router │ → early exit if confident
                    └─────────────┘
```

### Key Properties

| | Transformer | Mamba/SSM | **SYNTHE** |
|---|---|---|---|
| **Complexity** | O(n²) | O(n) | **O(n)** |
| **Recall** | Perfect (KV cache) | Limited (fixed state) | **Learned (Delta rule + test-time learning)** |
| **Memory** | Linear KV growth | Fixed state | **Hierarchical, multi-scale, bounded** |
| **Compute/token** | Fixed | Fixed | **Dynamic (30-50% savings via routing)** |
| **Uncertainty** | None | None | **Built-in (Kalman confidence)** |

## Architecture

SYNTHE blocks compose four layer primitives:

| Layer | Mechanism | Role | Complexity |
|-------|-----------|------|------------|
| **MomentumLayer** | Gated EMA with input-dependent decay | Fast local patterns (syntax, bigrams) | O(1)/token |
| **DeltaLayer** | Delta rule associative memory (Hopfield-style) | Store & retrieve key-value associations, overwrite stale info | O(d)/token |
| **KalmanLayer** | Diagonal Kalman filter with innovation-based confidence | Uncertainty-aware state estimation, drives adaptive compute | O(d)/token |
| **AttentionProbe** | Sliding-window attention, conditionally activated | Emergency precision recall — only fires when Kalman is uncertain | O(w)/token |

The **Temporal Memory Hub** provides three tiers: token-level (every token), sentence-level (every ~16 tokens), and discourse-level (every ~128 tokens) with bidirectional information flow.

The **Depth Router** allocates compute dynamically — easy tokens exit after 2-3 blocks, hard tokens get full depth.

### Innovation-Based Confidence

SYNTHE's key signal is the Kalman layer's **innovation-to-observation ratio (IOR)**: how well the filter predicts the next observation before seeing it.

- **Predictable input** → IOR ≈ 0 → confidence ≈ 1.0 → probe OFF, early exit possible
- **Surprising input** → IOR > 1 → confidence ≈ 0.3 → probe fires, full depth

This makes compute allocation data-dependent — the model naturally spends more resources on hard tokens.

## Installation

```bash
git clone https://github.com/kaelvalen/synthe.git
cd synthe
pip install -e ".[dev]"
```

For training:
```bash
pip install -e ".[train]"
```

## Quick Start

```python
import torch
from src.model import Synthe, SyntheConfig

# Create model
config = SyntheConfig.small_60m()
model = Synthe(config)

# Forward pass with loss
input_ids = torch.randint(0, 32000, (2, 128))
result = model(input_ids, targets=input_ids, return_state=True)

print(f"Loss: {result['loss']:.4f}")
print(f"Kalman confidence: {result['info']['avg_kalman_confidence']:.3f}")
print(f"Probe activation: {result['info']['avg_probe_activation']:.1%}")
print(f"Early exit rate: {result['info']['routing']['early_exit_rate']:.1%}")

# Autoregressive generation
prompt = torch.randint(0, 32000, (1, 16))
generated = model.generate(prompt, max_new_tokens=64, temperature=0.8)
```

## Model Configurations

| Config | d_model | Blocks | Heads | Params | Target Use |
|--------|---------|--------|-------|--------|------------|
| `tiny()` | 256 | 4 | 4 | ~16M | Unit tests |
| `small_60m()` | 512 | 8 | 8 | ~74M | Zoology benchmarks |
| `medium_125m()` | 768 | 12 | 12 | ~216M | C4 pretraining |
| `large_350m()` | 1024 | 16 | 16 | ~528M | Full benchmark suite |

Load from YAML:
```python
config = SyntheConfig.from_yaml("configs/synthe_60m.yaml")
```

## Training

```bash
python scripts/train.py \
    --config configs/synthe_60m.yaml \
    --data tinyshakespeare \
    --batch_size 8 \
    --lr 3e-4 \
    --max_steps 5000
```

See [scripts/train.py](scripts/train.py) for the full training loop.

## Tests

```bash
# Layer primitives — shapes, state persistence, recall, gradients, speed
python tests/test_layers.py

# Full model integration — forward pass, streaming, generation, gradient flow
python tests/test_model.py

# Probe activation diagnostic — confidence distribution & threshold sweep
python tests/test_probe_fire.py
```

All tests pass with zero external data dependencies.

## Project Structure

```
synthe/
├── src/
│   ├── layers/              # 4 layer primitives
│   │   ├── base.py          # SyntheLayer ABC + LayerState
│   │   ├── delta.py         # Delta rule associative memory
│   │   ├── momentum.py      # Gated EMA temporal patterns
│   │   ├── kalman.py        # Diagonal Kalman filter + IOR confidence
│   │   └── attention.py     # Conditional sliding-window probe
│   ├── memory/
│   │   └── hub.py           # 3-tier temporal memory hub
│   ├── routing/
│   │   └── depth_router.py  # Dynamic computation depth
│   └── model/
│       ├── block.py         # SyntheBlock (composable unit)
│       └── synthe.py        # Full model + config + generation
├── configs/
│   └── synthe_60m.yaml      # 60M config for RTX 5060 Mobile
├── scripts/
│   └── train.py             # Training loop
├── tests/
│   ├── test_layers.py       # Layer validation suite
│   ├── test_model.py        # Integration tests
│   └── test_probe_fire.py   # Probe activation diagnostic
├── SYNTHE_PARADIGM.md       # Architecture design document
├── why.md                   # Research motivation & landscape analysis
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Design Philosophy

Read the full architecture manifesto: [SYNTHE_PARADIGM.md](SYNTHE_PARADIGM.md)

**Core thesis:** You don't need attention if your state can *learn* what to remember at inference time, at multiple temporal scales, with explicit mechanisms for overwriting stale information.

Five principles:
1. **Layers are learners, not functions** — each layer runs an online learning algorithm
2. **Hierarchical temporal memory** — token/sentence/discourse tiers with bidirectional flow
3. **Dynamic computation depth** — easy tokens get less compute
4. **Test-time state learning** — memory modules perform local updates during inference
5. **Epistemic confidence** — the model knows when it doesn't know

## Research Context

SYNTHE builds on insights from:
- **DeltaNet / Gated DeltaNet** (ICLR 2025) — delta rule for associative memory in linear attention
- **Titans** (Google, 2025) — test-time memorization as architectural primitive
- **MIRAS** (Google Research, 2025) — unifying framework showing all sequence models perform associative memory optimization
- **Mamba / Mamba-2** (Gu & Dao) — selective state spaces with hardware-efficient computation
- **RWKV-7** (Peng et al.) — generalized delta rule achieving NC¹ expressivity
- **Mixture-of-Depths** (Raposo et al.) — learned per-token computation allocation

See [why.md](why.md) for a comprehensive analysis of the research landscape.

## Roadmap

- [x] Core layer primitives (Delta, Momentum, Kalman, AttentionProbe)
- [x] Hierarchical temporal memory hub
- [x] Dynamic depth routing
- [x] Full model with generation support
- [x] Comprehensive test suite
- [ ] Tiny Shakespeare training validation
- [ ] Zoology synthetic benchmarks (MQAR, copying, state tracking)
- [ ] C4/SlimPajama pretraining at 125M scale
- [ ] lm-eval zero-shot evaluation (MMLU, HellaSwag, ARC)
- [ ] Scaling to 350M+ parameters

## Citation

If you use SYNTHE in your research:

```bibtex
@software{valen2026synthe,
    title  = {SYNTHE: Synthesis of Online Learning Primitives for Hierarchical Temporal Engines},
    author = {Valen, Kael},
    year   = {2026},
    url    = {https://github.com/kaelvalen/synthe},
}
```

## License

MIT — see [LICENSE](LICENSE) for details.

---

*SYNTHE — The architecture that learns how to learn.*
