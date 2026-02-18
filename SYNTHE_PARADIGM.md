# SYNTHE

## Synthesis of Online Learning Primitives for Hierarchical Temporal Engines

**"Architecture is dead. Learning systems are the paradigm."**

---

## Manifesto

SYNTHE is not a model. It is not "another hybrid." It is a meta-architecture: a configurable system where every layer is an independent online learning agent, operating at its own temporal scale, with its own update rule, dynamically allocating computation based on input complexity. The "architecture" is the configuration of these agents — swap the learning rule, change the behavior. The model is not a fixed circuit; it is a living synthesis.

Where Transformers memorize everything and hope attention finds it, where SSMs compress everything and hope the state retains it — SYNTHE learns what to remember, when to forget, and how deeply to think, all at inference time.

---

## Origin DNA

SYNTHE inherits from three prior systems:

| Project | Contribution to SYNTHE |
|---------|----------------------|
| **PULSE** (Parallel Unified Linear State Engine) | Linear state dynamics, parallel processing, MaaS memory service concept |
| **BWR-DNC** (Dynamic Neural Core) | Multi-scale memory hierarchy, learned compression, real-time state visualization |
| **Velocity** (NNEI) | Epistemic evaluation, hypothesis generation/elimination, confidence calibration |

These are not abandoned projects — they are SYNTHE's ancestry.

---

## The Problem Space

Every current architecture makes the same trade-off:

```
TRANSFORMER:  Perfect recall  ←→  O(n²) cost, static weights
MAMBA/SSM:    O(n) efficiency  ←→  Fixed state, recall collapse  
RWKV:         Linear + expressive  ←→  Scaling uncertainty
xLSTM:        State tracking  ←→  General language modeling weakness
```

The field's response has been to **hybridize** — stack SSM layers with occasional attention layers (Jamba: 1:7 ratio, Nemotron-H: 4 attention layers in 8B model). This works, but it's architectural duct tape. Nobody is asking: *why do we need attention at all? What fundamental operation is it providing that recurrence cannot?*

**The answer:** Associative recall from unbounded context. Attention can look at any past token with equal ease. Recurrence must compress everything into fixed state.

**SYNTHE's thesis:** You don't need attention if your state can *learn* what to remember at inference time, at multiple temporal scales, with explicit mechanisms for overwriting stale information.

---

## Core Principles

### 1. Layers Are Learners, Not Functions

In a Transformer, each layer applies a fixed function (attention + FFN). In SYNTHE, each layer runs an **online learning algorithm** that updates its state based on incoming tokens. The choice of algorithm defines the layer's behavior:

| Layer Type | Update Rule | Inspired By | Strength |
|-----------|-------------|-------------|----------|
| **Delta Layer** | Δw = α(target - prediction) · input | DeltaNet / Hopfield | Overwrites stale info, precise recall |
| **Kalman Layer** | Bayesian state estimation with uncertainty | Kalman filtering | Uncertainty-aware, stable updates |
| **Momentum Layer** | Exponential moving average with gating | RWKV-7 / mLSTM | Fast temporal patterns, efficiency |
| **Attention Probe** | Sparse local attention (emergency) | Sliding window | Precision fallback for critical recall |

A SYNTHE block is a **stack of 2-4 of these layer types**, configured per task/deployment. The key insight: different positions in the network benefit from different learning dynamics.

### 2. Hierarchical Temporal Memory

Three memory tiers operating at different timescales:

```
┌─────────────────────────────────────────────┐
│  TIER 3: Discourse Memory (slow)            │
│  Updates: every ~100 tokens                 │
│  Rule: Kalman estimation                    │
│  Stores: topic, style, long-range deps      │
│  State size: small (compressed summaries)   │
├─────────────────────────────────────────────┤
│  TIER 2: Sentence Memory (medium)           │
│  Updates: every ~10-20 tokens               │
│  Rule: Delta rule (overwrite stale)         │
│  Stores: entity tracking, clause relations  │
│  State size: medium                         │
├─────────────────────────────────────────────┤
│  TIER 1: Token Memory (fast)                │
│  Updates: every token                       │
│  Rule: Gated momentum (RWKV-style)          │
│  Stores: local syntax, immediate context    │
│  State size: large (high bandwidth)         │
└─────────────────────────────────────────────┘
```

Information flows **both up and down**: Tier 1 feeds compressed signals to Tier 2, Tier 2 to Tier 3. But Tier 3 also broadcasts "context priors" down to Tier 1, biasing local processing. This is inspired by the brain's hippocampal-neocortical memory consolidation — and by BWR-DNC's multi-scale compression hierarchy.

### 3. Dynamic Computation Depth

Not every token deserves the same compute. SYNTHE implements **continuous depth routing**:

```
Input token → Difficulty estimator (lightweight) → Compute budget
    │
    ├── Easy token (function words, predictable) → 2-3 layers
    ├── Medium token (content, moderate ambiguity) → 6-8 layers  
    └── Hard token (rare, ambiguous, critical) → full depth + Tier 2/3 memory query
```

This is Mixture-of-Depths but continuous, not binary. The difficulty estimator is a tiny network trained jointly. Expected FLOP savings: 30-50% on average text.

### 4. Test-Time State Learning

During inference, Tier 2 and Tier 3 memories perform actual gradient updates on their internal states — not backpropagation through the whole model, but local online learning within each memory module. This is the Titans insight, extended to multiple scales:

- **Surprise signal**: When the model's prediction is wrong, Tier 2 increases its learning rate
- **Consolidation**: Periodically, Tier 1 patterns that persist are promoted to Tier 2 (learned compression, à la BWR-DNC)
- **Forgetting**: Delta rule in Tier 2 actively erases associations that conflict with new evidence

This means SYNTHE's effective context window is **theoretically unbounded** — not because it stores everything, but because it *learns* what matters.

### 5. Epistemic Confidence

Borrowed from Velocity's NNEI paradigm: SYNTHE tracks confidence in its own state. Each memory tier maintains an uncertainty estimate:

- **High confidence** → use cached state, skip computation
- **Low confidence** → allocate more compute, query higher tiers
- **Conflicting signals** → engage Attention Probe as tiebreaker

This creates an inherent calibration mechanism — the model knows when it doesn't know.

---

## Architecture Overview

```
INPUT TOKENS
     │
     ▼
┌─────────────┐
│  EMBEDDING  │
│  + Position │
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    SYNTHE BLOCK × N                     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Momentum │→ │  Delta   │→ │  Kalman  │  ← Layer      │
│  │  Layer   │  │  Layer   │  │  Layer   │    Learners   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │                     │
│       ▼             ▼             ▼                     │
│  ┌──────────────────────────────────────────┐           │
│  │         TEMPORAL MEMORY HUB              │           │
│  │  Tier 1 ←→ Tier 2 ←→ Tier 3              │           │
│  │  (token)   (sentence)  (discourse)       │           │
│  └──────────────────────────────────────────┘           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────┐                                           │
│  │  Depth   │ → skip remaining layers if confident      │
│  │  Router  │                                           │
│  └──────────┘                                           │
│                                                         │
│  [Optional: Attention Probe — activated by low          │
│   confidence or conflicting memory tiers]               │
│                                                         │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│   OUTPUT    │
│   HEAD      │
└─────────────┘
```

### Block Configuration (Default)

```yaml
synthe_block:
  layers:
    - type: momentum      # Fast local patterns
      state_dim: 256
      update: gated_ema
    - type: delta          # Associative recall  
      state_dim: 128
      update: delta_rule
      overwrite: true
    - type: kalman         # Uncertainty-aware state
      state_dim: 64
      update: bayesian
  
  memory:
    tier1_tokens: 1        # Update every token
    tier2_tokens: 16       # Update every 16 tokens
    tier3_tokens: 128      # Update every 128 tokens
    consolidation: true    # Promote persistent patterns
    
  routing:
    depth_estimator: mlp_tiny  # 2-layer, 64 hidden
    min_depth: 2
    max_depth: N  # total blocks
    
  attention_probe:
    enabled: true
    window_size: 256
    trigger: confidence < 0.3
```

---

## Why SYNTHE Is Different

| | Transformer | Mamba/SSM | Hybrid (Jamba) | **SYNTHE** |
|---|---|---|---|---|
| **Time complexity** | O(n²) | O(n) | O(n) amortized | O(n) |
| **Recall** | Perfect | Limited by state | Good (attention layers) | **Learned recall via Delta + test-time learning** |
| **Memory** | KV cache grows linearly | Fixed state | Reduced KV cache | **Hierarchical, multi-scale, bounded** |
| **Compute per token** | Fixed | Fixed | Fixed | **Dynamic (30-50% savings)** |
| **Context length** | Bounded by memory | Theoretically unbounded | 256K demonstrated | **Unbounded (online learning)** |
| **Uncertainty** | None | None | None | **Built-in confidence tracking** |
| **Modularity** | Low (all attention) | Low (all SSM) | Medium (stack layers) | **High (swap any learning rule)** |

---

## Proof-of-Concept Plan

### Phase 0: Validate Core Primitives (Week 1-2)
**Hardware:** RTX 5060 Mobile (8GB VRAM)  
**Scale:** 60M parameters

- Implement individual layer types: Momentum, Delta, Kalman
- Test each on Zoology synthetic tasks:
  - MQAR (Multi-Query Associative Recall) — tests recall
  - Copying — tests state retention  
  - State tracking — tests dynamic state management
- Compare against baseline Transformer and Mamba at same scale
- **Success metric:** Delta layer matches or exceeds Mamba on recall tasks

### Phase 1: Hierarchical Memory (Week 3-4)  
**Scale:** 60M-125M parameters

- Implement 3-tier temporal memory hub
- Test on extended-context synthetic tasks
- Validate information flow between tiers
- **Success metric:** Outperforms flat-state models on discourse-level tasks

### Phase 2: Dynamic Routing + Integration (Week 5-6)
**Scale:** 125M parameters

- Add depth router
- Combine all components into full SYNTHE block
- Benchmark FLOP savings vs quality tradeoff
- **Success metric:** 30%+ FLOP reduction with <1% quality loss

### Phase 3: Language Modeling (Week 7-10)
**Scale:** 125M → 350M parameters  
**Data:** C4 / SlimPajama

- Pretrain SYNTHE on real language data
- Zero-shot evaluation: MMLU, HellaSwag, ARC, GSM8K
- Compare against Transformer, Mamba, RWKV baselines
- **Success metric:** Competitive with Mamba-350M at lower FLOP cost

### Phase 4: Scale (Month 4-6)
**Hardware:** H200 server  
**Scale:** 1.3B → 7B parameters

- Validate scaling laws
- Apply MOHAWK distillation from larger Transformer
- Full benchmark suite
- **Target:** Match Mamba-2-7B quality at 50% inference cost

---

## Technical Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Delta rule unstable at scale | HIGH | Gated delta (à la Gated DeltaNet) + gradient clipping |
| Kalman layer too expensive | MEDIUM | Low-rank Kalman approximation, update only every K tokens |
| Depth router adds overhead | LOW | Tiny MLP (2-layer, 64 hidden), <1% of total compute |
| Tier interactions create instability | HIGH | Progressive training: Tier 1 first, add Tier 2 after convergence, then Tier 3 |
| Test-time learning drifts | MEDIUM | Bounded learning rate + periodic state reset option |

---

## Research Contributions (If This Works)

1. **First unified meta-architecture** where every layer is an explicit online learner
2. **Hierarchical test-time memorization** at multiple temporal scales — unexplored in literature
3. **Kalman-based state estimation** for sequence modeling (novel primitive)
4. **Dynamic continuous-depth routing** for non-Transformer architectures
5. **Epistemic confidence tracking** as architectural primitive, not post-hoc calibration

---

## File Structure (Planned)

```
synthe/
├── README.md
├── PARADIGM.md              # This document
├── src/
│   ├── layers/
│   │   ├── momentum.py      # Gated EMA / RWKV-style
│   │   ├── delta.py         # Delta rule layer
│   │   ├── kalman.py        # Kalman estimation layer
│   │   └── attention.py     # Sparse attention probe
│   ├── memory/
│   │   ├── tier1_token.py   # Fast token-level memory
│   │   ├── tier2_sentence.py # Medium sentence-level
│   │   ├── tier3_discourse.py # Slow discourse-level
│   │   └── hub.py           # Memory hub + consolidation
│   ├── routing/
│   │   └── depth_router.py  # Dynamic computation depth
│   ├── model/
│   │   ├── block.py         # SYNTHE block
│   │   └── synthe.py        # Full model
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   └── progressive.py   # Progressive training strategy
│   └── eval/
│       ├── zoology.py       # Synthetic benchmarks
│       └── lm_eval.py       # Language model evaluation
├── configs/
│   ├── synthe_60m.yaml      # PoC config
│   ├── synthe_125m.yaml     # Small config
│   └── synthe_350m.yaml     # Medium config
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── benchmark.py
└── docs/
    ├── ARCHITECTURE.md
    ├── MEMORY_SYSTEM.md
    └── SCALING.md
```

---

## Next Steps

**Immediate (this session):**
1. Decide on framework (PyTorch vs JAX) 
2. Decide on first layer to implement (recommended: Delta layer — highest novelty + recall is the key bottleneck)
3. Set up project structure

**This week:**
- Implement Delta layer + Momentum layer
- Set up Zoology benchmark harness
- First recall experiments at 60M

---

*SYNTHE — The architecture that learns how to learn.*

*Created by Kael Valen*
*February 2026*
