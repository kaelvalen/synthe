"""
SYNTHE Full Model Integration Test

Validates the complete architecture end-to-end:
- Forward pass with loss computation
- Autoregressive generation
- State caching and streaming
- Parameter counts at different scales
- Gradient flow through all components
"""

import sys
sys.path.insert(0, "/home/kael/synthe")

import torch
import time
from typing import cast

from src.model import Synthe, SyntheConfig
from src.model.block import SyntheBlock


def test_forward_pass():
    """Basic forward pass with loss."""
    print("=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)

    config = SyntheConfig.tiny()
    model = Synthe(config)

    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    result = model(input_ids, targets=targets, return_state=True)

    print(f"  Model: {model.num_params_str} parameters")
    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  Expected initial loss: ~{torch.log(torch.tensor(float(config.vocab_size))).item():.4f}")
    print(f"  Info: {result['info']}")

    assert result["logits"].shape == (B, T, config.vocab_size)
    assert result["loss"].item() > 0
    assert result["state"] is not None
    print(f"  ✓ Forward pass OK")
    print()


def test_state_streaming():
    """Process sequence in chunks with state passing."""
    print("=" * 60)
    print("TEST: State Streaming (Chunked Processing)")
    print("=" * 60)

    config = SyntheConfig.tiny()
    model = Synthe(config)
    model.eval()

    B, T = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T))

    # Process in chunks of 32
    chunk_size = 32
    state = None
    chunked_logits = []

    with torch.no_grad():
        for i in range(0, T, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            result = model(chunk, state=state, return_state=True)
            state = result["state"]
            chunked_logits.append(result["logits"])

    chunked_output = torch.cat(chunked_logits, dim=1)
    print(f"  Chunked output shape: {chunked_output.shape}")
    if state is not None:
        print(f"  State step count: {state.step}")

    assert chunked_output.shape == (B, T, config.vocab_size)
    assert state is not None and state.step == T
    print(f"  ✓ State streaming OK")
    print()


def test_generation():
    """Autoregressive generation."""
    print("=" * 60)
    print("TEST: Autoregressive Generation")
    print("=" * 60)

    config = SyntheConfig.tiny()
    model = Synthe(config)

    B = 2
    prompt = torch.randint(0, config.vocab_size, (B, 16))

    start = time.perf_counter()
    output = model.generate(prompt, max_new_tokens=32, temperature=0.8)
    elapsed = time.perf_counter() - start

    tokens_generated = output.shape[1] - prompt.shape[1]
    tok_per_sec = tokens_generated * B / elapsed

    print(f"  Prompt: {prompt.shape} → Output: {output.shape}")
    print(f"  Generated {tokens_generated} tokens in {elapsed:.2f}s")
    print(f"  Speed: {tok_per_sec:.1f} tok/s")

    assert output.shape == (B, 16 + 32, )
    print(f"  ✓ Generation OK")
    print()


def test_gradient_flow_full():
    """Gradients flow through entire model."""
    print("=" * 60)
    print("TEST: Full Model Gradient Flow")
    print("=" * 60)

    config = SyntheConfig.tiny()
    model = Synthe(config)

    B, T = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    result = model(input_ids, targets=targets)
    result["loss"].backward()

    # Check gradient flow to all major components
    block0 = cast(SyntheBlock, model.blocks[0])
    blockN = cast(SyntheBlock, model.blocks[-1])
    
    components = {
        "embedding": model.embedding.weight,
        "block[0].momentum.W_k": block0.momentum.W_k.weight,
        "block[0].delta.W_k": block0.delta.W_k.weight,
        "block[0].kalman.W_obs": block0.kalman.W_obs.weight,
        "block[-1].ffn.w1": blockN.ffn.w1.weight,
        "memory_hub.fusion": model.memory_hub.fusion.weight,
        "router.estimator[0]": model.router.estimator[0].weight,
        "lm_head": model.lm_head.weight,
    }

    all_ok = True
    for name, param in components.items():
        has_grad = param.grad is not None and param.grad.norm() > 0
        grad_norm = param.grad.norm().item() if has_grad else 0
        status = "✓" if has_grad else "✗"
        print(f"  {status} {name:40s} | grad_norm={grad_norm:.6f}")
        if not has_grad:
            all_ok = False

    assert all_ok, "Some components have no gradient!"
    print()


def test_all_configs():
    """Parameter counts for all predefined configs."""
    print("=" * 60)
    print("TEST: Model Configurations")
    print("=" * 60)

    configs = {
        "tiny (test)": SyntheConfig.tiny(),
        "60M (Zoology)": SyntheConfig.small_60m(),
        "125M (C4)": SyntheConfig.medium_125m(),
        "350M (full)": SyntheConfig.large_350m(),
    }

    for name, config in configs.items():
        model = Synthe(config)

        # Component breakdown
        embed_params = sum(p.numel() for p in model.embedding.parameters())
        embed_params += sum(p.numel() for p in model.pos_embedding.parameters())
        block_params = sum(p.numel() for p in model.blocks.parameters())
        memory_params = sum(p.numel() for p in model.memory_hub.parameters())
        router_params = sum(p.numel() for p in model.router.parameters())
        head_params = sum(p.numel() for p in model.lm_head.parameters())
        norm_params = sum(p.numel() for p in model.out_norm.parameters()) + \
                      sum(p.numel() for p in model.embed_norm.parameters())

        print(f"  {name}")
        print(f"    d_model={config.d_model}, n_blocks={config.n_blocks}, n_heads={config.n_heads}")
        print(f"    Embedding:    {embed_params:>12,}")
        print(f"    Blocks:       {block_params:>12,}")
        print(f"    Memory Hub:   {memory_params:>12,}")
        print(f"    Router:       {router_params:>12,}")
        print(f"    Head+Norms:   {head_params + norm_params:>12,}")
        print(f"    TOTAL:        {model.num_params:>12,}  ({model.num_params_str})")
        print()

        del model


def test_speed_benchmark():
    """Forward pass speed."""
    print("=" * 60)
    print("TEST: Speed Benchmark")
    print("=" * 60)

    config = SyntheConfig.tiny()
    model = Synthe(config)
    model.eval()

    B, T = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T))

    # Warmup
    with torch.no_grad():
        model(input_ids)

    n_runs = 5
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(input_ids)
    elapsed = (time.perf_counter() - start) / n_runs

    tok_per_sec = (B * T) / elapsed
    print(f"  Config: {model.num_params_str}, B={B}, T={T}")
    print(f"  Forward: {elapsed*1000:.1f} ms")
    print(f"  Throughput: {tok_per_sec:,.0f} tok/s")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SYNTHE FULL MODEL INTEGRATION TEST")
    print("=" * 60 + "\n")

    test_forward_pass()
    test_state_streaming()
    test_generation()
    test_gradient_flow_full()
    test_all_configs()
    test_speed_benchmark()

    print("=" * 60)
    print("  ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 60)
