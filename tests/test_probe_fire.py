"""Quick test: does the probe actually fire now?"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.layers import WienerCore

B, T, D = 4, 64, 128
layer = WienerCore(d_model=D, state_dim=32, n_heads=4)

# Random data
x = torch.randn(B, T, D)
_, s = layer(x)

# Repeated pattern
pat = torch.randn(1, 1, D).expand(B, T, D)
_, sp = layer(pat)

# High noise
noise = torch.randn(B, T, D) * 10
_, sn = layer(noise)

print("=== Confidence Distribution ===")
print(f"  Random:  mean={s.confidence.mean():.4f}  min={s.confidence.min():.4f}  max={s.confidence.max():.4f}  std={s.confidence.std():.4f}")
print(f"  Pattern: mean={sp.confidence.mean():.4f}  min={sp.confidence.min():.4f}  max={sp.confidence.max():.4f}  std={sp.confidence.std():.4f}")
print(f"  Noise:   mean={sn.confidence.mean():.4f}  min={sn.confidence.min():.4f}  max={sn.confidence.max():.4f}  std={sn.confidence.std():.4f}")

print("\n=== Threshold Sweep (random data) ===")
for t in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
    rate = (s.confidence < t).float().mean().item()
    bar = "#" * int(rate * 40)
    print(f"  thresh={t:.2f}  fire={rate:5.1%}  {bar}")

print("\n=== Threshold Sweep (pattern) ===")
for t in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
    rate = (sp.confidence < t).float().mean().item()
    bar = "#" * int(rate * 40)
    print(f"  thresh={t:.2f}  fire={rate:5.1%}  {bar}")

print("\n=== Threshold Sweep (noise) ===")
for t in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
    rate = (sn.confidence < t).float().mean().item()
    bar = "#" * int(rate * 40)
    print(f"  thresh={t:.2f}  fire={rate:5.1%}  {bar}")
