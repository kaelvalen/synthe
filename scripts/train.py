"""
SYNTHE Training Script

Minimal but complete training loop for SYNTHE language models.
Supports character-level (Tiny Shakespeare) and token-level training.

Usage:
    # Tiny Shakespeare (downloads automatically)
    python scripts/train.py --data tinyshakespeare --max_steps 5000

    # Custom text file
    python scripts/train.py --data path/to/corpus.txt --max_steps 10000

    # With config file
    python scripts/train.py --config configs/synthe_60m.yaml --data tinyshakespeare
"""

import sys
import os
import math
import time
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import Synthe, SyntheConfig


# =============================================================================
# Dataset
# =============================================================================

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharDataset(Dataset):
    """Character-level text dataset for quick training validation."""

    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len
        # Build character vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_tinyshakespeare(data_dir: str = "data") -> str:
    """Download Tiny Shakespeare if needed, return text."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"  Downloading Tiny Shakespeare → {path}")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    with open(path, "r") as f:
        return f.read()


def load_text(path: str) -> str:
    """Load text from a file."""
    with open(path, "r") as f:
        return f.read()


# =============================================================================
# Training
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"  SYNTHE Training")
    print(f"{'='*60}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")
        print(f"  TF32: enabled")
    else:
        print(f"  Device: {device}")

    # --- Data ---
    if args.data == "tinyshakespeare":
        text = load_tinyshakespeare()
        print(f"  Data: Tiny Shakespeare ({len(text):,} chars)")
    else:
        text = load_text(args.data)
        print(f"  Data: {args.data} ({len(text):,} chars)")

    dataset = CharDataset(text, seq_len=args.seq_len)

    # Train/val split
    n = len(dataset)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    print(f"  Train: {n_train:,} sequences | Val: {n_val:,} sequences")
    print(f"  Vocab size: {dataset.vocab_size} (character-level)")

    # --- Model ---
    if args.config:
        config = SyntheConfig.from_yaml(args.config)
    else:
        config = SyntheConfig.small_60m()

    # Override vocab for character-level
    config.vocab_size = dataset.vocab_size
    config.max_seq_len = args.seq_len

    model = Synthe(config).to(device)

    # Compile for speed (PyTorch 2.x)
    if args.compile and hasattr(torch, 'compile'):
        print(f"  Compiling model with torch.compile...")
        model = torch.compile(model)

    print(f"  Model: {model.num_params_str if hasattr(model, 'num_params_str') else '?'} parameters")
    print(f"  Config: d={config.d_model}, blocks={config.n_blocks}, heads={config.n_heads}")
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU memory (model+optim): ~{allocated:.1f}GB")
    print()

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine annealing with warmup
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # --- Training Loop ---
    model.train()
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    log_interval = args.log_every
    eval_interval = args.eval_every
    start_time = time.time()
    accum_steps = args.grad_accum
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tokens_per_step = args.batch_size * args.seq_len * accum_steps
    print(f"  Training for {args.max_steps:,} steps...")
    print(f"  Effective batch: {args.batch_size}×{accum_steps} = {args.batch_size * accum_steps} seqs")
    print(f"  Tokens/step: {tokens_per_step:,}")
    if use_amp:
        print(f"  Mixed precision: BF16")
    print(f"  {'='*60}")

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(accum_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Forward with optional AMP
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                result = model(x, targets=y)
                loss = result["loss"] / accum_steps

            # Backward
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping + step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = accum_loss  # already accumulated
        step += 1

        # --- Logging ---
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step * tokens_per_step) / elapsed
            lr_now = scheduler.get_last_lr()[0]
            info = result.get("info", {})
            probe_act = info.get("avg_probe_activation", 0)
            wiener_conf = info.get("avg_wiener_confidence", 0)
            routing = info.get("routing", {})
            exit_rate = routing.get("early_exit_rate", 0)

            gpu_info = ""
            if device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated(0) / 1e9
                gpu_info = f" | gpu {mem_gb:.1f}GB"

            print(
                f"  step {step:>6d} | loss {loss_val:.4f} | "
                f"lr {lr_now:.2e} | grad {grad_norm:.2f} | "
                f"probe {probe_act:.0%} | conf {wiener_conf:.2f} | "
                f"exit {exit_rate:.0%} | {tok_per_sec:,.0f} tok/s{gpu_info}"
            )

        # --- Evaluation ---
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            improved = " ★" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.save_path:
                    save_checkpoint(model, optimizer, step, val_loss, args.save_path)
            print(f"  {'─'*60}")
            print(f"  eval  step {step:>6d} | val_loss {val_loss:.4f} | best {best_val_loss:.4f}{improved}")
            print(f"  {'─'*60}")
            model.train()

    # --- Final eval + sample ---
    total_time = time.time() - start_time
    val_loss = evaluate(model, val_loader, device)
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Steps: {step:,} | Time: {total_time:.1f}s | Final val loss: {val_loss:.4f}")
    print(f"{'='*60}\n")

    # Generate a sample
    print("  Sample generation:")
    print("  " + "─" * 56)
    sample = generate_sample(model, dataset, device, max_tokens=200)
    for line in sample.split("\n")[:10]:
        print(f"  {line}")
    print("  " + "─" * 56)

    if args.save_path:
        save_checkpoint(model, optimizer, step, val_loss, args.save_path)
        print(f"\n  Checkpoint saved → {args.save_path}")


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        result = model(x, targets=y)
        total_loss += result["loss"].item()
        n_batches += 1
        if n_batches >= 50:  # Cap eval batches
            break
    model.train()
    return total_loss / max(1, n_batches)


@torch.no_grad()
def generate_sample(model, dataset, device, prompt_text=None, max_tokens=200):
    """Generate text from the model."""
    model.eval()
    if prompt_text is None:
        # Use first 32 chars of dataset as prompt
        prompt_ids = dataset.data[:32].unsqueeze(0).to(device)
    else:
        prompt_ids = torch.tensor(
            [[dataset.stoi.get(c, 0) for c in prompt_text]], dtype=torch.long, device=device
        )

    output = model.generate(
        prompt_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=40,
    )
    return dataset.decode(output[0].cpu().tolist())


def save_checkpoint(model, optimizer, step, val_loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config.__dict__,
    }, path)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train SYNTHE language model")
    # Data
    parser.add_argument("--data", type=str, default="tinyshakespeare",
                        help="'tinyshakespeare' or path to text file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: tiny preset)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile (PyTorch 2.x)")

    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="checkpoints/synthe.pt",
                        help="Checkpoint save path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
