"""
LSTM Language Model Baseline for WikiText-2.

Architecture:
    Embedding → LSTM (multi-layer) → LayerNorm → Linear (output projection)

Features:
    - Configurable layers, embedding size, dropout
    - Weight tying (embedding ↔ output projection)
    - AdamW + cosine LR schedule with warmup
    - Mixed precision (bfloat16/float16)
    - PPL/BPC evaluation
    - Checkpoint save/load

Usage:
    cd LSTM_baseline
    # Train with default config:
    python src/lstm.py

    # Train with custom config:
    python src/lstm.py --config=src/config_lstm.py

    # Evaluate only:
    python src/lstm.py --eval_only --checkpoint=out/ckpt.pt
"""

import os
import sys
import math
import time
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class LSTMLM(nn.Module):
    """LSTM-based language model."""

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int,
                 dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.tie_weights = tie_weights

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Init
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LSTM Model: vocab_size={vocab_size}, n_embd={n_embd}, "
              f"n_layer={n_layer}, dropout={dropout}, tie_weights={tie_weights}")
        print(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, hidden=None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target token indices, or None for inference
            hidden: optional initial hidden state (h, c) tuple
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (if targets provided), else None
            hidden: final hidden state (h, c)
        """
        x = self.token_embedding(idx)          # (B, T, n_embd)
        x, hidden = self.lstm(x, hidden)        # (B, T, n_embd)
        x = self.ln(x)                         # (B, T, n_embd)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
            )

        return logits, loss, hidden

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text from a prompt."""
        hidden = None
        for _ in range(max_new_tokens):
            # Take the last block_size tokens if sequence gets too long
            idx_cond = idx[:, -256:]
            logits, _, hidden = self(idx_cond, hidden=hidden)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Sample a random batch from a 1D token array."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_batch_sequential(data: np.ndarray, batch_size: int, block_size: int,
                         device: str, start: int = 0):
    """Get a sequential batch (no random sampling), for evaluation.

    Returns (x, y) tensors or (None, None) if no more data available.
    """
    remaining = len(data) - start - 1  # -1 to ensure we can get y (shifted by 1)
    if remaining < block_size:
        return None, None

    usable = min(remaining, batch_size * block_size)
    num_blocks = usable // block_size

    x = torch.from_numpy(data[start:start + num_blocks * block_size].astype(np.int64))
    y = torch.from_numpy(data[start + 1:start + 1 + num_blocks * block_size].astype(np.int64))

    x = x.view(num_blocks, block_size)
    y = y.view(num_blocks, block_size)

    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model: nn.Module, data: np.ndarray, eval_iters: int,
                  batch_size: int, block_size: int, device: str) -> dict:
    """Estimate loss from random batches during training.

    This is fast enough to run frequently, but it is noisier than the final
    full-split evaluation used in the reported tables.
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, block_size, device)
        _, loss, _ = model(X, Y)
        losses[k] = loss.item()
    model.train()

    val_loss = losses.mean().item()
    ppl = math.exp(val_loss)
    bpc = val_loss / math.log(2)

    return {"loss": val_loss, "ppl": ppl, "bpc": bpc}


@torch.no_grad()
def evaluate_full(model: nn.Module, data: np.ndarray, batch_size: int,
                  block_size: int, device: str, max_batches: int = None) -> dict:
    """Evaluate sequentially over the full split for comparable PPL/BPC."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    i = 0

    while True:
        if max_batches is not None and i >= max_batches:
            break

        start = i * batch_size * block_size
        X, Y = get_batch_sequential(data, batch_size, block_size, device, start)
        if X is None:
            break

        _, loss, _ = model(X, Y)
        total_loss += loss.item() * X.numel()
        total_tokens += X.numel()
        i += 1

    model.train()

    val_loss = total_loss / total_tokens
    ppl = math.exp(val_loss)
    bpc = val_loss / math.log(2)

    return {"loss": val_loss, "ppl": ppl, "bpc": bpc, "tokens": total_tokens}


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------

def _write_report(out_dir: str, config: dict, best_val_loss: float,
                  val_stats: dict, test_stats: dict):
    """Generate a text report summarizing the training run."""
    n_embd = config.get("n_embd", "-")
    n_layer = config.get("n_layer", "-")
    block_size = config.get("block_size", "-")
    batch_size = config.get("batch_size", "-")
    dropout = config.get("dropout", "-")
    tie = config.get("tie_weights", True)
    lr = config.get("learning_rate", "-")
    wd = config.get("weight_decay", "-")
    gc = config.get("grad_clip", "-")
    max_iters = config.get("max_iters", "-")
    V = config.get("vocab_size", 50257)
    warmup = config.get("warmup_iters", "-")
    min_lr = config.get("min_lr", "-")

    # Estimate params
    emb = V * n_embd
    lstm_p = 4 * n_layer * (2 * n_embd * n_embd)
    total = emb + lstm_p

    lines = []
    L = lines.append
    L("=" * 58)
    L("  LSTM Language Model Baseline — Training Report")
    L("=" * 58)
    L("")
    L("  Dataset:      WikiText-2")
    L("  Tokenizer:    GPT-2 BPE  (vocab = {})".format(V))
    L("")
    L("  ── Model ──")
    L("    Embedding dim  : {}".format(n_embd))
    L("    LSTM layers    : {}".format(n_layer))
    L("    Block size     : {}".format(block_size))
    L("    Batch size     : {}".format(batch_size))
    L("    Dropout        : {}".format(dropout))
    L("    Weight tying   : {}".format(tie))
    L("    Total params   : {:.2f}M".format(total / 1e6))
    L("")
    L("  ── Optimizer ──")
    L("    Learning rate  : {}".format(lr))
    L("    Min LR         : {}".format(min_lr))
    L("    Warmup iters   : {}".format(warmup))
    L("    Weight decay   : {}".format(wd))
    L("    Gradient clip  : {}".format(gc))
    L("    Max iterations : {}".format(max_iters))
    L("")
    L("  ── Results ──")
    L("    Best Val   | loss={:.4f}  ppl={:.1f}  bpc={:.4f}".format(
        best_val_loss, math.exp(best_val_loss), best_val_loss / math.log(2)))
    L("    Val (full) | loss={:.4f}  ppl={:.1f}  bpc={:.4f}".format(
        val_stats["loss"], val_stats["ppl"], val_stats["bpc"]))
    L("    Test       | loss={:.4f}  ppl={:.1f}  bpc={:.4f}".format(
        test_stats["loss"], test_stats["ppl"], test_stats["bpc"]))
    L("")
    uniform = V
    test_ppl = test_stats["ppl"]
    if test_ppl > 0:
        L("    Uniform baseline PPL = {}  (x{:.1f} reduction)".format(
            uniform, uniform / test_ppl))
    L("")
    L("=" * 58)

    report = "\n".join(lines)
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"Report → {report_path}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(config: dict):
    """Main training loop."""

    # --- System setup ---
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = config.get("dtype", "auto")
    compile_model = config.get("compile", False)

    # Require GPU — CPU training is not supported
    if not torch.cuda.is_available():
        print("ERROR: GPU not detected. CPU training is not supported.")
        print("  Install GPU PyTorch (see requirements.txt):")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    if device == "cpu":
        print("ERROR: device='cpu' requested but GPU training is required.")
        print("  Set device='cuda' in config or use --device=cuda")
        sys.exit(1)

    # Resolve dtype — "auto" picks best available, explicit values get fallback
    if dtype_str == "auto":
        if torch.cuda.is_bf16_supported():
            dtype_str = "bfloat16"
        else:
            dtype_str = "float16"
    elif dtype_str == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("WARNING: bfloat16 not supported on this GPU. Falling back to float16.")
        dtype_str = "float16"

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype_str]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    print(f"Device: {device}, dtype: {dtype_str}, compile: {compile_model}")

    # --- Data ---
    data_dir = config["data_dir"]
    train_data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    test_data = np.fromfile(os.path.join(data_dir, "test.bin"), dtype=np.uint16)

    vocab_size = config.get("vocab_size", 50257)
    print(f"Data: train={len(train_data):,} tokens, val={len(val_data):,} tokens, "
          f"test={len(test_data):,} tokens, vocab={vocab_size}")

    # --- Model ---
    model = LSTMLM(
        vocab_size=vocab_size,
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        dropout=config.get("dropout", 0.0),
        tie_weights=config.get("tie_weights", True),
    )
    model.to(device)

    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # --- Optimizer ---
    optimizer = model.configure_optimizers(
        weight_decay=config.get("weight_decay", 0.0),
        learning_rate=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        device_type=device,
    )

    # --- Training params ---
    max_iters = config["max_iters"]
    eval_interval = config.get("eval_interval", 100)
    log_interval = config.get("log_interval", 10)
    batch_size = config.get("batch_size", 64)
    block_size = config.get("block_size", 256)
    grad_clip = config.get("grad_clip", 1.0)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    # LR schedule
    warmup_iters = config.get("warmup_iters", 100)
    lr_decay_iters = config.get("lr_decay_iters", max_iters)
    min_lr = config.get("min_lr", config["learning_rate"] / 10)
    eval_iters = config.get("eval_iters", 200)

    out_dir = config.get("out_dir", "out")
    os.makedirs(out_dir, exist_ok=True)

    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"Training: max_iters={max_iters}, batch_size={batch_size}, "
          f"block_size={block_size}")
    print(f"  tokens/iter: {batch_size * block_size * grad_accum_steps:,}")
    print(f"{'='*60}\n")

    t0 = time.time()
    iter_num = 0
    best_val_loss = float("inf")
    train_losses = []
    log_history = []  # Record {iter, train_loss, val_loss, lr, time}

    while iter_num < max_iters:
        # Learning rate schedule
        if iter_num < warmup_iters:
            lr = config["learning_rate"] * (iter_num + 1) / warmup_iters
        elif iter_num > lr_decay_iters:
            lr = min_lr
        else:
            decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
            lr = min_lr + 0.5 * (config["learning_rate"] - min_lr) * \
                 (1.0 + math.cos(math.pi * decay_ratio))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            with ctx:
                X, Y = get_batch(train_data, batch_size, block_size, device)
                _, loss, _ = model(X, Y)
                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(accum_loss)

        # Logging
        if iter_num % log_interval == 0:
            elapsed = time.time() - t0
            tokens_processed = (iter_num + 1) * batch_size * block_size * grad_accum_steps
            print(f"iter {iter_num:5d} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | dt {elapsed:.2f}s | "
                  f"tok/s {tokens_processed / max(elapsed, 1e-6):.0f}")
            log_history.append({
                "iter": iter_num, "train_loss": accum_loss,
                "val_loss": None, "lr": lr, "time": elapsed,
            })

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            val_stats = estimate_loss(
                model, val_data, eval_iters, batch_size, block_size, device
            )
            elapsed = time.time() - t0
            print(f"\n{'─'*50}")
            print(f"[Eval  @ iter {iter_num:5d}] "
                  f"val_loss={val_stats['loss']:.4f} | "
                  f"ppl={val_stats['ppl']:.2f} | "
                  f"bpc={val_stats['bpc']:.4f} | "
                  f"time={elapsed:.1f}s")
            print(f"{'─'*50}\n")

            # Record val loss in the closest log entry
            if log_history:
                log_history[-1]["val_loss"] = val_stats["loss"]

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                checkpoint_path = os.path.join(out_dir, "ckpt_best.pt")
                torch.save({
                    "iter": iter_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_stats["loss"],
                    "ppl": val_stats["ppl"],
                    "bpc": val_stats["bpc"],
                    "config": config,
                }, checkpoint_path)
                print(f"  → Best model saved to {checkpoint_path}\n")

        iter_num += 1

    # --- Final: evaluate on test set ---
    print(f"\n{'='*60}")
    print("Training complete! Running final evaluation...")
    print(f"{'='*60}")

    # Load best checkpoint for test evaluation
    best_ckpt = os.path.join(out_dir, "ckpt_best.pt")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint (iter={checkpoint['iter']}, "
              f"val_loss={checkpoint['val_loss']:.4f})")

    test_stats = evaluate_full(
        model, test_data, batch_size, block_size, device, max_batches=100
    )
    val_stats_full = evaluate_full(
        model, val_data, batch_size, block_size, device, max_batches=100
    )

    # --- Save results ---
    results = {
        "val": val_stats_full,
        "test": test_stats,
        "best_val": {
            "loss": best_val_loss,
            "ppl": math.exp(best_val_loss),
            "bpc": best_val_loss / math.log(2),
        },
        "config": config,
        "log_history": log_history,
    }

    results_path = os.path.join(out_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # --- Write text report ---
    _write_report(out_dir, config, best_val_loss, val_stats_full, test_stats)

    # --- Final checkpoint ---
    final_ckpt = os.path.join(out_dir, "ckpt.pt")
    torch.save({
        "iter": max_iters,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_stats_full["loss"],
        "ppl": val_stats_full["ppl"],
        "bpc": val_stats_full["bpc"],
        "config": config,
    }, final_ckpt)
    print(f"Checkpoint → {final_ckpt}")

    return results


# Monkey-patch configure_optimizers onto the model class
def _configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """Configure AdamW optimizer with weight decay."""
    # Separate parameters into those with weight decay and those without
    decay_params = []
    no_decay_params = []
    for name, param in self.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to bias, LayerNorm, or embedding
        if name.endswith(".bias") or "ln" in name or "token_embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # Use fused AdamW if available
    try:
        fused_available = "fused" in torch.backends.cuda.supported_variants()
    except (AttributeError, TypeError):
        fused_available = False
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, fused=use_fused,
    )
    return optimizer


LSTMLM.configure_optimizers = _configure_optimizers


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def get_default_config() -> dict:
    """Return the default LSTM configuration."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    return {
        # System
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "auto",  # resolved in train(): bfloat16 > float16 > float32
        "compile": False,  # Windows: torch.compile may not work

        # Data
        "data_dir": os.path.join(project_dir, "data", "wikitext2"),
        "vocab_size": 50257,
        "out_dir": os.path.join(project_dir, "out"),

        # Model
        "n_embd": 512,
        "n_layer": 3,
        "dropout": 0.3,
        "tie_weights": True,

        # Training
        "batch_size": 64,
        "block_size": 256,
        "max_iters": 5000,
        "gradient_accumulation_steps": 1,

        # Optimization
        "learning_rate": 3e-3,
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
        "grad_clip": 1.0,

        # LR schedule
        "warmup_iters": 500,
        "lr_decay_iters": 5000,
        "min_lr": 1e-4,

        # Evaluation
        "eval_interval": 200,
        "eval_iters": 200,
        "log_interval": 10,
    }


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from a Python file (same format as nanoGPT configs)."""
    config = {}
    with open(config_path, "r", encoding="utf-8") as f:
        exec(f.read(), {}, config)
    return config


def main():
    parser = argparse.ArgumentParser(description="LSTM Language Model Baseline")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config .py file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="Use torch.compile")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate a trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for eval_only mode")
    parser.add_argument("--generate", action="store_true",
                        help="Generate text after training")
    parser.add_argument("--prompt", type=str, default="The",
                        help="Prompt for text generation")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of text samples to generate")
    args = parser.parse_args()

    # Build config
    config = get_default_config()

    if args.config:
        file_config = load_config_from_file(args.config)
        config.update(file_config)

    # Resolve relative paths → absolute (based on project dir, CWD-independent)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key in ["data_dir", "out_dir"]:
        val = config.get(key, "")
        if val and not os.path.isabs(val):
            config[key] = os.path.normpath(os.path.join(project_dir, val))

    # CLI overrides (resolve if relative)
    if args.out_dir:
        config["out_dir"] = args.out_dir
        if not os.path.isabs(config["out_dir"]):
            config["out_dir"] = os.path.normpath(
                os.path.join(project_dir, config["out_dir"]))
    if args.device:
        config["device"] = args.device
    if args.compile is not None:
        config["compile"] = args.compile

    # --- Eval-only mode ---
    if args.eval_only:
        checkpoint_path = args.checkpoint or os.path.join(
            config["out_dir"], "ckpt_best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Error: checkpoint not found at {checkpoint_path}")
            sys.exit(1)

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config["device"],
                                weights_only=False)
        model_config = checkpoint.get("config", config)

        device = config["device"]
        train_data = np.fromfile(
            os.path.join(config["data_dir"], "train.bin"), dtype=np.uint16)
        val_data = np.fromfile(
            os.path.join(config["data_dir"], "val.bin"), dtype=np.uint16)
        test_data = np.fromfile(
            os.path.join(config["data_dir"], "test.bin"), dtype=np.uint16)

        model = LSTMLM(
            vocab_size=config["vocab_size"],
            n_embd=model_config.get("n_embd", config["n_embd"]),
            n_layer=model_config.get("n_layer", config["n_layer"]),
            dropout=model_config.get("dropout", 0.0),
            tie_weights=model_config.get("tie_weights", True),
        )
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        batch_size = config.get("batch_size", 64)
        block_size = config.get("block_size", 256)

        print("\nEvaluating on test set...")
        test_stats = evaluate_full(
            model, test_data, batch_size, block_size, device)
        print(f"  Test  → loss={test_stats['loss']:.4f}  "
              f"ppl={test_stats['ppl']:.2f}  bpc={test_stats['bpc']:.4f}")

        print("\nEvaluating on validation set...")
        val_stats = evaluate_full(
            model, val_data, batch_size, block_size, device)
        print(f"  Val   → loss={val_stats['loss']:.4f}  "
              f"ppl={val_stats['ppl']:.2f}  bpc={val_stats['bpc']:.4f}")
        return

    # --- Train ---
    results = train(config)

    # --- Generate samples ---
    if args.generate:
        print(f"\n{'='*60}")
        print("Text Generation Samples")
        print(f"{'='*60}")

        checkpoint_path = os.path.join(config["out_dir"], "ckpt_best.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config["device"],
                                    weights_only=False)
            model = LSTMLM(
                vocab_size=config["vocab_size"],
                n_embd=config["n_embd"],
                n_layer=config["n_layer"],
                dropout=0.0,  # No dropout for generation
                tie_weights=config.get("tie_weights", True),
            )
            model.to(config["device"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            import tiktoken
            enc = tiktoken.get_encoding("gpt2")

            for i in range(args.num_samples):
                prompt_ids = enc.encode(args.prompt)
                context = torch.tensor([prompt_ids], dtype=torch.long,
                                       device=config["device"])
                output = model.generate(context, max_new_tokens=100, temperature=0.8)
                output_text = enc.decode(output[0].tolist())
                print(f"\n--- Sample {i+1} ---")
                print(output_text)
                print("─" * 50)
        else:
            print("No checkpoint found for generation.")


if __name__ == "__main__":
    main()
