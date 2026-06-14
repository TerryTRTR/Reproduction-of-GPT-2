"""Unified WikiText-2 language-model evaluation.

The default data directory is the shared WikiText-2 GPT-2-BPE split:
``data/wikitext2``. Point every baseline at this same directory when reporting
final comparable numbers.

Examples:
    python eval/eval_lm.py --model ngram --data_dir data/wikitext2
    python eval/eval_lm.py --model lstm --checkpoint LSTM_baseline/out/ckpt_best.pt
    python eval/eval_lm.py --model nanogpt --checkpoint nanogpt/out-wikitext2-14m/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LSTM_baseline"))
sys.path.insert(0, str(ROOT / "nanogpt"))
sys.path.insert(0, str(ROOT))

from ngram.baselines.ngram import (  # noqa: E402
    build_counts,
    evaluate as evaluate_ngram_split,
    load_tokens,
    load_vocab_size,
    parse_lambdas,
)
from src.lstm import LSTMLM, evaluate_full as evaluate_lstm_split  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def load_uint16(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return np.fromfile(path, dtype=np.uint16)


def metric_row(split: str, stats: dict[str, float]) -> dict[str, float]:
    """Convert split metrics to flat JSON fields used by all model types."""
    loss = float(stats["loss"])
    return {
        f"{split}_loss": loss,
        f"{split}_ppl": float(stats.get("ppl", math.exp(loss))),
        f"{split}_bpc": float(stats.get("bpc", loss / math.log(2))),
        f"{split}_tokens": int(stats.get("tokens", 0)),
    }


def evaluate_ngram(args: argparse.Namespace, data_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    train = load_tokens(data_dir / "train.bin")
    if args.max_train_tokens is not None:
        train = train[: args.max_train_tokens]
    val = load_tokens(data_dir / "val.bin")
    test = load_tokens(data_dir / "test.bin")
    vocab_size = load_vocab_size(data_dir, train)
    lambdas = parse_lambdas(args.lambdas, args.n)

    tables = build_counts(train, args.n)
    train_seconds = time.perf_counter() - start

    val_stats = evaluate_ngram_split(val, tables, vocab_size, args.alpha, lambdas)
    test_stats = evaluate_ngram_split(test, tables, vocab_size, args.alpha, lambdas)

    return {
        "model": "ngram",
        "name": f"{args.n}-gram",
        "params": None,
        "data_dir": str(data_dir),
        "train_tokens": len(train),
        "vocab_size": vocab_size,
        "alpha": args.alpha,
        "lambdas": lambdas,
        "train_seconds": train_seconds,
        **metric_row("val", val_stats),
        **metric_row("test", test_stats),
    }


def evaluate_lstm(args: argparse.Namespace, data_dir: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = LSTMLM(
        vocab_size=cfg.get("vocab_size", 50257),
        n_embd=cfg.get("n_embd", 512),
        n_layer=cfg.get("n_layer", 3),
        dropout=0.0,
        tie_weights=cfg.get("tie_weights", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val = load_uint16(data_dir / "val.bin")
    test = load_uint16(data_dir / "test.bin")
    block_size = args.block_size or cfg.get("block_size", 256)
    batch_size = args.batch_size or 64

    val_stats = evaluate_lstm_split(
        model, val, batch_size, block_size, device, args.max_batches
    )
    test_stats = evaluate_lstm_split(
        model, test, batch_size, block_size, device, args.max_batches
    )

    return {
        "model": "lstm",
        "name": "LSTM",
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "block_size": block_size,
        "batch_size": batch_size,
        **metric_row("val", val_stats),
        **metric_row("test", test_stats),
    }


@torch.no_grad()
def evaluate_nanogpt_split(
    model: GPT,
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_id = 0

    while True:
        if max_batches is not None and batch_id >= max_batches:
            break
        start = batch_id * batch_size * block_size
        remaining = len(data) - start - 1
        if remaining < block_size:
            break

        usable = min(remaining, batch_size * block_size)
        num_blocks = usable // block_size
        # Full-split evaluation walks through the token stream sequentially.
        # This avoids random-batch noise and makes final PPL comparable across runs.
        x_np = data[start : start + num_blocks * block_size].astype(np.int64)
        y_np = data[start + 1 : start + 1 + num_blocks * block_size].astype(np.int64)
        x = torch.from_numpy(x_np).view(num_blocks, block_size).to(device)
        y = torch.from_numpy(y_np).view(num_blocks, block_size).to(device)

        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
        batch_id += 1

    if total_tokens == 0:
        raise ValueError("No evaluation tokens. Lower --block_size or check the data split.")
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "bpc": loss / math.log(2),
        "tokens": total_tokens,
    }


def evaluate_nanogpt(args: argparse.Namespace, data_dir: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    if args.block_size is not None:
        model_args = dict(model_args)
        model_args["block_size"] = min(args.block_size, model_args["block_size"])

    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val = load_uint16(data_dir / "val.bin")
    test = load_uint16(data_dir / "test.bin")
    block_size = model.config.block_size
    batch_size = args.batch_size or 16

    val_stats = evaluate_nanogpt_split(
        model, val, batch_size, block_size, device, args.max_batches
    )
    test_stats = evaluate_nanogpt_split(
        model, test, batch_size, block_size, device, args.max_batches
    )

    return {
        "model": "nanogpt",
        "name": "nanoGPT",
        "params": model.get_num_params(),
        "checkpoint": str(args.checkpoint),
        "best_iter": int(checkpoint.get("iter_num", -1)),
        "best_val_loss_in_ckpt": float(checkpoint.get("best_val_loss", math.nan)),
        "data_dir": str(data_dir),
        "block_size": block_size,
        "batch_size": batch_size,
        **metric_row("val", val_stats),
        **metric_row("test", test_stats),
    }


def print_summary(result: dict[str, Any]) -> None:
    params = result["params"]
    params_text = "-" if params is None else f"{params / 1e6:.2f}M"
    print("\nEvaluation Results")
    print("=" * 64)
    print(f"Model:      {result['name']}")
    print(f"Params:     {params_text}")
    print(f"Data:       {result['data_dir']}")
    print(f"Val loss:   {result['val_loss']:.4f}")
    print(f"Val PPL:    {result['val_ppl']:.2f}")
    print(f"Val BPC:    {result['val_bpc']:.4f}")
    print(f"Test loss:  {result['test_loss']:.4f}")
    print(f"Test PPL:   {result['test_ppl']:.2f}")
    print(f"Test BPC:   {result['test_bpc']:.4f}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ngram", "lstm", "nanogpt"], required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data" / "wikitext2",
        help="Directory containing train.bin, val.bin, and test.bin.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--n", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambdas", default=None)
    parser.add_argument("--max_train_tokens", type=int, default=None)
    args = parser.parse_args()

    if args.model in {"lstm", "nanogpt"} and args.checkpoint is None:
        parser.error(f"--checkpoint is required for --model {args.model}")

    data_dir = args.data_dir.resolve()
    if args.model == "ngram":
        result = evaluate_ngram(args, data_dir)
    elif args.model == "lstm":
        result = evaluate_lstm(args, data_dir, args.device)
    else:
        result = evaluate_nanogpt(args, data_dir, args.device)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
        if args.output is not None:
            print(f"\nSaved JSON to {args.output}")


if __name__ == "__main__":
    main()
