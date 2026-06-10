"""
Unified evaluation script for language models on WikiText-2.

Computes PPL (Perplexity) and BPC (Bits Per Character) on val/test splits.
Defaults to the shared project-standard data in data/wikitext2.

Usage:
    cd LSTM_baseline
    python src/eval_lm.py --checkpoint=out/ckpt_best.pt
"""

import os
import sys
import math
import json
import argparse
import pickle

# Allow running as both `python src/eval_lm.py` and `python -m src.eval_lm`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.lstm import LSTMLM, evaluate_full


def evaluate_lstm(checkpoint_path: str, data_dir: str, device: str = None,
                  batch_size: int = 64, block_size: int = None,
                  max_batches: int = None) -> dict:
    """
    Evaluate a trained LSTM model on val and test sets.

    Args:
        checkpoint_path: path to model checkpoint .pt file
        data_dir: path to directory with train/val/test .bin files
        device: 'cuda' or 'cpu'
        batch_size: evaluation batch size
        block_size: context window size
        max_batches: max number of batches (None = all)

    Returns:
        dict with val_loss, val_ppl, val_bpc, test_loss, test_ppl, test_bpc
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})
    if block_size is None:
        block_size = ckpt_config.get("block_size", 256)

    # Build model from checkpoint config
    model = LSTMLM(
        vocab_size=ckpt_config.get("vocab_size", 50257),
        n_embd=ckpt_config.get("n_embd", 512),
        n_layer=ckpt_config.get("n_layer", 3),
        dropout=0.0,  # No dropout for evaluation
        tie_weights=ckpt_config.get("tie_weights", True),
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    # Load data
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)
    test_data = np.fromfile(os.path.join(data_dir, "test.bin"), dtype=np.uint16)
    print(f"Data: val={len(val_data):,} tokens, test={len(test_data):,} tokens")

    # Evaluate
    print("\nEvaluating validation set...")
    val_stats = evaluate_full(model, val_data, batch_size, block_size, device, max_batches)

    print("Evaluating test set...")
    test_stats = evaluate_full(model, test_data, batch_size, block_size, device, max_batches)

    results = {
        "model_type": "lstm",
        "params": n_params,
        "val_loss": val_stats["loss"],
        "val_ppl": val_stats["ppl"],
        "val_bpc": val_stats["bpc"],
        "test_loss": test_stats["loss"],
        "test_ppl": test_stats["ppl"],
        "test_bpc": test_stats["bpc"],
        "val_tokens": val_stats.get("tokens", 0),
        "test_tokens": test_stats.get("tokens", 0),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified LM evaluation script")
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm"],
                        help="Type of model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory (default: ../data/wikitext2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--block_size", type=int, default=None,
                        help="Context block size (default: checkpoint config)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max batches for evaluation (None=all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: print to stdout)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON only (no progress messages)")
    args = parser.parse_args()

    # Resolve data_dir
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = os.path.dirname(project_dir)
    data_dir = args.data_dir or os.path.join(repo_dir, "data", "wikitext2")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        print("Run 'python data/wikitext2/prepare.py' from the repo root first.", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    if args.model_type == "lstm":
        results = evaluate_lstm(
            checkpoint_path=args.checkpoint,
            data_dir=data_dir,
            device=args.device,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_batches=args.max_batches,
        )
    else:
        print(f"Unknown model_type: {args.model_type}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        # Machine-readable JSON output
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"  Model:     LSTM ({results['params']/1e6:.2f}M params)")
        print(f"  Val Loss:  {results['val_loss']:.4f}")
        print(f"  Val PPL:   {results['val_ppl']:.2f}")
        print(f"  Val BPC:   {results['val_bpc']:.4f}")
        print(f"  Test Loss: {results['test_loss']:.4f}")
        print(f"  Test PPL:  {results['test_ppl']:.2f}")
        print(f"  Test BPC:  {results['test_bpc']:.4f}")
        print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
