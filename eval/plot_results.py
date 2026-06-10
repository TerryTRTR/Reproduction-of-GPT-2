"""Create report figures from unified evaluation results."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_metric_bars(results: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] if r["model"] != "nanogpt" else "nanoGPT 14M" for r in results]
    val_ppl = [r["val_ppl"] for r in results]
    test_ppl = [r["test_ppl"] for r in results]
    val_loss = [r["val_loss"] for r in results]
    test_loss = [r["test_loss"] for r in results]
    x = np.arange(len(names))
    width = 0.36
    colors = ["#3b82f6", "#ef4444"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, val_ppl, width, label="Validation", color=colors[0])
    ax.bar(x + width / 2, test_ppl, width, label="Test", color=colors[1])
    ax.set_yscale("log")
    ax.set_ylim(top=max(val_ppl + test_ppl) * 1.5)
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Unified WikiText-2 Perplexity")
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    for xpos, value in zip(x - width / 2, val_ppl):
        ax.text(xpos, value * 1.06, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2, test_ppl):
        ax.text(xpos, value * 1.06, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "ppl_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, val_loss, width, label="Validation", color=colors[0])
    ax.bar(x + width / 2, test_loss, width, label="Test", color=colors[1])
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Unified WikiText-2 Loss")
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    for xpos, value in zip(x - width / 2, val_loss):
        ax.text(xpos, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2, test_loss):
        ax.text(xpos, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lstm_loss_curve(results_path: Path, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results_path.exists():
        return
    with results_path.open("rb") as f:
        results = pickle.load(f)
    history = results.get("log_history", [])
    if not history:
        return

    iters = [row["iter"] for row in history]
    train = [row["train_loss"] for row in history]
    eval_iters = [row["iter"] for row in history if row.get("val_loss") is not None]
    val = [row["val_loss"] for row in history if row.get("val_loss") is not None]
    window = max(1, len(train) // 40)
    if window > 1:
        smooth = np.convolve(train, np.ones(window) / window, mode="valid")
        smooth_iters = iters[window - 1 :]
    else:
        smooth = train
        smooth_iters = iters

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(iters, train, color="#60a5fa", alpha=0.22, linewidth=0.7, label="Train loss")
    ax.plot(smooth_iters, smooth, color="#2563eb", linewidth=2.0, label="Train loss (smoothed)")
    if eval_iters:
        ax.plot(eval_iters, val, "o-", color="#dc2626", linewidth=1.8, markersize=4, label="Validation loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LSTM Training Curve")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lstm_loss_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "eval" / "figures")
    parser.add_argument("--lstm_results", type=Path, default=ROOT / "LSTM_baseline" / "out" / "results.pkl")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        load_json(ROOT / "eval" / "results_ngram_unified.json"),
        load_json(ROOT / "eval" / "results_lstm_unified.json"),
        load_json(ROOT / "eval" / "results_nanogpt14m_unified.json"),
    ]
    plot_metric_bars(results, args.output_dir)
    plot_lstm_loss_curve(args.lstm_results, args.output_dir)
    print(f"saved figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
