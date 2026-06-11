"""Create final comparison figures including tuned and architecture variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


RESULTS = [
    ("3-gram", ROOT / "eval" / "results_ngram_unified.json"),
    ("LSTM", ROOT / "eval" / "results_lstm_unified.json"),
    ("nanoGPT default", ROOT / "eval" / "results_nanogpt14m_unified.json"),
    ("nanoGPT B6", ROOT / "eval" / "results_nanogpt14m_b6.json"),
    ("nanoGPT Modern", ROOT / "eval" / "results_nanogpt14m_modern.json"),
    ("Modern+B6", ROOT / "eval" / "results_nanogpt14m_modern_b6.json"),
]


def load_results() -> list[tuple[str, dict]]:
    rows = []
    for name, path in RESULTS:
        rows.append((name, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = ROOT / "eval" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results()

    names = [name for name, _ in rows]
    x = np.arange(len(names))
    width = 0.36

    val_ppl = [r["val_ppl"] for _, r in rows]
    test_ppl = [r["test_ppl"] for _, r in rows]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - width / 2, val_ppl, width, label="Validation", color="#2563eb")
    ax.bar(x + width / 2, test_ppl, width, label="Test", color="#dc2626")
    ax.set_yscale("log")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Final WikiText-2 Perplexity Comparison")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    for xpos, value in zip(x - width / 2, val_ppl):
        ax.text(xpos, value * 1.06, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, test_ppl):
        ax.text(xpos, value * 1.06, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "final_ppl_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    val_loss = [r["val_loss"] for _, r in rows]
    test_loss = [r["test_loss"] for _, r in rows]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - width / 2, val_loss, width, label="Validation", color="#2563eb")
    ax.bar(x + width / 2, test_loss, width, label="Test", color="#dc2626")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Final WikiText-2 Loss Comparison")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    for xpos, value in zip(x - width / 2, val_loss):
        ax.text(xpos, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, test_loss):
        ax.text(xpos, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved final figures -> {output_dir}")


if __name__ == "__main__":
    main()
