"""Create Task B/C ablation figures for the final report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "eval" / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_task_b() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase1 = load_json(ROOT / "nanogpt" / "phase1_results.json")
    order = ["B1", "B2", "B3", "B4", "B5", "B6"]
    labels = [
        "B1\nlr6e-4\ndrop0.1\nbs512",
        "B2\nlr3e-4\ndrop0.1\nbs512",
        "B3\nlr2e-4\ndrop0.1\nbs512",
        "B4\nlr3e-4\ndrop0.2\nbs512",
        "B5\nlr3e-4\ndrop0.3\nbs512",
        "B6\nlr3e-4\ndrop0.2\nbs256",
    ]
    losses = [float(phase1[key]["best_val_loss"]) for key in order]
    ppls = [math.exp(loss) for loss in losses]
    x = np.arange(len(labels))
    colors = ["#94a3b8"] * len(labels)
    colors[-1] = "#16a34a"

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(x, ppls, color=colors)
    ax.set_ylabel("Best validation perplexity")
    ax.set_title("Task B Hyperparameter Sweep")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, value in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_b_hparam_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_c() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("Default", ROOT / "eval" / "results_nanogpt14m_default_arch.json"),
        ("RMSNorm", ROOT / "eval" / "results_nanogpt14m_rmsnorm.json"),
        ("SwiGLU", ROOT / "eval" / "results_nanogpt14m_swiglu.json"),
        ("LoRA r8", ROOT / "eval" / "results_nanogpt14m_lora_attn_r8.json"),
        ("RoPE", ROOT / "eval" / "results_nanogpt14m_rope.json"),
        ("Modern", ROOT / "eval" / "results_nanogpt14m_modern.json"),
        ("Modern+B6", ROOT / "eval" / "results_nanogpt14m_modern_b6.json"),
    ]
    names = [name for name, _ in rows]
    results = [load_json(path) for _, path in rows]
    val_ppl = [r["val_ppl"] for r in results]
    test_ppl = [r["test_ppl"] for r in results]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.bar(x - width / 2, val_ppl, width, label="Validation", color="#2563eb")
    ax.bar(x + width / 2, test_ppl, width, label="Test", color="#dc2626")
    ax.set_ylabel("Perplexity")
    ax.set_title("Task C Architecture and LoRA Ablation")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    for xpos, value in zip(x - width / 2, val_ppl):
        ax.text(xpos, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, test_ppl):
        ax.text(xpos, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "task_c_arch_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_task_b()
    plot_task_c()
    print(f"saved ablation figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
