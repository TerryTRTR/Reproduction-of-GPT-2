"""Compare default nanoGPT and Modern+B6 training curves from log files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_nanogpt_training_curve import parse_log


ROOT = Path(__file__).resolve().parents[1]


def add_curve(ax, log_path: Path, label: str, color: str) -> tuple[float, float]:
    _, eval_losses = parse_log(log_path)
    if not eval_losses:
        raise SystemExit(f"no eval loss records found in {log_path}")

    steps = [row[0] for row in eval_losses]
    train_losses = [row[1] for row in eval_losses]
    val_losses = [row[2] for row in eval_losses]
    ax.plot(steps, val_losses, "o-", color=color, linewidth=2.0, markersize=4, label=f"{label} val")
    ax.plot(steps, train_losses, "--", color=color, alpha=0.45, linewidth=1.5, label=f"{label} train")
    return train_losses[-1], val_losses[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default_log",
        type=Path,
        default=ROOT / "nanogpt" / "out-wikitext2-14m-default-3k" / "experiment.log",
    )
    parser.add_argument(
        "--modern_log",
        type=Path,
        default=ROOT / "nanogpt" / "out-wikitext2-14m-modern-b6-3k" / "experiment.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "figures" / "nanogpt_default_vs_modern_b6_3k_loss.png",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    default_final = add_curve(ax, args.default_log, "Default", "#64748b")
    modern_final = add_curve(ax, args.modern_log, "Modern+B6", "#2563eb")
    ax.set_title("nanoGPT Training Curve Comparison (3000 iters)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=2)
    ax.text(
        0.02,
        0.03,
        (
            f"Default final: train {default_final[0]:.4f}, val {default_final[1]:.4f}\n"
            f"Modern+B6 final: train {modern_final[0]:.4f}, val {modern_final[1]:.4f}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved nanoGPT comparison curve -> {args.output}")


if __name__ == "__main__":
    main()

