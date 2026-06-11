"""Plot nanoGPT training curves from an experiment.log file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ITER_RE = re.compile(r"^iter (?P<iter>\d+): loss (?P<loss>[0-9.]+),")
STEP_RE = re.compile(
    r"^step (?P<step>\d+): train loss (?P<train>[0-9.]+), "
    r"val loss (?P<val>[0-9.]+), val ppl (?P<ppl>[0-9.]+)"
)


def moving_average(values: list[float], window: int) -> list[float]:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def parse_log(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float, float]]]:
    iter_losses = []
    eval_losses = []

    for line in path.read_text(encoding="utf-8").splitlines():
        iter_match = ITER_RE.match(line)
        if iter_match:
            iter_losses.append(
                (int(iter_match.group("iter")), float(iter_match.group("loss")))
            )
            continue

        step_match = STEP_RE.match(line)
        if step_match:
            eval_losses.append(
                (
                    int(step_match.group("step")),
                    float(step_match.group("train")),
                    float(step_match.group("val")),
                )
            )

    return iter_losses, eval_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT
        / "nanogpt"
        / "out-wikitext2-14m-modern-b6-3k"
        / "experiment.log",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "figures" / "nanogpt_modern_b6_3k_loss_curve.png",
    )
    parser.add_argument("--smooth_window", type=int, default=9)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iter_losses, eval_losses = parse_log(args.log)
    if not iter_losses or not eval_losses:
        raise SystemExit(f"no usable loss records found in {args.log}")

    iter_x = [row[0] for row in iter_losses]
    iter_y = [row[1] for row in iter_losses]
    smooth_y = moving_average(iter_y, args.smooth_window)
    smooth_x = iter_x[args.smooth_window - 1 :] if len(iter_y) >= args.smooth_window else iter_x

    eval_x = [row[0] for row in eval_losses]
    eval_train = [row[1] for row in eval_losses]
    eval_val = [row[2] for row in eval_losses]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(iter_x, iter_y, color="#93c5fd", alpha=0.28, linewidth=0.8, label="Iter train loss")
    ax.plot(smooth_x, smooth_y, color="#2563eb", linewidth=2.0, label="Iter train loss (smoothed)")
    ax.plot(eval_x, eval_train, "o-", color="#16a34a", linewidth=1.7, markersize=4, label="Eval train loss")
    ax.plot(eval_x, eval_val, "o-", color="#dc2626", linewidth=1.7, markersize=4, label="Eval val loss")
    ax.set_title("nanoGPT Modern+B6 Training Curve (3000 iters)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved nanoGPT training curve -> {args.output}")


if __name__ == "__main__":
    main()

