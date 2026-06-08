"""Token-level n-gram language-model baseline.

Example:
    python baselines/ngram.py --data_dir data/wikitext2 --n 3 --alpha 0.1

The model is trained and evaluated on GPT-2 BPE token ids produced by
data/wikitext2/prepare.py, so its loss/PPL are directly comparable with
the Transformer and LSTM runs that use the same token stream.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict


Counts = DefaultDict[tuple[int, ...], Counter[int]]


def load_tokens(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python data/wikitext2/prepare.py"
        )
    tokens = array("H")
    with path.open("rb") as f:
        tokens.fromfile(f, path.stat().st_size // tokens.itemsize)
    return tokens.tolist()


def load_vocab_size(data_dir: Path, train_tokens: list[int]) -> int:
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "vocab_size" in meta:
            return int(meta["vocab_size"])
    return int(train_tokens.max()) + 1


def build_counts(tokens: list[int], n: int) -> list[Counts]:
    """Build count tables for orders 1..n."""
    tables: list[Counts] = [defaultdict(Counter) for _ in range(n)]
    for i, token in enumerate(tokens):
        max_order = min(n, i + 1)
        for order in range(1, max_order + 1):
            context = tuple(tokens[i - order + 1 : i])
            tables[order - 1][context][token] += 1
    return tables


def conditional_prob(
    token: int,
    context: tuple[int, ...],
    table: Counts,
    vocab_size: int,
    alpha: float,
) -> float:
    counts = table.get(context)
    if not counts:
        return 1.0 / vocab_size
    total = sum(counts.values())
    return (counts.get(token, 0) + alpha) / (total + alpha * vocab_size)


def interpolated_prob(
    token: int,
    context: tuple[int, ...],
    tables: list[Counts],
    vocab_size: int,
    alpha: float,
    lambdas: list[float],
) -> float:
    n = len(tables)
    prob = 0.0
    for order in range(1, n + 1):
        order_context = context[-(order - 1) :] if order > 1 else ()
        prob += lambdas[order - 1] * conditional_prob(
            token, order_context, tables[order - 1], vocab_size, alpha
        )
    return prob


def parse_lambdas(raw: str | None, n: int) -> list[float]:
    if raw is None:
        # Bias toward the requested order while retaining lower-order fallback.
        weights = [1.0 / (2 ** (n - order)) for order in range(1, n + 1)]
    else:
        weights = [float(x) for x in raw.split(",")]
        if len(weights) != n:
            raise ValueError(f"--lambdas must contain exactly {n} values")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--lambdas must sum to a positive value")
    return [w / total for w in weights]


def evaluate(
    tokens: list[int],
    tables: list[Counts],
    vocab_size: int,
    alpha: float,
    lambdas: list[float],
) -> dict[str, float]:
    n = len(tables)
    context_len = n - 1
    nll = 0.0
    count = 0

    for i in range(context_len, len(tokens)):
        context = tuple(tokens[i - context_len : i]) if context_len else ()
        prob = interpolated_prob(tokens[i], context, tables, vocab_size, alpha, lambdas)
        nll -= math.log(prob)
        count += 1

    loss = nll / count
    return {
        "tokens": float(count),
        "loss": loss,
        "ppl": math.exp(loss),
        "bpc": loss / math.log(2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/wikitext2"))
    parser.add_argument("--n", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--lambdas",
        default=None,
        help="Comma-separated interpolation weights for orders 1..n.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    train = load_tokens(args.data_dir / "train.bin")
    if args.max_train_tokens is not None:
        train = train[: args.max_train_tokens]
    val = load_tokens(args.data_dir / "val.bin")
    test = load_tokens(args.data_dir / "test.bin")
    vocab_size = load_vocab_size(args.data_dir, train)
    lambdas = parse_lambdas(args.lambdas, args.n)

    tables = build_counts(train, args.n)
    train_seconds = time.perf_counter() - start

    val_metrics = evaluate(val, tables, vocab_size, args.alpha, lambdas)
    test_metrics = evaluate(test, tables, vocab_size, args.alpha, lambdas)

    result = {
        "model": f"{args.n}-gram",
        "alpha": args.alpha,
        "lambdas": lambdas,
        "vocab_size": vocab_size,
        "train_tokens": len(train),
        "train_seconds": train_seconds,
        "val": val_metrics,
        "test": test_metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
