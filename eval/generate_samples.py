"""Generate fixed-prompt qualitative samples for all baselines.

Outputs:
    eval/fixed_samples_unified.json
    eval/fixed_samples_unified.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LSTM_baseline"))
sys.path.insert(0, str(ROOT / "nanogpt"))
sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from ngram.baselines.ngram import (  # noqa: E402
    build_counts,
    interpolated_prob,
    load_tokens,
    load_vocab_size,
    parse_lambdas,
)
from src.lstm import LSTMLM  # noqa: E402


DEFAULT_PROMPTS = [
    "The meaning of life is",
    "In the beginning",
    "The history of the United States",
]


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_ngram(
    prompt: str,
    max_new_tokens: int,
    tables: list[Any],
    n: int,
    alpha: float,
    lambdas: list[float],
    vocab_size: int,
    enc: Any,
    unigram_fallback: list[int],
    next_token_cache: dict[tuple[int, ...], int],
) -> str:
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})

    for _ in range(max_new_tokens):
        context = tuple(ids[-(n - 1) :]) if n > 1 else ()
        if context in next_token_cache:
            ids.append(next_token_cache[context])
            continue

        candidates: list[int] = []
        for order in range(n, 1, -1):
            order_context = context[-(order - 1) :] if order > 1 else ()
            counts = tables[order - 1].get(order_context)
            if counts:
                candidates = list(counts.keys())
                break
        if not candidates:
            candidates = unigram_fallback
        next_id = max(
            candidates,
            key=lambda token: (
                interpolated_prob(token, context, tables, vocab_size, alpha, lambdas),
                -token,
            ),
        )
        next_token_cache[context] = next_id
        ids.append(next_id)
    return enc.decode(ids)


@torch.no_grad()
def generate_lstm(
    prompt: str,
    max_new_tokens: int,
    model: LSTMLM,
    enc: Any,
) -> str:
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    if not ids:
        ids = enc.encode("\n")
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    logits, _, hidden = model(x)
    for _ in range(max_new_tokens):
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        ids.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, _, hidden = model(x, hidden=hidden)
    return enc.decode(ids)


@torch.no_grad()
def generate_nanogpt(
    prompt: str,
    max_new_tokens: int,
    model: GPT,
    enc: Any,
) -> str:
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    if not ids:
        ids = enc.encode("\n")
    device = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
    return enc.decode(y[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data" / "wikitext2")
    parser.add_argument("--lstm_checkpoint", type=Path, default=ROOT / "LSTM_baseline" / "out" / "ckpt_best.pt")
    parser.add_argument("--nanogpt_checkpoint", type=Path, default=ROOT / "nanogpt" / "out-wikitext2-14m" / "ckpt.pt")
    parser.add_argument("--prompts_file", type=Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=Path, default=ROOT / "eval" / "fixed_samples_unified.json")
    parser.add_argument("--output_txt", type=Path, default=ROOT / "eval" / "fixed_samples_unified.txt")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambdas", default=None)
    args = parser.parse_args()

    torch.manual_seed(1337)
    enc = tiktoken.get_encoding("gpt2")
    prompts = load_prompts(args.prompts_file)
    train_tokens = load_tokens(args.data_dir / "train.bin")
    vocab_size = load_vocab_size(args.data_dir, train_tokens)
    lambdas = parse_lambdas(args.lambdas, args.n)
    tables = build_counts(train_tokens, args.n)
    unigram_counts = tables[0].get((), {})
    unigram_fallback = [
        token for token, _ in sorted(unigram_counts.items(), key=lambda item: item[1], reverse=True)[:200]
    ]
    next_token_cache: dict[tuple[int, ...], int] = {}

    lstm_checkpoint = torch.load(args.lstm_checkpoint, map_location=args.device, weights_only=False)
    lstm_cfg = lstm_checkpoint.get("config", {})
    lstm_model = LSTMLM(
        vocab_size=lstm_cfg.get("vocab_size", 50257),
        n_embd=lstm_cfg.get("n_embd", 512),
        n_layer=lstm_cfg.get("n_layer", 3),
        dropout=0.0,
        tie_weights=lstm_cfg.get("tie_weights", True),
    ).to(args.device)
    lstm_model.load_state_dict(lstm_checkpoint["model_state_dict"])
    lstm_model.eval()

    nanogpt_checkpoint = torch.load(args.nanogpt_checkpoint, map_location=args.device, weights_only=False)
    nanogpt_model = GPT(GPTConfig(**nanogpt_checkpoint["model_args"]))
    state_dict = nanogpt_checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    nanogpt_model.load_state_dict(state_dict)
    nanogpt_model.to(args.device)
    nanogpt_model.eval()

    samples = []
    for prompt in prompts:
        samples.append(
            {
                "prompt": prompt,
                "ngram": generate_ngram(
                    prompt,
                    args.max_new_tokens,
                    tables,
                    args.n,
                    args.alpha,
                    lambdas,
                    vocab_size,
                    enc,
                    unigram_fallback,
                    next_token_cache,
                ),
                "lstm": generate_lstm(prompt, args.max_new_tokens, lstm_model, enc),
                "nanogpt_14m": generate_nanogpt(prompt, args.max_new_tokens, nanogpt_model, enc),
            }
        )

    result = {
        "decode": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "data_dir": str(args.data_dir.resolve()),
        "samples": samples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Unified fixed-prompt samples",
        f"# decode=greedy max_new_tokens={args.max_new_tokens}",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"## Prompt: {sample['prompt']}",
                "",
                "### 3-gram",
                sample["ngram"],
                "",
                "### LSTM",
                sample["lstm"],
                "",
                "### nanoGPT 14M",
                sample["nanogpt_14m"],
                "",
            ]
        )
    args.output_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved -> {args.output_json}")
    print(f"saved -> {args.output_txt}")


if __name__ == "__main__":
    main()
