"""Prepare WikiText-2 with the GPT-2 BPE tokenizer.

Outputs train.bin, val.bin, test.bin, and meta.json in this directory.
The .bin files store uint16 GPT-2 token ids, matching nanoGPT's common
data format while keeping all baselines on the same tokenizer/splits.
"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
from typing import Protocol

from datasets import load_dataset


class Tokenizer(Protocol):
    n_vocab: int

    def encode(self, text: str) -> list[int]:
        ...


def get_tokenizer() -> Tokenizer:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        class TiktokenTokenizer:
            n_vocab = enc.n_vocab

            def encode(self, text: str) -> list[int]:
                return enc.encode_ordinary(text)

        print("tokenizer: tiktoken/gpt2")
        return TiktokenTokenizer()
    except Exception as exc:
        print(f"tiktoken GPT-2 tokenizer unavailable ({exc}); falling back to transformers")

    from transformers import GPT2TokenizerFast

    hf_tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    class TransformersTokenizer:
        n_vocab = hf_tok.vocab_size

        def encode(self, text: str) -> list[int]:
            return hf_tok.encode(text, add_special_tokens=False)

    print("tokenizer: transformers/openai-community-gpt2")
    return TransformersTokenizer()


def encode_split(texts: list[str], enc: Tokenizer) -> array:
    # Preserve document boundaries without adding a custom token.
    text = "\n\n".join(texts)
    tokens = enc.encode(text)
    return array("H", tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--dataset_name",
        default="Salesforce/wikitext",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--dataset_config",
        default="wikitext-2-raw-v1",
        help="HuggingFace dataset config.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset_name, args.dataset_config)
    enc = get_tokenizer()

    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    stats: dict[str, int] = {}

    for hf_split, out_name in split_map.items():
        texts = [row["text"] for row in ds[hf_split]]
        ids = encode_split(texts, enc)
        with (args.out_dir / f"{out_name}.bin").open("wb") as f:
            ids.tofile(f)
        stats[out_name] = len(ids)
        print(f"{out_name}: {len(ids):,} tokens")

    meta = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer": "gpt2",
        "vocab_size": enc.n_vocab,
        "dtype": "uint16",
        "splits": stats,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
