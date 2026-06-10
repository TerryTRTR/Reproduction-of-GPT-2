"""Prepare the shared WikiText-2 dataset for all baselines.

This is the project-standard data entry point. It writes GPT-2 BPE token ids to:

    data/wikitext2/train.bin
    data/wikitext2/val.bin
    data/wikitext2/test.bin

The preprocessing intentionally matches the nanoGPT run used in the baseline
comparison: HuggingFace Salesforce/wikitext parquet files, raw v1 split, all
rows concatenated per split, no added special tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


HERE = Path(__file__).resolve().parent
ENC = tiktoken.get_encoding("gpt2")
BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
SPLIT_TO_FILENAME = {"train": "train.bin", "validation": "val.bin", "test": "test.bin"}


def encode_split(ds, split: str) -> np.ndarray:
    text = "".join(ds[split]["text"])
    ids = ENC.encode_ordinary(text)
    return np.array(ids, dtype=np.uint16)


def main() -> None:
    print("loading wikitext-2-raw-v1 ...")
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{BASE_URL}/train-00000-of-00001.parquet",
            "validation": f"{BASE_URL}/validation-00000-of-00001.parquet",
            "test": f"{BASE_URL}/test-00000-of-00001.parquet",
        },
    )

    stats: dict[str, int] = {}
    for split, filename in SPLIT_TO_FILENAME.items():
        arr = encode_split(ds, split)
        out_path = HERE / filename
        arr.tofile(out_path)
        out_name = "val" if split == "validation" else split
        stats[out_name] = int(len(arr))
        print(f"{split}: {len(arr):,} tokens -> {out_path}")

    meta = {
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "source": BASE_URL,
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": ENC.n_vocab,
        "dtype": "uint16",
        "preprocessing": "concat_rows_per_split_no_special_tokens",
        "splits": stats,
    }
    (HERE / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done. GPT-2 BPE vocab_size = 50257")


if __name__ == "__main__":
    main()
