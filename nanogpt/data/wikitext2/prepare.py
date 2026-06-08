"""
准备 WikiText-2 数据集（nanoGPT 复现，主线数据）。

- 用 HuggingFace 官方划分加载 wikitext-2-raw-v1（train / validation / test），不重切；
- 用 GPT-2 BPE（tiktoken）分词，便于与 nanoGPT / GPT-2 直接比较；
- 三个划分分别保存为 uint16 的 train.bin / val.bin / test.bin。

注意：train.py 只用 train.bin / val.bin；test.bin 留给后续统一 PPL/BPC 评测。
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

here = os.path.dirname(__file__)

enc = tiktoken.get_encoding("gpt2")

print("loading wikitext-2-raw-v1 ...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

# HF 的 validation 在本仓库里命名为 val.bin
split_to_filename = {"train": "train.bin", "validation": "val.bin", "test": "test.bin"}


def encode_split(split):
    # 把该划分所有行拼接成一段文本再编码（保留原始换行）
    text = "".join(ds[split]["text"])
    ids = enc.encode_ordinary(text)  # 不插入特殊 token
    return np.array(ids, dtype=np.uint16)


for split, filename in split_to_filename.items():
    arr = encode_split(split)
    out_path = os.path.join(here, filename)
    arr.tofile(out_path)
    print(f"{split}: {len(arr):,} tokens -> {out_path}")

print("done. GPT-2 BPE vocab_size = 50257 (训练时模型自动取 50304)")
