"""
准备字符级 Tiny Shakespeare 数据集（nanoGPT 复现）。

下载 input.txt（~1MB），构建字符级词表，按 90/10 切分为 train/val，
保存为 uint16 的 train.bin / val.bin，并把词表元信息写入 meta.pkl。
"""

import os
import pickle
import requests
import numpy as np

here = os.path.dirname(__file__)

input_file_path = os.path.join(here, "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"downloading {data_url} ...")
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 构建字符级词表
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


# 90/10 切分
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

np.array(train_ids, dtype=np.uint16).tofile(os.path.join(here, "train.bin"))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(here, "val.bin"))

meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(here, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("done. wrote train.bin, val.bin, meta.pkl to", here)
