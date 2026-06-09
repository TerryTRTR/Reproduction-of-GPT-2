"""
从训练好的 checkpoint 采样生成文本（nanoGPT 复现，代码自写）。

自动选择解码器：
  - 若数据集目录下存在 meta.pkl（字符级），用其中的 stoi/itos；
  - 否则回退到 tiktoken 的 GPT-2 BPE。

用法：
  python sample.py --out_dir=out-shakespeare-char
  python sample.py --out_dir=out-wikitext2 --start="The meaning of life is" --num_samples=3
"""

import os
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = "resume"        # 'resume'（从 out_dir 加载）或 'gpt2*'（HF 预训练）
out_dir = "out"
start = "\n"                # 起始 prompt；也可用 "FILE:prompt.txt" 从文件读取
num_samples = 5
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
exec(open("configurator.py", encoding="utf-8").read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# 加载模型
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError(f"未知 init_from: {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 选择编解码器
load_meta = False
if init_from == "resume" and "config" in checkpoint and "dataset" in checkpoint["config"]:
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 BPE encodings (tiktoken)...")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 准备起始 prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# 生成
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------")
