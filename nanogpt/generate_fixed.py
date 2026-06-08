"""
固定 prompt 的确定性生成（nanoGPT 复现，用于三方模型定性对比）。

动机：报告里"生成质量（定性）"一项要求"同一 prompt 的样本"。为了让 N-gram /
LSTM / nanoGPT 三方公平对比，这里用**同一组固定 prompt** + **确定性（贪心）解码**
+ **固定随机种子**生成可复现的续写，并把结果同时写入：
  - <out_dir>/fixed_samples.txt   （人读，直接贴进报告）
  - <out_dir>/fixed_samples.json  （结构化，便于脚本汇总三方对比）

解码方式：
  - 默认 greedy=True（等价 top_k=1），输出完全确定、可复现；
  - 也可设 --greedy=False 配合 --temperature / --top_k 做随机采样对比。

用法：
  python generate_fixed.py --out_dir=out-wikitext2
  python generate_fixed.py --out_dir=out-shakespeare-char --prompts_file=prompts_shakespeare.txt
  python generate_fixed.py --out_dir=out-wikitext2 --greedy=False --temperature=0.8 --top_k=200
"""

import os
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = "out"
prompts_file = ""           # 每行一个 prompt 的文本文件；留空则用下面的内置默认 prompt
max_new_tokens = 100
greedy = True               # True=贪心确定性解码（推荐用于跨模型对比）
temperature = 0.8           # 仅 greedy=False 时生效
top_k = 200                 # 仅 greedy=False 时生效
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

# 内置默认 prompt（BPE / 英文数据集通用，可被 --prompts_file 覆盖）
DEFAULT_PROMPTS = [
    "The meaning of life is",
    "In the beginning",
    "The history of the United States",
    "Scientists have recently discovered that",
    "Once upon a time",
]

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

# 加载模型（仅支持从 out_dir 的 checkpoint 续训结果）
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
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 选择编解码器（与 sample.py 一致）
load_meta = False
if "config" in checkpoint and "dataset" in checkpoint["config"]:
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 BPE encodings (tiktoken)...")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# 读取 prompt 列表
if prompts_file:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip() != ""]
else:
    prompts = DEFAULT_PROMPTS

# 贪心解码用 top_k=1、temperature 极小，确保确定性
gen_temperature = 1.0 if greedy else temperature
gen_top_k = 1 if greedy else top_k

results = []
for prompt in prompts:
    start_ids = encode(prompt)
    if len(start_ids) == 0:
        start_ids = encode("\n")
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=gen_temperature, top_k=gen_top_k)
    full = decode(y[0].tolist())
    completion = full[len(prompt):] if full.startswith(prompt) else full
    results.append({"prompt": prompt, "completion": completion, "full": full})

# 写文件
os.makedirs(out_dir, exist_ok=True)
decode_mode = "greedy" if greedy else f"sample(temp={temperature},top_k={top_k})"
header = (
    f"# nanoGPT fixed-prompt samples\n"
    f"# out_dir={out_dir}  decode={decode_mode}  "
    f"max_new_tokens={max_new_tokens}  seed={seed}\n"
)

txt_path = os.path.join(out_dir, "fixed_samples.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(header)
    for r in results:
        f.write("\n=== PROMPT ===\n")
        f.write(r["prompt"] + "\n")
        f.write("=== OUTPUT ===\n")
        f.write(r["full"] + "\n")

json_path = os.path.join(out_dir, "fixed_samples.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "out_dir": out_dir,
            "decode": decode_mode,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "samples": results,
        },
        f, ensure_ascii=False, indent=2,
    )

# 控制台打印
print(header)
for r in results:
    print("=== PROMPT ===")
    print(r["prompt"])
    print("=== OUTPUT ===")
    print(r["full"])
    print()
print(f"saved -> {txt_path}")
print(f"saved -> {json_path}")
