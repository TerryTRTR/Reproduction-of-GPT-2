"""
GPT-2 训练脚本（nanoGPT 复现，代码自写）。

特性：
  - 单机单卡 / CPU 训练（可选 DDP 多卡，通过 torchrun 启动）
  - 梯度累积模拟大 batch
  - AdamW + warmup + cosine 学习率衰减
  - 自动混合精度（bf16 / fp16+GradScaler），bf16 支持自动探测
  - 定期在 train/val 上估计 loss 并保存最优 checkpoint
  - 支持从头训练 / 断点续训 / 从 GPT-2 预训练初始化

用法：
  python train.py config/train_shakespeare_char.py
  python train.py config/train_wikitext2.py --compile=False --device=cpu
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 默认配置（可被 config 文件或命令行 --key=value 覆盖）
# I/O
out_dir = "out"
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False           # 若为 True，第一次 eval 后立即退出
always_save_checkpoint = False  # 若为 True，每次 eval 都保存（否则只在 val loss 下降时保存）
init_from = "scratch"       # 'scratch' | 'resume' | 'gpt2*'
# 数据
dataset = "shakespeare_char"
data_dir = ""                 # 可选：覆盖默认 data/<dataset>，例如 ../data/wikitext2
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# 模型
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
use_rope = False
use_rmsnorm = False
use_swiglu = False
swiglu_hidden_mult = 8 / 3
use_lora = False
lora_rank = 8
lora_alpha = 16.0
lora_dropout = 0.05
lora_targets = "attn"
lora_freeze_base = True
lora_base_checkpoint = ""  # 可指向已有 full-training ckpt.pt，用其初始化 LoRA base
# AdamW 优化器
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# 学习率衰减
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP
backend = "nccl"            # Windows 下若多卡可改 'gloo'
# 系统
device = "cuda"             # 'cuda' | 'cpu' | 'cuda:0' | 'mps' 等
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True              # PyTorch 2.0 torch.compile（Windows 不支持时设 False）
# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py", encoding="utf-8").read())  # 用命令行 / config 文件覆盖上面的默认值
config = {k: globals()[k] for k in config_keys}  # 用于日志 / checkpoint
# -----------------------------------------------------------------------------

# DDP 设置：torchrun 会注入 RANK 等环境变量
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * block_size
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# 数据加载：默认从 data/<dataset> 读取；WikiText-2 可由 config 指向仓库共享目录。
data_dir = data_dir or os.path.join("data", dataset)


def get_batch(split):
    # 每次重新 memmap，避免内存泄漏（参见 nanoGPT issue）
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9

# 尝试从数据集 meta.pkl 读取 vocab_size（字符级数据集会有）
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# 初始化模型
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_rope=use_rope,
    use_rmsnorm=use_rmsnorm,
    use_swiglu=use_swiglu,
    swiglu_hidden_mult=swiglu_hidden_mult,
    use_lora=use_lora,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    lora_targets=lora_targets,
    lora_freeze_base=lora_freeze_base,
)
base_model_arg_keys = ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]


def load_base_weights_into_lora(model, state_dict):
    """把普通 Linear checkpoint 权重映射到 LoRALinear.linear，LoRA adapter 保持初始化。"""
    model_state = model.state_dict()
    remapped = {}
    for key, value in state_dict.items():
        key = key.removeprefix("_orig_mod.")
        if key in model_state:
            remapped[key] = value
            continue
        if key.endswith(".weight") or key.endswith(".bias"):
            prefix, suffix = key.rsplit(".", 1)
            lora_key = f"{prefix}.linear.{suffix}"
            if lora_key in model_state:
                remapped[lora_key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("attn.bias")]
    missing_non_lora = [
        k for k in missing
        if ".lora_" not in k and not k.endswith("rope_inv_freq")
    ]
    if missing_non_lora or unexpected:
        raise RuntimeError(
            f"LoRA base checkpoint 加载不完整，missing={missing_non_lora}, unexpected={unexpected}"
        )


if use_lora and lora_base_checkpoint:
    print(f"Initializing LoRA model from base checkpoint: {lora_base_checkpoint}")
    checkpoint = torch.load(lora_base_checkpoint, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in base_model_arg_keys:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    load_base_weights_into_lora(model, checkpoint["model"])
elif init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in model_args:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # 修复 torch.compile 可能加上的前缀
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    for k in base_model_arg_keys:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"未知 init_from: {init_from}")

# 若指定的 block_size 小于模型默认，则裁剪
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size
model.to(device)
if model.config.use_lora and model.config.lora_freeze_base:
    model.mark_only_lora_as_trainable()
print(
    "trainable parameters: %.2fM / %.2fM"
    % (model.get_num_trainable_params() / 1e6, model.get_num_params(non_embedding=False) / 1e6)
)

# GradScaler：仅 fp16 时启用
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # 释放内存

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model  # DDP 包装下取原始模型


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    # 1) warmup 阶段线性增长
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 超过衰减区间后用 min_lr
    if it > lr_decay_iters:
        return min_lr
    # 3) 中间用 cosine 衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# 训练主循环
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # 定期评估并保存
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        ppl = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, val ppl {ppl:.2f}"
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # 前向 / 反向 + 梯度累积
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # 只在累积的最后一步同步梯度
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # 缩放以匹配平均
        X, Y = get_batch("train")  # 异步预取下一批
        scaler.scale(loss).backward()
    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
            f"mfu {running_mfu * 100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
