# LSTM Language Model Configuration for WikiText-2 (GPU)
# Usage: python src/lstm.py --config=src/config_lstm.py
# Target: <5 min on modern GPU (RTX 3060+)

# ---- System ----
device = "cuda"
dtype = "auto"          # auto = bfloat16 > float16 > float32 (safe for all GPUs)
compile = False         # torch.compile — Linux only; set True if supported

# ---- Output ----
out_dir = "out"
data_dir = "../data/wikitext2"  # shared project-standard WikiText-2 split

# ---- Data ----
vocab_size = 50257
batch_size = 64
block_size = 128
gradient_accumulation_steps = 1

# ---- Model (smaller for fast training ~10M params) ----
n_embd = 256
n_layer = 2
dropout = 0.2
tie_weights = True

# ---- Optimization ----
learning_rate = 3e-3
max_iters = 2000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# ---- LR Schedule ----
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 1e-4

# ---- Evaluation ----
eval_interval = 200
eval_iters = 200
log_interval = 10
