# WikiText-2 + GPT-2 BPE，约 14M 参数的 nanoGPT 配置。
# 用于和小型 LSTM 做同参数量级 baseline 对比。

out_dir = "out-wikitext2-14m"
eval_interval = 100
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

dataset = "wikitext2"
data_dir = "../data/wikitext2"
gradient_accumulation_steps = 8
batch_size = 12
block_size = 512

# 约 14.28M 参数；224 / 4 = 56 维每头。
n_layer = 5
n_head = 4
n_embd = 224
dropout = 0.1
bias = False

learning_rate = 6e-4
max_iters = 6000
lr_decay_iters = 6000
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
