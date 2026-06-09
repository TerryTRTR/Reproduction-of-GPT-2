# 主线 "GPT-2 mini" 配置：WikiText-2 + GPT-2 BPE，单卡 8-12G 可稳定训练。
# 约 25-45M 参数。显存不够时：减小 batch_size、增大 gradient_accumulation_steps、减小 block_size。

out_dir = "out-wikitext2"
eval_interval = 250
eval_iters = 100
log_interval = 10

always_save_checkpoint = False

dataset = "wikitext2"
# 有效 batch = batch_size * gradient_accumulation_steps * block_size 个 token
gradient_accumulation_steps = 8
batch_size = 12
block_size = 512

# GPT-2 mini：6-8 层，n_embd=512
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# vocab_size 留空由 train.py 默认取 50304（GPT-2 BPE 50257 向上取整）

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 500
