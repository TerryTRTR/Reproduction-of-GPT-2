# 14M nanoGPT attention-only LoRA。
# 先训练 baseline 得到 out-wikitext2-14m/ckpt.pt，再运行本配置。

exec(open("config/train_wikitext2_14m.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-lora-attn-r8"
use_lora = True
lora_rank = 8
lora_alpha = 16.0
lora_dropout = 0.05
lora_targets = "attn"
lora_freeze_base = True
lora_base_checkpoint = "out-wikitext2-14m/ckpt.pt"

# Adapter 通常可以用更大的学习率，训练步数也更短。
learning_rate = 1e-3
min_lr = 1e-4
max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 100
