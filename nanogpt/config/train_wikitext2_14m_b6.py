# Best Task B hyperparameter configuration for 14M nanoGPT.
# B6: lower LR, stronger dropout, shorter context.

exec(open("config/train_wikitext2_14m.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-b6"
learning_rate = 3e-4
min_lr = 3e-5
dropout = 0.2
block_size = 256
max_iters = 6000
lr_decay_iters = 6000
seed = 1337
