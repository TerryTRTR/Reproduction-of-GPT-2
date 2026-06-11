# 14M nanoGPT + SwiGLU 消融。
# swiglu_hidden_mult=8/3 使 MLP 参数量与原 4x GeLU 近似对齐。

exec(open("config/train_wikitext2_14m.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-swiglu"
use_swiglu = True
swiglu_hidden_mult = 8 / 3
