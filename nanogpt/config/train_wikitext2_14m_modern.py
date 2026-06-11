# 14M nanoGPT 组合改动：RoPE + RMSNorm + SwiGLU。
# 建议在单项消融完成后再跑，用来验证组合收益是否可叠加。

exec(open("config/train_wikitext2_14m.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-modern"
use_rope = True
use_rmsnorm = True
use_swiglu = True
swiglu_hidden_mult = 8 / 3
