# Combined final candidate: Task C Modern architecture + Task B B6 hyperparameters.
# Modern = RoPE + RMSNorm + SwiGLU.

exec(open("config/train_wikitext2_14m_b6.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-modern-b6"
use_rope = True
use_rmsnorm = True
use_swiglu = True
swiglu_hidden_mult = 8 / 3
