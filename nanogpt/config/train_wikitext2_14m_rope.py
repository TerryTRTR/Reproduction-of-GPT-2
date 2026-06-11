# 14M nanoGPT + RoPE 消融。
# 只替换 learned absolute position embedding，其他训练设置继承 14M baseline。

exec(open("config/train_wikitext2_14m.py", encoding="utf-8").read())

out_dir = "out-wikitext2-14m-rope"
use_rope = True
