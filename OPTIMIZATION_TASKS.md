# 后续优化计划与任务分配

> 日期：2026-06-09  
> 目标：在保持 baseline 对比公平、可复现的前提下，提高 nanoGPT 结果，并设计后续调优/消融实验。

---

## 1. 最高优先级

在加入新方法前，必须先把三方对比协议统一。

| 优先级 | 任务 | 负责人 | 产出 |
|---:|---|---|---|
| P0 | 统一 N-gram、LSTM、nanoGPT 的 WikiText-2 预处理 | 全员 | 一份共享的 `data/wikitext2/prepare.py` 输出 |
| P0 | 增加统一评测脚本，计算 val/test loss、PPL、BPC | 全员 | `eval/eval_lm.py` 或等价脚本 |
| P0 | 统一数据后重新跑最终指标 | A/B/C | token 数完全一致的最终结果表 |

当前三个分支生成的 token 数略有差异，因此最终报告前必须先解决这个问题。

---

## 2. Baseline 负责人

| 成员 | 模块 | 主要职责 |
|---|---|---|
| A | N-gram | 维护 3-gram baseline，有余力可加 Kneser-Ney 或更强插值 |
| B | LSTM | 维护 14M 同参数量级 LSTM，补充 checkpoint 导出与生成脚本 |
| C | nanoGPT | 维护 14M/44M GPT 配置，负责调参、架构消融与 LoRA 尝试 |

---

## 3. nanoGPT 优化路线

| 阶段 | 实验 | 目的 |
|---|---|---|
| 1 | 学习率搜索：`6e-4`、`3e-4`、`2e-4` | WikiText-2 较小，当前 LR 可能导致过拟合或震荡 |
| 2 | Dropout 搜索：`0.1`、`0.2`、`0.3` | 增强正则，改善 validation loss |
| 3 | 上下文长度：`block_size=256` vs `512` | 短上下文降低计算量，也可能起到正则效果 |
| 4 | RoPE 消融 | 比较学习式绝对位置编码与 rotary position embedding |
| 5 | RMSNorm 消融 | 尝试现代 Transformer 常用归一化 |
| 6 | SwiGLU 消融 | 尝试更强 MLP block，注意尽量保持参数量可比 |
| 7 | LoRA fine-tuning | 尝试低秩适配，提高调优效率 |

---

## 4. LoRA 计划

LoRA 值得尝试，但实验定位要写清楚。由于当前 nanoGPT 是在 WikiText-2 上从零训练，LoRA 可以有两种合理设置：

1. 先训练一个 WikiText-2 base nanoGPT checkpoint，然后冻结大部分权重，只训练 LoRA adapter。
2. 加载 pretrained GPT-2 checkpoint，加入 LoRA adapter，然后在 WikiText-2 上 fine-tune。

第二种方式更可能得到显著更低的 loss，但它使用了外部预训练，因此不能和 from-scratch baseline 混在一起比较。报告中应单独命名为 `GPT-2 pretrained + LoRA`。

### 4.1 建议的 LoRA 注入位置

针对当前 nanoGPT 代码，第一批 LoRA target 建议是：

| 模块 | 路径模式 | 原因 |
|---|---|---|
| Attention QKV projection | `Block.attn.c_attn` | 影响注意力的 Q/K/V 表示，是高收益适配点 |
| Attention output projection | `Block.attn.c_proj` | 调整注意力输出混合 |
| MLP input projection | `Block.mlp.c_fc` | 调整前馈层特征 |
| MLP output projection | `Block.mlp.c_proj` | 调整前馈层输出 |

建议先做 attention-only LoRA；如果稳定，再加入 MLP LoRA。

### 4.2 初始 LoRA 超参数

| 超参数 | 初始取值 |
|---|---:|
| rank `r` | 4 或 8 |
| alpha | 16 |
| dropout | 0.05 |
| trainable params | 必须单独报告 |
| learning rate | adapter 用 `1e-3` 起步，再 sweep |
| max_iters | 1000-3000 |

### 4.3 LoRA 实现任务

| 任务 | 负责人 | 细节 |
|---|---|---|
| 实现 `LoRALinear` | C | 在冻结的 base linear 外加低秩 `A @ B` 更新 |
| 添加 config 开关 | C | `use_lora`、`lora_rank`、`lora_alpha`、`lora_dropout`、`lora_targets` |
| 冻结 base 权重 | C | 只有 LoRA 参数设置 `requires_grad=True` |
| 增加参数量报告 | C | 同时打印 total params 和 trainable params |
| 从 scratch-base checkpoint 做 LoRA | C | 和 14M nanoGPT full fine-tuning 对比 |
| 可选：pretrained GPT-2 + LoRA | C | 单独报告，因为使用了外部预训练 |

---

## 5. LSTM 后续任务

| 任务 | 负责人 | 产出 |
|---|---|---|
| 增加确定性生成脚本 | B | 与 nanoGPT 使用同一 prompt 的输出 |
| checkpoint 不进 Git，或单独上传 | B | 避免大文件污染仓库 |
| 跑 3 个随机种子 | B | 报告 mean ± std |
| 尝试更强 LSTM | B | 可选 20M-30M 模型，用于 scaling 对比 |

---

## 6. N-gram 后续任务

| 任务 | 负责人 | 产出 |
|---|---|---|
| 增加确定性生成 helper | A | 与神经模型使用同一 prompt 的输出 |
| 尝试 Kneser-Ney smoothing | A | 更强 N-gram baseline |
| 报告 table size | A | unigram/bigram/trigram entry 数量 |
| 改成共享数据路径 | A | 使用统一 `.bin` 文件 |

---

## 7. 最终报告建议图表

| 图表 | 内容 |
|---|---|
| 主结果表 | N-gram、LSTM、nanoGPT 14M、nanoGPT 44M |
| 参数匹配表 | LSTM 13.92M vs nanoGPT 14.28M |
| 定性样例表 | 三个模型同 prompt 输出 |
| Loss 曲线 | train/val loss 随 iter 变化 |
| 可选消融表 | LR/dropout/RoPE/RMSNorm/SwiGLU/LoRA |

---

## 8. 推荐下一步命令

统一数据后，重新跑 nanoGPT 14M：

```bash
cd nanogpt
python train.py config/train_wikitext2_14m.py --compile=False
```

统一数据后，重新跑 LSTM：

```bash
cd LSTM_baseline
python src/lstm.py --config=src/config_lstm.py
```

生成 nanoGPT 定性样例：

```bash
cd nanogpt
python generate_fixed.py --out_dir=out-wikitext2-14m --max_new_tokens=60 --prompts_file=prompts_wikitext.txt
```

LoRA 建议放在统一数据和统一评测之后做，否则很难判断提升来自模型方法还是数据处理差异。
