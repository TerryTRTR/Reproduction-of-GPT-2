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

## 2. 后续分工原则

Baseline 的主要作用是提供对照组。N-gram 和 LSTM 在完成最终结果表、统一评测和定性样例后，不再继续作为主要优化方向。

后续三个人应共同转向 nanoGPT 优化，把精力集中在同一个主方法上，这样报告会更有深度，也更容易形成清晰主线：

```text
统一 baseline 对比 -> nanoGPT 默认复现 -> nanoGPT 系统优化/消融
```

| 成员 | 后续重点 | 主要职责 |
|---|---|---|
| A | nanoGPT 数据与评测 | 统一预处理、统一 eval、结果表、loss 曲线、定性样例汇总 |
| B | nanoGPT 超参与训练实验 | 学习率/dropout/block_size sweep，多随机种子，整理训练日志 |
| C | nanoGPT 架构与 LoRA | RoPE/RMSNorm/SwiGLU/LoRA 实现与消融实验 |

N-gram 与 LSTM 后续只做必要维护：保证能复现指标、能生成同 prompt 样例、能接入统一评测脚本。

### 2.1 A 的详细任务：数据、评测、结果汇总

A 的核心目标是保证所有实验“可比”。如果数据和评测不统一，后面的优化结果就很难解释。

| 任务 | 具体内容 | 交付物 |
|---|---|---|
| 统一数据预处理 | 检查 `ngram/`、`LSTM_baseline/`、`nanogpt/` 三份 prepare 逻辑；确定最终只使用一份 WikiText-2 + GPT-2 BPE `.bin` 文件 | 统一后的 prepare 脚本；三方 token 数一致截图/日志 |
| 统一评测脚本 | 写或整合一个统一 `eval/eval_lm.py`，对 N-gram/LSTM/nanoGPT 都输出 val/test loss、PPL、BPC | `eval/eval_lm.py`；统一 JSON/表格输出 |
| 定性样例汇总 | 固定 prompt、固定 decoding 设置，收集三方输出 | `BASELINE_COMPARISON.md` 中的样例表 |
| 结果表维护 | 汇总 N-gram、LSTM、nanoGPT 默认版、nanoGPT 调优版指标 | 最终主结果表 |
| 画图 | 根据训练日志画 loss curve、PPL 对比图、消融柱状图 | 报告用 figures |
| 报告结果部分 | 负责 Results & Analysis 初稿 | 结果分析文字、图表说明 |

A 需要重点关注：

```text
1. 三方是否真的使用同一份 train/val/test。
2. PPL 是否都由 exp(loss) 得到。
3. BPC 是否都由 loss / ln(2) 得到。
4. 表格中的 best val 和 test 是不是来自同一个 checkpoint。
5. 所有实验命令是否能被别人复现。
```

建议 A 优先完成：

```bash
# 目标：最终能一条命令评测三方模型
python eval/eval_lm.py --model=ngram --...
python eval/eval_lm.py --model=lstm --...
python eval/eval_lm.py --model=nanogpt --...
```

### 2.2 B 的详细任务：nanoGPT 超参实验

B 的核心目标是找到一个比默认 14M nanoGPT 更稳、更低 validation loss 的训练配置。

| 任务 | 具体内容 | 交付物 |
|---|---|---|
| 学习率搜索 | 在 14M nanoGPT 上跑 `6e-4`、`3e-4`、`2e-4` | LR sweep 表格 |
| Dropout 搜索 | 在较优 LR 上跑 `dropout=0.1/0.2/0.3` | dropout sweep 表格 |
| block size 对比 | 比较 `block_size=256` 与 `512`，保持有效 batch token 尽量可比 | context length 对比 |
| 训练步数分析 | 记录 best val 出现在哪个 iter，判断过拟合时间点 | early stopping 建议 |
| 多随机种子 | 对最终配置跑至少 3 个 seed | mean ± std |
| 日志整理 | 保存每个实验的命令、best val loss、PPL、训练时间 | 实验记录表 |

B 建议优先跑 14M 参数匹配模型，因为它和 LSTM baseline 最公平：

```bash
cd nanogpt
python train.py config/train_wikitext2_14m.py --compile=False \
  --learning_rate=3e-4 --min_lr=3e-5 --dropout=0.2 \
  --out_dir=out-wikitext2-14m-lr3e4-drop02
```

推荐实验矩阵：

| 实验编号 | learning_rate | dropout | block_size | 说明 |
|---|---:|---:|---:|---|
| B1 | 6e-4 | 0.1 | 512 | 当前默认配置 |
| B2 | 3e-4 | 0.1 | 512 | 降低 LR |
| B3 | 2e-4 | 0.1 | 512 | 更保守 LR |
| B4 | 3e-4 | 0.2 | 512 | 增强正则 |
| B5 | 3e-4 | 0.3 | 512 | 更强正则 |
| B6 | 3e-4 | 0.2 | 256 | 短上下文对比 |

B 每次实验需要记录：

```text
out_dir
params
seed
learning_rate
dropout
block_size
max_iters
best_iter
best_val_loss
best_val_ppl
是否出现过拟合
训练耗时
```

### 2.3 C 的详细任务：架构消融与 LoRA

C 的核心目标是实现 nanoGPT 的方法改进，并用消融实验说明“为什么变好”。

| 任务 | 具体内容 | 交付物 |
|---|---|---|
| RoPE | 将学习式绝对位置编码替换或扩展为 rotary position embedding | RoPE 配置和对比结果 |
| RMSNorm | 将 LayerNorm 替换为 RMSNorm | RMSNorm 消融结果 |
| SwiGLU | 将 GeLU MLP 替换为 SwiGLU MLP，注意参数量控制 | SwiGLU 消融结果 |
| LoRA | 实现 `LoRALinear`，冻结 base 权重，只训练 adapter | LoRA 训练结果 |
| 参数量统计 | 对每个架构报告 total params 和 trainable params | 参数表 |
| 代码开关 | 所有改动都通过 config flag 控制，不能破坏默认 nanoGPT | config 化实现 |

C 的实现原则：

```text
1. 每次只改一个变量。
2. 默认配置必须还能跑。
3. 架构改动要能通过命令行开关打开/关闭。
4. 尽量保持参数量和 14M baseline 接近。
5. LoRA 要单独报告 trainable params，而不是只报 total params。
```

建议 C 的代码改动路径：

| 文件 | 可能改动 |
|---|---|
| `nanogpt/model.py` | 增加 RoPE、RMSNorm、SwiGLU、LoRA 模块 |
| `nanogpt/train.py` | 增加 trainable params 统计、冻结 base 权重、LoRA checkpoint 保存 |
| `nanogpt/config/train_wikitext2_14m.py` | 增加可选 config flags |
| `nanogpt/config/` | 新增消融配置，如 `train_wikitext2_14m_rope.py` |

LoRA 第一版建议只做 attention projection：

```text
Block.attn.c_attn
Block.attn.c_proj
```

如果 attention-only LoRA 稳定，再加入：

```text
Block.mlp.c_fc
Block.mlp.c_proj
```

推荐 LoRA 对比：

| 实验编号 | Base | LoRA target | rank | 训练方式 |
|---|---|---|---:|---|
| C1 | 14M scratch checkpoint | attention only | 4 | freeze base，只训 LoRA |
| C2 | 14M scratch checkpoint | attention only | 8 | freeze base，只训 LoRA |
| C3 | 14M scratch checkpoint | attention + MLP | 8 | freeze base，只训 LoRA |
| C4 可选 | pretrained GPT-2 | attention only | 8 | 单独作为 pretrained + LoRA |

注意：如果使用 pretrained GPT-2 + LoRA，必须在报告中单独成类，不能和 from-scratch nanoGPT 直接混为同一组。

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

## 5. 三人协作计划

| 阶段 | A | B | C | 共同产出 |
|---|---|---|---|---|
| 统一基础设施 | 整理共享 `prepare.py` | 检查 LSTM 接入共享数据 | 检查 nanoGPT 接入共享数据 | 三方 token 数一致 |
| 统一评测 | 写/整合 eval 脚本 | 接入 LSTM checkpoint | 接入 nanoGPT checkpoint | 一张最终 baseline 表 |
| 默认复现 | 汇总日志与图表 | 跑 nanoGPT 14M 多种子 | 跑 nanoGPT 44M/默认配置 | 默认 nanoGPT 结果 |
| 超参优化 | 整理结果表 | 主跑 LR/dropout/block_size | 辅助分析过拟合 | 最优超参配置 |
| 架构优化 | 画消融图 | 辅助复跑对照组 | 实现 RoPE/RMSNorm/SwiGLU | 架构消融表 |
| LoRA 实验 | 设计对比表 | 跑 LoRA 多种子/日志 | 实现 LoRA 模块 | LoRA 对比结果 |
| 报告写作 | Results 表与图 | Experimental Setup | Method/Architecture | 最终报告与答辩材料 |

---

## 6. Baseline 收尾任务

| 模块 | 必做任务 | 可选任务 |
|---|---|---|
| N-gram | 接入共享数据；记录 val/test PPL；提供同 prompt 输出 | Kneser-Ney smoothing；报告 n-gram table size |
| LSTM | 接入共享数据；保存可复现实验配置；提供同 prompt 输出 | 跑 3 seeds；尝试 20M-30M stronger LSTM |
| nanoGPT | 作为主优化对象持续迭代 | LoRA、RoPE、RMSNorm、SwiGLU、多种子 |

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
