# Baseline 对比记录

> 日期：2026-06-10  
> 目标数据集：WikiText-2 raw v1  
> 目标分词器：GPT-2 BPE  
> 定性对比统一 prompt：`The meaning of life is`

---

## 1. 代码概览

| 模块 | 文件夹 | 模型 | 当前定位 |
|---|---|---|---|
| N-gram | `ngram/` | 插值 3-gram + 平滑 | 非神经统计 baseline |
| LSTM | `LSTM_baseline/` | 2 层 LSTM + 权重绑定 | 同参数量级神经 baseline |
| nanoGPT | `nanogpt/` | GPT-style causal Transformer | 主复现模型 |

理想对比协议是：三方使用同一份 WikiText-2 train / validation / test 划分、同一个 GPT-2 BPE tokenizer、同一个验证/测试评测脚本，并在定性分析中使用同一组 prompt。

---

## 2. 统一评测指标

统一口径：

- 数据：`data/wikitext2/{train,val,test}.bin`
- 分词器：GPT-2 BPE，`vocab_size=50257`
- 评测入口：`eval/eval_lm.py`
- PPL：`exp(loss)`
- BPC：`loss / ln(2)`

| Model | Params | Val Loss ↓ | Val PPL ↓ | Val BPC ↓ | Test Loss ↓ | Test PPL ↓ | Test BPC ↓ | 来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3-gram | - | 7.1944 | 1331.93 | 10.3793 | 7.2151 | 1359.86 | 10.4092 | `eval/results_ngram_unified.json` |
| LSTM | 13.92M | 5.5037 | 245.59 | 7.9401 | 5.5611 | 260.11 | 8.0230 | `eval/results_lstm_unified.json` |
| nanoGPT 14M | 14.28M | 5.1721 | 176.29 | 7.4618 | 5.2149 | 183.99 | 7.5235 | `eval/results_nanogpt14m_unified.json` |

统一评测命令：

```bash
nanogpt/.venv/bin/python eval/eval_lm.py --model ngram \
  --output eval/results_ngram_unified.json

nanogpt/.venv/bin/python eval/eval_lm.py --model lstm \
  --checkpoint LSTM_baseline/out/ckpt_best.pt --device cuda --batch_size 64 \
  --output eval/results_lstm_unified.json

nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt \
  --checkpoint nanogpt/out-wikitext2-14m/ckpt.pt --device cuda --batch_size 4 \
  --output eval/results_nanogpt14m_unified.json
```

注意：当前 LSTM checkpoint 已在统一 `data/wikitext2` 数据上重新训练；N-gram、LSTM、nanoGPT 的最终评测都指向同一份 `.bin` 数据。

## 3. 旧口径指标（开发阶段记录）

| Model | Params | Val Loss ↓ | Val PPL ↓ | Test Loss ↓ | Test PPL ↓ | 来源 |
|---|---:|---:|---:|---:|---:|---|
| 3-gram | - | 7.1050 | 1218.10 | 7.1243 | 1241.81 | `ngram/results_ngram_3gram.json` |
| LSTM | 13.92M | 5.4346 | 229.20 | 5.4756 | 238.80 | 本地按 `LSTM_baseline/src/config_lstm.py` 复跑 |
| nanoGPT 14M | 14.28M | 5.1680 | 175.56 | - | - | `nanogpt/out-wikitext2-14m/ckpt.pt` |
| nanoGPT 44M | 44.64M | 5.0809 | 160.92 | - | - | `nanogpt/out-wikitext2/ckpt.pt` |

其中 14M nanoGPT 是目前和 13.92M LSTM 最公平的神经模型对比。44M nanoGPT 可以作为更大模型的补充结果，但不应该作为严格同参数量级对比的唯一依据。

---

## 4. 数据统一修正记录

开发阶段三个分支各自生成过不同 `.bin` 文件，token 数如下：

| 模块 | Train Tokens | Val Tokens | Test Tokens |
|---|---:|---:|---:|
| N-gram | 2,428,601 | 251,048 | 287,644 |
| LSTM | 2,415,651 | 249,750 | 286,178 |
| nanoGPT 旧位置 | 2,391,884 | 247,289 | 283,287 |
| `data/wikitext2` / 统一口径 | 2,391,884 | 247,289 | 283,287 |

这说明虽然三方都声称使用 WikiText-2 + GPT-2 BPE，但具体预处理脚本并不是完全一致的。2026-06-10 已修正为以 `data/wikitext2` 作为唯一标准数据，并用这份数据重新训练 LSTM。

旧口径表格只适合作为开发阶段记录；最终论文中应使用第 2 节统一评测指标。

---

## 5. 定性生成样例

解码设置：

| 项目 | 取值 |
|---|---|
| Prompts | `The meaning of life is`; `In the beginning`; `The history of the United States` |
| Max new tokens | 60 |
| Decode | deterministic greedy / top-1 |
| 输出文件 | `eval/fixed_samples_unified.{json,txt}` |

生成命令：

```bash
nanogpt/.venv/bin/python eval/generate_samples.py --device cuda --max_new_tokens 60
```

下面展示第一个 prompt；完整三组样例见 `eval/fixed_samples_unified.txt`。

### 5.1 N-gram 输出

```text
The meaning of life is a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a " a "
```

观察：N-gram 很快进入局部短语循环。这符合预期，因为它只能基于极短上下文建模。

### 5.2 LSTM 输出

```text
The meaning of life is the first time of the war . 
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
```

观察：LSTM 学到了 WikiText 的标题格式和实体文本风格，但仍然存在模板重复，长程一致性弱于 Transformer。

### 5.3 nanoGPT 14M 输出

```text
The meaning of life is a very good enough to be a good job . 
 = = = = = = 
 The marriage of marriage = = = = = 
 The marriage of marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage marriage
```

观察：nanoGPT 的续写比 N-gram 更有结构，也和 LSTM 相比更接近 Transformer 语言模型的模式；但由于它是在小数据 WikiText-2 上从零训练，仍然明显重复高频 WikiText 模板。

---

## 6. 报告图表

图表生成命令：

```bash
nanogpt/.venv/bin/python eval/plot_results.py
```

已生成的报告图：

| Figure | 用途 |
|---|---|
| `eval/figures/ppl_comparison.png` | 三方 validation/test PPL 对比，使用 log y-axis，适合放在 Results 主图 |
| `eval/figures/loss_comparison.png` | 三方 validation/test cross-entropy loss 对比 |
| `eval/figures/lstm_loss_curve.png` | LSTM 训练 loss 与 validation loss 曲线 |

由于当前 nanoGPT 训练脚本没有保存逐步 loss history，暂时只能报告 checkpoint 的 best validation loss 和统一 full-split val/test 评测结果；若后续 B/C 继续跑 nanoGPT 调参，建议把训练日志保存为 JSONL，方便补 nanoGPT loss curve。

---

## 7. Results & Analysis 初稿

统一评测量化结果排序为：

```text
nanoGPT 14M < LSTM < 3-gram
```

其中 loss 越低越好。参数量匹配的神经模型对比是：

```text
LSTM 13.92M:    val loss 5.5037, PPL 245.59
nanoGPT 14.28M: val loss 5.1721, PPL 176.29
```

14M nanoGPT 相比 LSTM 的验证集 perplexity 相对下降约：

```text
(245.59 - 176.29) / 245.59 = 28.2%
```

这一结果说明，在相近参数量级下，Transformer-style self-attention 相比循环结构具有更好的语言建模能力。N-gram 的 validation PPL 为 `1331.93`，明显高于两个神经模型，说明短上下文统计模型很难覆盖 WikiText-2 中较长距离的实体、标题结构和语义依赖。LSTM 相比 N-gram 已有大幅改善，但仍受顺序递归建模和有限隐状态容量限制；nanoGPT 14M 进一步降低到 `176.29` validation PPL，说明 self-attention 对上下文模式的建模更有效。

从 test split 看，三方排序与 validation split 一致：

```text
3-gram test PPL:      1359.86
LSTM test PPL:         260.11
nanoGPT 14M test PPL:  183.99
```

这说明 nanoGPT 的提升并不是只出现在 validation split 上。nanoGPT 的 test loss 为 `5.2149`，略高于 validation loss `5.1721`，属于合理泛化差距。

定性样例也支持量化结论。3-gram 很快退化为局部短语循环；LSTM 能生成 WikiText 风格的标题、实体和 `<|endoftext|>` 分隔符，但仍经常重复模板；nanoGPT 的输出更接近百科文本结构，不过也会在小数据从零训练下反复生成高频标题和实体短语。整体来看，本项目的主要结论不是 nanoGPT 已经具备高质量开放式生成能力，而是在同一 WikiText-2 + GPT-2 BPE 评测协议下，小型 Transformer 在 loss/PPL 上稳定优于传统 N-gram 和参数量相近的 LSTM baseline。

---

## 8. 可写进报告的表述草稿

在统一评测设置下，14.28M 参数的 nanoGPT 在 WikiText-2 validation split 上取得 `5.1721` 的 validation loss，而 13.92M 参数的 LSTM baseline 为 `5.5037`。换算成 perplexity，nanoGPT 从 LSTM 的 `245.59` 降到 `176.29`，相对下降约 `28.2%`。N-gram baseline 的 validation perplexity 超过 `1300`，说明短上下文统计模型在该任务上明显受限。

在 test split 上，nanoGPT 14M 同样取得最低 perplexity：`183.99`，优于 LSTM 的 `260.11` 和 3-gram 的 `1359.86`。定性样例中，N-gram 很快陷入短语循环；LSTM 和 nanoGPT 都学到 WikiText 的标题与实体文本模式，但仍有重复。nanoGPT 的输出相对更有结构，不过小数据从零训练仍然限制了生成质量。
