# Baseline 对比记录

> 日期：2026-06-09  
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

## 2. 当前指标

| Model | Params | Val Loss ↓ | Val PPL ↓ | Test Loss ↓ | Test PPL ↓ | 来源 |
|---|---:|---:|---:|---:|---:|---|
| 3-gram | - | 7.1050 | 1218.10 | 7.1243 | 1241.81 | `ngram/results_ngram_3gram.json` |
| LSTM | 13.92M | 5.4346 | 229.20 | 5.4756 | 238.80 | 本地按 `LSTM_baseline/src/config_lstm.py` 复跑 |
| nanoGPT 14M | 14.28M | 5.1680 | 175.56 | - | - | `nanogpt/out-wikitext2-14m/ckpt.pt` |
| nanoGPT 44M | 44.64M | 5.0809 | 160.92 | - | - | `nanogpt/out-wikitext2/ckpt.pt` |

其中 14M nanoGPT 是目前和 13.92M LSTM 最公平的神经模型对比。44M nanoGPT 可以作为更大模型的补充结果，但不应该作为严格同参数量级对比的唯一依据。

---

## 3. 预处理差异提醒

当前三个分支生成的 token 数略有差异：

| 模块 | Train Tokens | Val Tokens | Test Tokens |
|---|---:|---:|---:|
| N-gram | 2,428,601 | 251,048 | 287,644 |
| LSTM | 2,415,651 | 249,750 | 286,178 |
| nanoGPT | 2,391,884 | 247,289 | 283,287 |

这说明虽然三方都声称使用 WikiText-2 + GPT-2 BPE，但具体预处理脚本并不是完全一致的。最终报告前，应该统一使用同一份 `data/wikitext2/prepare.py` 和同一个评测脚本重新跑最终指标。

在完成统一之前，当前表格适合作为开发阶段对比；最终论文中必须说明使用的是哪一个预处理版本。

---

## 4. 定性生成样例

解码设置：

| 项目 | 取值 |
|---|---|
| Prompt | `The meaning of life is` |
| Max new tokens | 60 |
| Decode | deterministic greedy / top-1 |

### 4.1 N-gram 输出

```text
The meaning of life is a song by American singer and songwriter Mariah Carey , released in the United States , and the other hand , the first time in the United States , and the other hand , the first time in the United States , and the other hand , the first time in the United States , and the other
```

观察：N-gram 很快进入局部短语循环。这符合预期，因为它只能基于极短上下文建模。

### 4.2 LSTM 输出

```text
The meaning of life is a reference to the name of the name of the name " The Great " . 
<|endoftext|> = = = Early years = = = 
<|endoftext|> = = = Early life = = = 
<|endoftext|> The Chapel of Our Lady of the Holy Land , the parish of the parish of the parish of
```

观察：LSTM 学到了 WikiText 的标题格式和实体文本风格，但仍然存在模板重复，并且会不自然地输出 `<|endoftext|>`。

### 4.3 nanoGPT 14M 输出

```text
The meaning of life is a very good enough to be a good job . 
 = = = = = Reception = = = 
 The episode was written by critics . The episode was written by critics and directed by critics . The episode was written by critics , and directed by critics . The episode was written by critics
```

观察：nanoGPT 的续写比 N-gram 更有结构，也和 LSTM 相比更接近 Transformer 语言模型的模式；但由于它是在小数据 WikiText-2 上从零训练，仍然明显重复高频 WikiText 模板。

---

## 5. 结果解释

当前量化结果排序为：

```text
nanoGPT 44M < nanoGPT 14M < LSTM < 3-gram
```

其中 loss 越低越好。参数量匹配的神经模型对比是：

```text
LSTM 13.92M:    val loss 5.4346, PPL 229.20
nanoGPT 14.28M: val loss 5.1680, PPL 175.56
```

14M nanoGPT 相比 LSTM 的验证集 perplexity 相对下降约：

```text
(229.20 - 175.56) / 229.20 = 23.4%
```

这个提升是有意义的，但不是压倒性的。主要原因可能是 WikiText-2 对从零训练 Transformer 来说太小，Transformer 的优势通常需要更大语料和更长预训练才能充分体现。

---

## 6. 可写进报告的表述草稿

在参数量匹配设置下，14.28M 参数的 nanoGPT 在 WikiText-2 validation split 上取得 `5.1680` 的 validation loss，而 13.92M 参数的 LSTM baseline 为 `5.4346`。换算成 perplexity，nanoGPT 从 LSTM 的 `229.20` 降到 `175.56`，相对下降约 `23.4%`。N-gram baseline 的 validation perplexity 超过 `1200`，说明短上下文统计模型在该任务上明显受限。

定性样例中，N-gram 很快陷入短语循环；LSTM 和 nanoGPT 都学到 WikiText 的标题与实体文本模式，但仍有重复。nanoGPT 的输出相对更有结构，不过小数据从零训练仍然限制了生成质量。
