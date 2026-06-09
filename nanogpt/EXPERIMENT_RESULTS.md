# nanoGPT WikiText-2 实验结果记录

> 记录时间：2026-06-08  
> 项目角色：成员 C，nanoGPT 复现与调优  
> 主数据集：WikiText-2 raw v1，GPT-2 BPE tokenizer  

---

## 1. 实验目的

本部分实验用于在统一数据集与统一分词设置下，对比传统语言模型 baseline 与 nanoGPT 复现模型的语言建模性能。

项目不追求复现完整 GPT-2 124M 在大规模语料上的预训练结果，而是在单张消费级 GPU 上训练小规模 GPT-style Transformer，并与 N-gram、LSTM baseline 进行量化对比。

---

## 2. 数据集设置

三方模型统一使用 WikiText-2 官方划分：

| Split | Tokens |
|---|---:|
| train | 2,391,884 |
| validation | 247,289 |
| test | 283,287 |

分词方式统一为 GPT-2 BPE，词表大小为 50,257；训练时模型使用向上取整后的 `vocab_size=50304` 以提升计算效率。

数据预处理输出：

```text
data/wikitext2/train.bin
data/wikitext2/val.bin
data/wikitext2/test.bin
```

---

## 3. 模型配置

### 3.1 nanoGPT 14M 参数匹配版本

该配置用于和小型 LSTM baseline 做同参数量级对比。

| Item | Value |
|---|---:|
| Params | 14.28M |
| n_layer | 5 |
| n_head | 4 |
| n_embd | 224 |
| head_dim | 56 |
| block_size | 512 |
| dropout | 0.1 |
| batch_size | 12 |
| gradient_accumulation_steps | 8 |
| tokens / iter | 49,152 |
| learning_rate | 6e-4 |
| warmup_iters | 200 |
| max_iters | 6000 |

配置文件：

```text
config/train_wikitext2_14m.py
```

训练命令：

```bash
python train.py config/train_wikitext2_14m.py --compile=False
```

### 3.2 nanoGPT 44M 主线版本

该配置作为较大的 GPT-2 mini 主线展示模型。

| Item | Value |
|---|---:|
| Params | 44.64M |
| n_layer | 6 |
| n_head | 8 |
| n_embd | 512 |
| head_dim | 64 |
| block_size | 512 |
| dropout | 0.1 |
| batch_size | 12 |
| gradient_accumulation_steps | 8 |
| tokens / iter | 49,152 |
| learning_rate | 6e-4 |
| warmup_iters | 500 |
| max_iters | 20000 |

配置文件：

```text
config/train_wikitext2.py
```

---

## 4. 当前结果

checkpoint 只在验证集 loss 创新低时保存，因此下表记录的是当前保存的最佳验证集结果。

| Model | Params | Best Iter | Val Loss ↓ | Val PPL ↓ | Notes |
|---|---:|---:|---:|---:|---|
| LSTM baseline | ~14M | - | 5.40 | 221.41 | 同参数量级神经 baseline |
| nanoGPT 14M | 14.28M | 1600 | 5.1680 | 175.56 | 参数匹配 Transformer |
| nanoGPT 44M | 44.64M | 750 | 5.0809 | 160.92 | 更大 GPT-2 mini，后期过拟合明显 |

相对于 LSTM baseline，14M nanoGPT 的 perplexity 改善为：

```text
(221.41 - 175.56) / 221.41 = 20.7%
```

虽然交叉熵 loss 只下降约 `0.23`，但换算成 perplexity 后约有 20% 的相对改善。

---

## 5. 过拟合现象

44M nanoGPT 在较早阶段达到最佳验证集 loss，之后训练 loss 继续下降，但 validation loss 开始上升。这说明模型开始记忆 WikiText-2 训练集，而泛化性能下降。

该现象是合理的，原因包括：

- WikiText-2 训练集只有约 2.39M BPE tokens；
- 44.64M 参数量相对于该数据集偏大；
- 每个 iteration 使用 49,152 tokens，训练到 750 iter 已经相当于看过约 36.9M tokens，即训练集约 15.4 个 epoch；
- Transformer 从零训练通常更依赖大规模语料，小数据上更容易过拟合。

因此报告中应使用 `best val loss`，而不是最后一个 iteration 的 loss。

---

## 6. 与原始 nanoGPT 仓库的区别

原始 Karpathy nanoGPT 仓库主要复现 GPT-2 在 OpenWebText 上的大规模预训练，其标准配置接近：

| Item | Original nanoGPT GPT-2 style run |
|---|---:|
| Dataset | OpenWebText |
| Params | 124M |
| Context length | 1024 |
| Training tokens | hundreds of billions scale |
| Hardware | multi-GPU A100 setup |

本项目设置为：

| Item | This project |
|---|---:|
| Dataset | WikiText-2 |
| Params | 14M / 44M |
| Context length | 512 |
| Training tokens | 2.39M unique train tokens |
| Hardware | single RTX 5070 Laptop GPU |

因此原始仓库或 GPT-2 checkpoint 中的 loss 数字不能直接作为本实验的目标值。本项目的重点是相同数据集、相同 tokenizer、相同评测流程下的 baseline 对比。

---

## 7. 公平性说明

N-gram 是非神经统计模型，不具有和 neural network 相同意义下的可训练参数量。因此 N-gram baseline 不参与等参数对齐，其作用是作为传统统计语言模型参照。

LSTM 和 nanoGPT 都是神经语言模型，因此主要采用同参数量级对比。当前 14M nanoGPT 与约 14M LSTM baseline 属于参数匹配设置，可用于分析 Transformer 相对于循环神经网络的结构优势。

44M nanoGPT 不属于与 LSTM 严格参数匹配的结果，应作为更大模型规模下的补充结果展示，而不是唯一公平对比依据。

---

## 8. 可写入报告的结论草稿

在参数匹配设置下，14.28M 参数的 nanoGPT 在 WikiText-2 validation split 上取得 `5.1680` 的最佳 validation loss，对应 perplexity 为 `175.56`。相同参数量级的 LSTM baseline validation loss 为约 `5.40`，perplexity 为 `221.41`。nanoGPT 将 perplexity 降低约 `20.7%`，说明 self-attention 架构在相同数据与分词设置下优于循环结构。

不过，该优势并非压倒性。主要原因是 WikiText-2 数据规模较小，只有约 2.39M BPE training tokens，而 Transformer 从零训练通常需要更大的语料才能充分发挥优势。实验中 44.64M 参数的 nanoGPT 虽然取得更低的最佳 validation loss `5.0809`，但很快出现 train loss 下降而 validation loss 上升的过拟合现象。这表明在小数据设置下，继续扩大模型规模并不一定带来稳定泛化收益。

---

## 9. 后续建议

建议后续围绕 14M 参数匹配版本继续调优：

| Experiment | Change |
|---|---|
| Lower LR | `learning_rate=3e-4` 或 `2e-4` |
| Stronger regularization | `dropout=0.2` |
| Shorter training | 对比 best val loss 出现在 1000-3000 iter 的情况 |
| Multi-seed | 至少 3 个 random seed，报告 mean ± std |
| Architecture ablation | RoPE / RMSNorm / SwiGLU |

当前最重要的是保留并报告最佳 checkpoint，而不是最后 checkpoint。
