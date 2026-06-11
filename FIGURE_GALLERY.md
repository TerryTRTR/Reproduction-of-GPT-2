# Figure Gallery

> Date: 2026-06-11  
> Dataset: WikiText-2 raw v1  
> Evaluation: unified full-split evaluation

这个文件专门集中展示项目里的主要对比图，方便检查结果、写报告和做展示。

## 1. Task A: Baseline Comparison

Task A 统一了数据和评估方式，比较了 3-gram、LSTM 和 nanoGPT 14M。

### Perplexity

<img src="./eval/figures/ppl_comparison.png" alt="Task A perplexity comparison" width="900">

### Loss

<img src="./eval/figures/loss_comparison.png" alt="Task A loss comparison" width="900">

### LSTM Training Curve

<img src="./eval/figures/lstm_loss_curve.png" alt="LSTM training curve" width="900">

### nanoGPT Training Curve

下面三张图来自重新训练的 3000 iters 日志，用于展示 default nanoGPT 和最终 `Modern+B6` 的 train/val loss 变化。注意，这些曲线用于观察训练趋势；最终主结果仍以 6000 iters 的 full-split evaluation 为准。

#### Default nanoGPT

<img src="./eval/figures/nanogpt_default_3k_loss_curve.png" alt="nanoGPT default 3000 iters training curve" width="900">

#### Modern+B6 nanoGPT

<img src="./eval/figures/nanogpt_modern_b6_3k_loss_curve.png" alt="nanoGPT Modern+B6 3000 iters training curve" width="900">

#### Default vs Modern+B6

<img src="./eval/figures/nanogpt_default_vs_modern_b6_3k_loss.png" alt="nanoGPT default vs Modern+B6 3000 iters training curve" width="900">

## 2. Task B: Hyperparameter Tuning

Task B 比较了不同 learning rate、dropout 和 block size。最好的配置是 B6。

<img src="./eval/figures/task_b_hparam_sweep.png" alt="Task B hyperparameter sweep" width="900">

关键结论：

```text
B6 = lr=3e-4, dropout=0.2, block_size=256
```

## 3. Task C: Architecture and LoRA Ablation

Task C 比较了 RoPE、RMSNorm、SwiGLU、LoRA 以及组合后的 Modern 架构。

<img src="./eval/figures/task_c_arch_ablation.png" alt="Task C architecture and LoRA ablation" width="900">

关键结论：

```text
RoPE is the strongest single architecture change.
Modern = RoPE + RMSNorm + SwiGLU is the best C-only variant.
```

## 4. Final Comparison

最终模型是 Modern 架构和 B6 超参数的组合：

```text
Modern+B6 = RoPE + RMSNorm + SwiGLU
          + lr=3e-4
          + dropout=0.2
          + block_size=256
```

### Final Perplexity

<img src="./eval/figures/final_ppl_comparison.png" alt="Final perplexity comparison" width="900">

### Final Loss

<img src="./eval/figures/final_loss_comparison.png" alt="Final loss comparison" width="900">

## 5. Final Result Table

| Model | Val PPL ↓ | Test PPL ↓ |
|---|---:|---:|
| 3-gram | 1331.93 | 1359.86 |
| LSTM | 245.59 | 260.11 |
| nanoGPT 14M default | 176.29 | 183.99 |
| nanoGPT B6 tuned | 155.59 | 162.14 |
| nanoGPT Modern | 155.50 | 164.72 |
| **nanoGPT Modern+B6** | **134.29** | **142.15** |

最终推荐汇报模型：

```text
nanoGPT 14M Modern+B6
```
