# Final Results Summary

> Date: 2026-06-11  
> Evaluation protocol: unified full-split eval via `eval/eval_lm.py`  
> Data: `data/wikitext2/{train,val,test}.bin`

## 1. Main Comparison

| Model | Params | Val Loss ↓ | Val PPL ↓ | Test Loss ↓ | Test PPL ↓ | Role |
|---|---:|---:|---:|---:|---:|---|
| 3-gram | - | 7.1944 | 1331.93 | 7.2151 | 1359.86 | Statistical baseline |
| LSTM | 13.92M | 5.5037 | 245.59 | 5.5611 | 260.11 | Neural baseline |
| nanoGPT 14M default | 14.28M | 5.1721 | 176.29 | 5.2149 | 183.99 | Reproduction baseline |
| nanoGPT 14M B6 tuned | 14.28M | 5.0472 | 155.59 | 5.0885 | 162.14 | Best Task B hyperparams |
| nanoGPT 14M Modern | 14.28M | 5.0466 | 155.50 | 5.1043 | 164.72 | Best Task C architecture |
| **nanoGPT 14M Modern+B6** | **14.28M** | **4.9000** | **134.29** | **4.9569** | **142.15** | **Final model candidate** |

## 2. Final Model

The current best model is:

```text
nanoGPT 14M Modern+B6
```

Configuration:

```text
Architecture: RoPE + RMSNorm + SwiGLU
learning_rate = 3e-4
min_lr = 3e-5
dropout = 0.2
block_size = 256
max_iters = 6000
seed = 1337
```

Files:

| Artifact | Path |
|---|---|
| Config | `nanogpt/config/train_wikitext2_14m_modern_b6.py` |
| Unified eval JSON | `eval/results_nanogpt14m_modern_b6.json` |
| Local checkpoint | `nanogpt/out-wikitext2-14m-modern-b6/ckpt.pt` |

The checkpoint is intentionally not tracked by git because checkpoint files are still ignored.

## 3. Final Figures

Perplexity comparison:

![Final WikiText-2 perplexity comparison](./eval/figures/final_ppl_comparison.png)

Loss comparison:

![Final WikiText-2 loss comparison](./eval/figures/final_loss_comparison.png)

Task B hyperparameter sweep:

![Task B hyperparameter sweep](./eval/figures/task_b_hparam_sweep.png)

Task C architecture and LoRA ablation:

![Task C architecture ablation](./eval/figures/task_c_arch_ablation.png)

nanoGPT 3000-iteration training-curve comparison:

![nanoGPT default vs Modern+B6 training curve](./eval/figures/nanogpt_default_vs_modern_b6_3k_loss.png)

The training-curve figure is a shortened 3000-iteration diagnostic run. The main result table above still uses the 6000-iteration checkpoints and full-split evaluation.

Training-curve diagnostic endpoints:

| Run | Iterations | Train Loss | Val Loss | Val PPL |
|---|---:|---:|---:|---:|
| nanoGPT default 3k | 3000 | 3.2742 | 5.1467 | 171.86 |
| nanoGPT Modern+B6 3k | 3000 | 4.1813 | 5.0379 | 154.14 |

## 4. Improvements

Modern+B6 vs default nanoGPT:

```text
Val PPL:  176.29 -> 134.29  = 23.8% reduction
Test PPL: 183.99 -> 142.15  = 22.7% reduction
```

Modern+B6 vs LSTM:

```text
Val PPL:  245.59 -> 134.29  = 45.3% reduction
Test PPL: 260.11 -> 142.15  = 45.4% reduction
```

Modern+B6 vs B6 tuned:

```text
Val PPL:  155.59 -> 134.29  = 13.7% reduction
Test PPL: 162.14 -> 142.15  = 12.3% reduction
```

Modern+B6 vs Modern architecture alone:

```text
Val PPL:  155.50 -> 134.29  = 13.6% reduction
Test PPL: 164.72 -> 142.15  = 13.7% reduction
```

## 5. Interpretation

The final result shows that Task B and Task C are complementary:

1. Task B found a better training regime for WikiText-2: lower LR, higher dropout, and shorter context.
2. Task C found a better architecture: RoPE is the strongest single change, and RoPE + RMSNorm + SwiGLU performs best among architecture variants.
3. Combining both gives the best result, substantially outperforming either B6 or Modern alone.

This gives the final project a clean story:

```text
Fair baseline setup -> hyperparameter tuning -> architecture ablation -> combined final model
```

## 6. Remaining Work

Before the final deadline:

1. Merge `TASK_A_REPORT.md`, `TASK_B_REPORT.md`, `TASK_C_REPORT.md`, and this summary into the final project report.
2. Add one paragraph explaining why full-split eval numbers differ from sampled training-log validation loss.
3. Optional: run one extra seed for Modern+B6 if time allows.
4. Prepare slides around the final story: fair baseline -> tuning -> architecture -> combined final model.
