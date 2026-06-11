# Final Deliverables Checklist

> Date: 2026-06-11  
> Final model: `nanoGPT 14M Modern+B6`

## 1. Reports

| File | Purpose | Status |
|---|---|---|
| `TASK_A_REPORT.md` | Unified data, baselines, qualitative samples | Done |
| `TASK_B_REPORT.md` | Hyperparameter tuning from teammates | Done |
| `TASK_C_REPORT.md` | Architecture ablation and LoRA summary | Done |
| `FINAL_RESULTS_SUMMARY.md` | Final result table and main conclusion | Done |
| `PROJECT_STATUS_AND_NEXT_STEPS.md` | Current state and remaining plan | Done |
| `FINAL_REPORT_DRAFT.md` | Draft final report integrating A/B/C | Done |
| `NEXT_STEPS_PLAN.md` | Execution plan for final three days | Done |

## 2. Data and Evaluation

| File or Folder | Purpose | Status |
|---|---|---|
| `data/wikitext2/` | Shared WikiText-2 GPT-2 BPE data | Done |
| `eval/eval_lm.py` | Unified full-split evaluation | Done |
| `eval/fixed_samples_unified.txt` | Fixed-prompt qualitative examples | Done |
| `eval/fixed_samples_unified.json` | Fixed-prompt qualitative examples in JSON | Done |

## 3. Final Result Files

| File | Purpose | Status |
|---|---|---|
| `eval/results_ngram_unified.json` | 3-gram baseline eval | Done |
| `eval/results_lstm_unified.json` | LSTM baseline eval | Done |
| `eval/results_nanogpt14m_unified.json` | Default nanoGPT eval | Done |
| `eval/results_nanogpt14m_b6.json` | Best Task B config eval | Done |
| `eval/results_nanogpt14m_modern.json` | Best Task C architecture eval | Done |
| `eval/results_nanogpt14m_modern_b6.json` | Final Modern+B6 eval | Done |

## 4. Figures

| Figure | Used For | Status |
|---|---|---|
| `eval/figures/ppl_comparison.png` | Task A baseline PPL | Done |
| `eval/figures/loss_comparison.png` | Task A baseline loss | Done |
| `eval/figures/lstm_loss_curve.png` | LSTM retraining curve | Done |
| `eval/figures/nanogpt_default_3k_loss_curve.png` | default nanoGPT 3000-iter training curve | Done |
| `eval/figures/nanogpt_modern_b6_3k_loss_curve.png` | nanoGPT Modern+B6 3000-iter training curve | Done |
| `eval/figures/nanogpt_default_vs_modern_b6_3k_loss.png` | default vs Modern+B6 3000-iter training comparison | Done |
| `eval/figures/task_b_hparam_sweep.png` | Task B hyperparameter sweep | Done |
| `eval/figures/task_c_arch_ablation.png` | Task C architecture/LoRA ablation | Done |
| `eval/figures/final_ppl_comparison.png` | Final model PPL comparison | Done |
| `eval/figures/final_loss_comparison.png` | Final model loss comparison | Done |

Regeneration commands:

```bash
nanogpt/.venv/bin/python eval/plot_final_results.py
nanogpt/.venv/bin/python eval/plot_ablation_results.py
nanogpt/.venv/bin/python eval/plot_nanogpt_training_curve.py
nanogpt/.venv/bin/python eval/plot_nanogpt_training_comparison.py
```

## 5. Final Model Configuration

| File | Purpose | Status |
|---|---|---|
| `nanogpt/config/train_wikitext2_14m_b6.py` | Best Task B training recipe | Done |
| `nanogpt/config/train_wikitext2_14m_modern_b6.py` | Final combined model config | Done |

Final model settings:

```text
Architecture: RoPE + RMSNorm + SwiGLU
learning_rate = 3e-4
min_lr = 3e-5
dropout = 0.2
block_size = 256
max_iters = 6000
seed = 1337
```

## 6. Local Checkpoints

| Checkpoint | Purpose | Git Status |
|---|---|---|
| `nanogpt/out-wikitext2-14m-b6/ckpt.pt` | B6 tuned checkpoint | Ignored |
| `nanogpt/out-wikitext2-14m-modern-b6/ckpt.pt` | Final Modern+B6 checkpoint | Ignored |

These checkpoints are local artifacts and are not committed by default.

## 7. Final Numbers

| Model | Val PPL | Test PPL |
|---|---:|---:|
| 3-gram | 1331.93 | 1359.86 |
| LSTM | 245.59 | 260.11 |
| nanoGPT 14M default | 176.29 | 183.99 |
| nanoGPT B6 tuned | 155.59 | 162.14 |
| nanoGPT Modern | 155.50 | 164.72 |
| **nanoGPT Modern+B6** | **134.29** | **142.15** |

Final recommendation:

```text
Use nanoGPT Modern+B6 as the final reported model.
```
