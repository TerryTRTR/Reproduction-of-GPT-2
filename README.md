# Reproduction and Optimization of nanoGPT on WikiText-2

This repository contains a CS182 course project that reproduces a compact GPT-style language model and improves it on WikiText-2. The final comparison uses a shared GPT-2 BPE token stream and a unified evaluation script for all baselines.

Final reported model:

```text
nanoGPT 14M Modern+B6
= RoPE + RMSNorm + SwiGLU
+ learning_rate=3e-4
+ dropout=0.2
+ block_size=256
```

Final full-split evaluation:

| Model | Val PPL | Test PPL |
|---|---:|---:|
| 3-gram | 1331.93 | 1359.86 |
| LSTM | 245.59 | 260.11 |
| nanoGPT 14M default | 176.29 | 183.99 |
| nanoGPT B6 tuned | 155.59 | 162.14 |
| nanoGPT Modern | 155.50 | 164.72 |
| **nanoGPT Modern+B6** | **134.29** | **142.15** |

## Team Contributions

| Member | Main responsibility | Main files/modules |
|---|---|---|
| Xuzhenyu | nanoGPT reproduction, project-wide data/evaluation unification, final repository integration | `nanogpt/`, `data/wikitext2/`, `eval/`, root `README.md` |
| Julinjie | LSTM baseline and nanoGPT hyperparameter tuning | `LSTM_baseline/`, `nanogpt/config/train_wikitext2_14m_b6.py`, `nanogpt/phase1_results.json`, `nanogpt/phase2_results.json` |
| Juran | N-gram baseline and LoRA tuning/experiments | `ngram/`, `nanogpt/config/train_wikitext2_14m_lora_attn.py`, LoRA-related code paths in `nanogpt/model.py` and `nanogpt/train.py` |

## Repository Layout

```text
.
├── data/wikitext2/          # Shared WikiText-2 GPT-2 BPE data
├── eval/                    # Unified evaluation, sample generation, and plotting
├── ngram/                   # N-gram baseline
├── LSTM_baseline/           # LSTM baseline
├── nanogpt/                 # GPT-style model, training code, configs, ablations
└── README.md                # This file
```

The old per-model WikiText-2 data copies were removed from version control. All final experiments should use:

```text
data/wikitext2/{train,val,test}.bin
```

## Environment Setup

The project was run with Python 3.12 and PyTorch on CUDA. A CUDA GPU is recommended for nanoGPT training, but CPU can run small checks.

Create and activate an environment:

```bash
cd /path/to/Reproduction-of-GPT-2
python -m venv nanogpt/.venv
source nanogpt/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If dependencies are already installed, you can simply use:

```bash
source nanogpt/.venv/bin/activate
```

## Data Preparation

The shared WikiText-2 data is already included in `data/wikitext2`. To regenerate it:

```bash
python data/wikitext2/prepare.py
```

Expected token counts:

| Split | Tokens |
|---|---:|
| Train | 2,391,884 |
| Validation | 247,289 |
| Test | 283,287 |

## Training

Run commands from the repository root unless otherwise stated.

### nanoGPT Default Baseline

```bash
cd nanogpt
.venv/bin/python train.py config/train_wikitext2_14m.py --compile=False
cd ..
```

### nanoGPT B6 Tuned Model

```bash
cd nanogpt
.venv/bin/python train.py config/train_wikitext2_14m_b6.py --compile=False
cd ..
```

### nanoGPT Modern+B6 Final Model

```bash
cd nanogpt
.venv/bin/python train.py config/train_wikitext2_14m_modern_b6.py --compile=False
cd ..
```

### LSTM Baseline

```bash
cd LSTM_baseline
../nanogpt/.venv/bin/python src/lstm.py --config=src/config_lstm.py
cd ..
```

The LSTM config points to the shared `../data/wikitext2` directory.

## Unified Evaluation

All final numbers are produced by `eval/eval_lm.py`.

### N-gram

```bash
nanogpt/.venv/bin/python eval/eval_lm.py \
  --model ngram \
  --output eval/results_ngram_unified.json
```

### LSTM

```bash
nanogpt/.venv/bin/python eval/eval_lm.py \
  --model lstm \
  --checkpoint LSTM_baseline/out/ckpt_best.pt \
  --device cuda \
  --batch_size 64 \
  --output eval/results_lstm_unified.json
```

### nanoGPT

```bash
nanogpt/.venv/bin/python eval/eval_lm.py \
  --model nanogpt \
  --checkpoint nanogpt/out-wikitext2-14m-modern-b6/ckpt.pt \
  --device cuda \
  --batch_size 8 \
  --output eval/results_nanogpt14m_modern_b6.json
```

Replace the checkpoint/output paths to evaluate the default, B6, RoPE, RMSNorm, SwiGLU, Modern, or LoRA variants.

## Figures and Samples

Regenerate the main figures:

```bash
nanogpt/.venv/bin/python eval/plot_results.py
nanogpt/.venv/bin/python eval/plot_final_results.py
nanogpt/.venv/bin/python eval/plot_ablation_results.py
```

Generate fixed-prompt qualitative samples:

```bash
nanogpt/.venv/bin/python eval/generate_samples.py --device cuda --max_new_tokens 60
```

Important outputs:

```text
eval/figures/
eval/results_*.json
eval/fixed_samples_unified.txt
```

## Reproducing the Final Result Table

Use the already generated JSON files in `eval/`, or rerun:

```bash
nanogpt/.venv/bin/python eval/eval_lm.py --model ngram --output eval/results_ngram_unified.json
nanogpt/.venv/bin/python eval/eval_lm.py --model lstm --checkpoint LSTM_baseline/out/ckpt_best.pt --device cuda --batch_size 64 --output eval/results_lstm_unified.json
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt --checkpoint nanogpt/out-wikitext2-14m/ckpt.pt --device cuda --batch_size 4 --output eval/results_nanogpt14m_unified.json
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt --checkpoint nanogpt/out-wikitext2-14m-b6/ckpt.pt --device cuda --batch_size 8 --output eval/results_nanogpt14m_b6.json
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt --checkpoint nanogpt/out-wikitext2-14m-modern-b6/ckpt.pt --device cuda --batch_size 8 --output eval/results_nanogpt14m_modern_b6.json
```

Then regenerate:

```bash
nanogpt/.venv/bin/python eval/plot_final_results.py
```

## Notes

- Checkpoints and local training outputs are intentionally ignored by git.
- The tracked data in `data/wikitext2` is the single source of truth for final comparisons.
- Training-log validation loss is a random-batch estimate; final tables use full-split evaluation.
- The saved 3000-iteration diagnostic figure is only for showing optimization behavior. The main results use 6000-iteration checkpoints.
