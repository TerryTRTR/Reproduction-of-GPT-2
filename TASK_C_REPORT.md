# Task C Report: nanoGPT Architecture and LoRA Experiments

> Date: 2026-06-11  
> Model: nanoGPT 14M  
> Dataset: WikiText-2 raw v1, GPT-2 BPE  
> Standard data: `data/wikitext2/{train,val,test}.bin`

## 1. Objective

Task C focuses on improving the 14M nanoGPT model through architectural changes and parameter-efficient adaptation. The main goals are:

1. Add modern Transformer components behind config flags.
2. Run single-factor architecture ablations.
3. Test whether combined modern components improve over the default architecture.
4. Implement and evaluate attention-only LoRA.

All variants use the same unified WikiText-2 data and GPT-2 BPE tokenizer as Task A.

## 2. Implemented Features

The core implementation is in `nanogpt/model.py` and `nanogpt/train.py`.

| Feature | Code location | Description |
|---|---|---|
| RoPE | `model.py::_apply_rope`, `CausalSelfAttention` | Replaces learned absolute position embedding with rotary position embedding on q/k |
| RMSNorm | `model.py::RMSNorm` | Replaces LayerNorm with RMS normalization |
| SwiGLU | `model.py::MLP` | Replaces GeLU MLP with gated SwiGLU MLP |
| LoRA | `model.py::LoRALinear` | Adds low-rank adapters around selected Linear layers |
| LoRA freezing | `model.py::mark_only_lora_as_trainable` | Freezes base parameters and trains only adapter weights |
| LoRA checkpoint init | `train.py::load_base_weights_into_lora` | Loads a full-training base checkpoint into LoRA-wrapped layers |
| Trainable param report | `train.py` | Prints trainable params vs total params |

The new flags added to `GPTConfig` are:

```python
use_rope: bool = False
use_rmsnorm: bool = False
use_swiglu: bool = False
swiglu_hidden_mult: float = 8 / 3
use_lora: bool = False
lora_rank: int = 8
lora_alpha: float = 16.0
lora_dropout: float = 0.05
lora_targets: str = "attn"
lora_freeze_base: bool = True
```

## 3. Experiment Configurations

All architecture configs inherit from `nanogpt/config/train_wikitext2_14m.py`, which uses:

| Parameter | Value |
|---|---:|
| n_layer | 5 |
| n_head | 4 |
| n_embd | 224 |
| block_size | 512 |
| dropout | 0.1 |
| learning_rate | 6e-4 |
| max_iters | 6000 |
| seed | 1337 |

Additional C configs:

| Variant | Config file | Main change |
|---|---|---|
| RoPE | `nanogpt/config/train_wikitext2_14m_rope.py` | `use_rope=True` |
| RMSNorm | `nanogpt/config/train_wikitext2_14m_rmsnorm.py` | `use_rmsnorm=True` |
| SwiGLU | `nanogpt/config/train_wikitext2_14m_swiglu.py` | `use_swiglu=True` |
| Modern | `nanogpt/config/train_wikitext2_14m_modern.py` | RoPE + RMSNorm + SwiGLU |
| LoRA attn r8 | `nanogpt/config/train_wikitext2_14m_lora_attn.py` | attention-only LoRA, rank 8 |

The LoRA run uses:

```python
use_lora = True
lora_rank = 8
lora_alpha = 16.0
lora_dropout = 0.05
lora_targets = "attn"
lora_freeze_base = True
lora_base_checkpoint = "out-wikitext2-14m/ckpt.pt"
learning_rate = 1e-3
max_iters = 3000
```

## 4. Unified Evaluation Results

The following table reports full validation/test split evaluation from `eval/results_nanogpt14m_*.json`.

| Variant | Params | Best Iter | Val Loss ↓ | Val PPL ↓ | Test Loss ↓ | Test PPL ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Default arch | 14.28M | 1500 | 5.1836 | 178.32 | 5.2292 | 186.64 |
| RMSNorm | 14.28M | 1500 | 5.1775 | 177.24 | 5.2200 | 184.93 |
| SwiGLU | 14.28M | 1400 | 5.1717 | 176.22 | 5.2108 | 183.25 |
| LoRA attn r8 | 14.33M | 1800 | 5.1606 | 174.27 | 5.2063 | 182.41 |
| RoPE | 14.28M | 1000 | 5.0916 | 162.65 | 5.1537 | 173.08 |
| Modern | 14.28M | 1200 | **5.0466** | **155.50** | **5.1043** | **164.72** |

## 5. Relative Improvements

Relative to the default architecture:

| Variant | Δ Val Loss | Val PPL Improvement | Test PPL Improvement |
|---|---:|---:|---:|
| RMSNorm | 0.0061 | 0.60% | 0.92% |
| SwiGLU | 0.0119 | 1.18% | 1.82% |
| LoRA attn r8 | 0.0230 | 2.27% | 2.27% |
| RoPE | 0.0920 | 8.79% | 7.27% |
| Modern | **0.1369** | **12.80%** | **11.74%** |

## 6. Analysis

### 6.1 RoPE is the strongest single architectural change

RoPE reduces validation PPL from `178.32` to `162.65`, a relative improvement of `8.79%`. This is much larger than the gains from RMSNorm or SwiGLU alone. In this setup, changing positional encoding matters more than changing normalization or MLP activation.

### 6.2 RMSNorm and SwiGLU provide small but positive gains

RMSNorm improves validation PPL by `0.60%`, while SwiGLU improves it by `1.18%`. The gains are modest individually, but both changes are parameter-efficient and remain useful when combined with RoPE.

### 6.3 Combined modern architecture performs best

The Modern variant combines RoPE, RMSNorm, and SwiGLU. It reaches the best validation and test results:

```text
Val PPL:  155.50
Test PPL: 164.72
```

This is a `12.80%` validation PPL improvement over the default architecture. The combined result is better than any single ablation, suggesting that the improvements are at least partially complementary.

### 6.4 LoRA works, but the gain is limited in this setting

Attention-only LoRA with rank 8 improves validation PPL from `178.32` to `174.27`, a `2.27%` improvement. This confirms that the adapter path is functional and can improve the base model with relatively few extra trainable parameters.

However, LoRA does not outperform full architectural changes. This is expected because LoRA starts from the already trained baseline and only adapts a small subset of parameters, while RoPE and Modern change the model's inductive bias during training.

## 7. Reproduction Commands

Run from the `nanogpt/` directory.

```bash
python train.py config/train_wikitext2_14m_rope.py --compile=False
python train.py config/train_wikitext2_14m_rmsnorm.py --compile=False
python train.py config/train_wikitext2_14m_swiglu.py --compile=False
python train.py config/train_wikitext2_14m_modern.py --compile=False
python train.py config/train_wikitext2_14m_lora_attn.py --compile=False
```

Evaluate with the unified script from the repository root:

```bash
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt \
  --checkpoint nanogpt/out-wikitext2-14m-rope/ckpt.pt \
  --device cuda --batch_size 8 \
  --output eval/results_nanogpt14m_rope.json
```

Repeat with the corresponding checkpoint/output paths for RMSNorm, SwiGLU, Modern, and LoRA.

## 8. Deliverables

| Deliverable | File |
|---|---|
| Model implementation | `nanogpt/model.py` |
| Training support | `nanogpt/train.py` |
| RoPE config | `nanogpt/config/train_wikitext2_14m_rope.py` |
| RMSNorm config | `nanogpt/config/train_wikitext2_14m_rmsnorm.py` |
| SwiGLU config | `nanogpt/config/train_wikitext2_14m_swiglu.py` |
| Modern config | `nanogpt/config/train_wikitext2_14m_modern.py` |
| LoRA config | `nanogpt/config/train_wikitext2_14m_lora_attn.py` |
| Evaluation JSONs | `eval/results_nanogpt14m_*.json` |

## 9. Report-Ready Conclusion

Task C shows that architectural choices significantly affect small-scale GPT training on WikiText-2. Among single changes, RoPE gives the largest improvement, reducing validation PPL from `178.32` to `162.65`. RMSNorm and SwiGLU provide smaller but consistent gains. Combining RoPE, RMSNorm, and SwiGLU gives the best result, lowering validation PPL to `155.50` and test PPL to `164.72`, corresponding to a `12.80%` validation PPL reduction over the default architecture.

The LoRA experiment verifies that attention-only low-rank adaptation is implemented correctly and improves the base model modestly, reaching validation PPL `174.27`. For this project, the strongest C result is therefore the Modern architecture variant, while LoRA is best presented as a parameter-efficient adaptation experiment rather than the main performance winner.
