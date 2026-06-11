# Next Steps Plan: Final Three-Day Push

> Date: 2026-06-11  
> Goal: turn A/B/C into a coherent final result, identify the final best model, and prepare report-ready tables.

## Current Status

| Track | Status | Main Result |
|---|---|---|
| A: data/eval/report infrastructure | Complete | Unified data in `data/wikitext2`; shared eval/sample/figures |
| B: hyperparameter tuning | Complete locally | B6 full eval: val PPL `155.59`, test PPL `162.14` |
| C: architecture and LoRA | Complete | Modern = RoPE + RMSNorm + SwiGLU, test PPL `164.72` |
| Combined final candidate | Complete locally | Modern+B6 full eval: val PPL `134.29`, test PPL `142.15` |

The key missing bridge is the combined experiment:

```text
Modern architecture + B6 hyperparameters
```

## Day 1: Fill Critical Evaluation Gaps

1. Recreate the B6 checkpoint locally because it is not present in the repo checkout.
2. Run unified val/test eval for B6 and save `eval/results_nanogpt14m_b6.json`.
3. Add a reusable config file for B6 so future runs do not rely on long command-line overrides.
4. Add a reusable config file for Modern+B6.
5. Start Modern+B6 training.

Success criteria:

```text
B6 has a local checkpoint and unified val/test JSON. DONE.
Modern+B6 has completed with unified val/test JSON. DONE.
```

Day 1 result:

| Model | Val PPL | Test PPL |
|---|---:|---:|
| B6 tuned | 155.59 | 162.14 |
| Modern+B6 | **134.29** | **142.15** |

## Day 2: Decide Final Model

1. Compare:
   - nanoGPT default
   - B6 tuned
   - Modern architecture
   - Modern+B6
2. Modern+B6 currently beats B6 and should become the final model candidate.
3. Optional only if time remains: run one extra seed for Modern+B6.

## Day 3: Final Report and Presentation Cleanup

1. Create/update a final combined report table.
2. Regenerate final figures with all important models. DONE.
3. Write final narrative:
   - A fixed comparability.
   - B found that `block_size=256` and `dropout=0.2` are crucial.
   - C found that RoPE is the strongest architecture change.
   - The final model is selected by unified test PPL.
4. Check every table uses the same `data/wikitext2` eval protocol.
5. Push final artifacts to `main`.

## Priority Order

1. B6 checkpoint + test eval.
2. Modern+B6 run.
3. Final comparison table.
4. Extra seeds only if the final model choice is still ambiguous.

## Commands

Run B6:

```bash
cd nanogpt
../nanogpt/.venv/bin/python train.py config/train_wikitext2_14m_b6.py --compile=False
```

Evaluate B6:

```bash
cd ..
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt \
  --checkpoint nanogpt/out-wikitext2-14m-b6/ckpt.pt \
  --device cuda --batch_size 8 \
  --output eval/results_nanogpt14m_b6.json
```

Run Modern+B6:

```bash
cd nanogpt
../nanogpt/.venv/bin/python train.py config/train_wikitext2_14m_modern_b6.py --compile=False
```

Evaluate Modern+B6:

```bash
cd ..
nanogpt/.venv/bin/python eval/eval_lm.py --model nanogpt \
  --checkpoint nanogpt/out-wikitext2-14m-modern-b6/ckpt.pt \
  --device cuda --batch_size 8 \
  --output eval/results_nanogpt14m_modern_b6.json
```
