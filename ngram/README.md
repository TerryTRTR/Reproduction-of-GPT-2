# N-gram Baseline

This folder contains the token-level N-gram language-model baseline used in the final WikiText-2 comparison.

**Owner:** Juran, responsible for the N-gram baseline and LoRA tuning experiments.

## Run

From the repository root:

```bash
nanogpt/.venv/bin/python ngram/baselines/ngram.py \
  --data_dir data/wikitext2 \
  --n 3 \
  --alpha 0.1
```

The unified final evaluation is run through:

```bash
nanogpt/.venv/bin/python eval/eval_lm.py \
  --model ngram \
  --output eval/results_ngram_unified.json
```

All final results use the shared data in:

```text
data/wikitext2/{train,val,test}.bin
```
