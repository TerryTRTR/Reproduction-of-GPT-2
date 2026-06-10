# Shared WikiText-2 Data

This folder is the project-standard data location for N-gram, LSTM, and nanoGPT
experiments.

```text
data/wikitext2/train.bin
data/wikitext2/val.bin
data/wikitext2/test.bin
```

The files use WikiText-2 raw v1 with GPT-2 BPE tokenization. The preprocessing
matches the nanoGPT baseline run: HuggingFace `Salesforce/wikitext` parquet
splits, rows concatenated per split, and no inserted special tokens.

Regenerate with:

```bash
python data/wikitext2/prepare.py
```
