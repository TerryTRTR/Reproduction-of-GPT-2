"""
Prepare WikiText-2 dataset for language model training.
Uses GPT-2 BPE tokenizer for unified comparison with nanoGPT.

Supports two data sources (auto-fallback):
  1. HuggingFace datasets (primary)
  2. GitHub raw text mirror (fallback, for networks without HF access)

Output:
    train.bin  — uint16 array of tokenized training data
    val.bin    — uint16 array of tokenized validation data
    test.bin   — uint16 array of tokenized test data
    meta.pkl   — dict with vocab_size, itos, stoi
"""

import os
import sys
import pickle
import io
import numpy as np
import tiktoken


# WikiText-2 raw text mirrors (PyTorch examples repo)
WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    "val":   "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
    "test":  "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt",
}


def download_via_github(url: str) -> str:
    """Download raw text from URL and return as string."""
    import urllib.request

    print(f"    Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    # Try UTF-8 first, fall back to latin-1
    for enc in ["utf-8", "latin-1"]:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_wikitext2_huggingface():
    """Load WikiText-2 via HuggingFace datasets (preferred method)."""
    from datasets import load_dataset
    print("  Source: HuggingFace datasets")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    return {
        "train": [ex["text"] for ex in ds["train"]],
        "val":   [ex["text"] for ex in ds["validation"]],
        "test":  [ex["text"] for ex in ds["test"]],
    }


def load_wikitext2_github():
    """Load WikiText-2 via GitHub raw text mirror.

    Tokenizes each .txt by blank lines into documents,
    matching the HuggingFace document split.
    """
    print("  Source: GitHub raw text mirror (PyTorch examples repo)")

    raw_texts = {}
    for split_name in ["train", "val", "test"]:
        url = WIKITEXT2_URLS[split_name]
        text = download_via_github(url)

        # The raw .txt files have blank-line-separated paragraphs,
        # each wrapped with <s> / </s> markers for article ends.
        # Split by blank lines and treat each non-empty block as a document.
        paragraphs = text.split("\n\n")
        documents = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Remove the leading paragraph markers if present
            # (the raw format uses = Section = etc.)
            documents.append(para)

        raw_texts[split_name] = documents
        print(f"      {split_name}: {len(documents)} documents")

    return raw_texts


def tokenize_splits(raw_splits: dict, enc, data_dir: str):
    """Tokenize all splits and save as .bin files."""
    vocab_size = enc.n_vocab
    eot_token = enc.eot_token

    meta = {
        "vocab_size": vocab_size,
        "itos": {i: enc.decode([i]) for i in range(vocab_size)},
        "stoi": {},
    }

    for split_name in ["train", "val", "test"]:
        documents = raw_splits[split_name]
        print(f"  Tokenizing {split_name} split ({len(documents)} documents)...")

        all_tokens = []
        total_chars = 0

        for i, text in enumerate(documents):
            if text.strip():
                tokens = enc.encode_ordinary(text)
                all_tokens.extend(tokens)
                all_tokens.append(eot_token)
                total_chars += len(text)

            if (i + 1) % 2000 == 0:
                print(f"    processed {i + 1}/{len(documents)} documents...")

        tokens_array = np.array(all_tokens, dtype=np.uint16)
        bin_path = os.path.join(data_dir, f"{split_name}.bin")
        tokens_array.tofile(bin_path)

        print(f"  Saved {split_name}.bin: {len(tokens_array):,} tokens from "
              f"{total_chars:,} chars "
              f"({len(tokens_array) / max(total_chars, 1):.2f} tokens/char)")

    # Save metadata
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl to {meta_path}")


def prepare_wikitext2(data_dir: str):
    """Main entry: download and prepare WikiText-2."""
    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50257

    print(f"Tokenizer: GPT-2 BPE (vocab_size={vocab_size})")
    print("Loading WikiText-2 dataset...")

    # Try HuggingFace first, fall back to GitHub
    raw_splits = None
    errors = []

    # Method 1: HuggingFace
    try:
        raw_splits = load_wikitext2_huggingface()
    except Exception as e:
        errors.append(f"HuggingFace: {e}")
        print(f"  HuggingFace failed: {e}")

    # Method 2: GitHub raw text mirror
    if raw_splits is None:
        try:
            raw_splits = load_wikitext2_github()
        except Exception as e:
            errors.append(f"GitHub: {e}")
            print(f"  GitHub mirror also failed: {e}")

    if raw_splits is None:
        print("\nERROR: Could not load WikiText-2 from any source.")
        print("  Errors:", errors)
        print("\n  Manual workaround:")
        print("  1. Download https://raw.githubusercontent.com/pytorch/examples/"
              "master/word_language_model/data/wikitext-2/train.txt")
        print("  2. Save to data/wikitext2/train.txt")
        print("  3. Repeat for valid.txt and test.txt")
        print("  4. Re-run with --local_txt")
        sys.exit(1)

    # Tokenize and save
    tokenize_splits(raw_splits, enc, data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    for split_name in ["train", "val", "test"]:
        bin_path = os.path.join(data_dir, f"{split_name}.bin")
        size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        print(f"  {split_name}: {len(tokens):,} tokens ({size_mb:.2f} MB)")
    print(f"  vocab_size: {vocab_size}")
    print("=" * 60)


if __name__ == "__main__":
    # Save to data/wikitext2/ relative to project root
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data", "wikitext2")
    prepare_wikitext2(data_dir)
