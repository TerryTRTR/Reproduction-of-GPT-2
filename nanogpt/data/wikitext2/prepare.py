"""Compatibility wrapper for the shared WikiText-2 prepare script.

The project-standard WikiText-2 data now lives at the repository root:

    data/wikitext2

Run this file only for backward compatibility with old nanoGPT instructions.
It delegates to the shared prepare script and writes outputs to the shared
directory, not to nanogpt/data/wikitext2.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT / "data" / "wikitext2" / "prepare.py"), run_name="__main__")
