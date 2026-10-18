"""Pack the training split into one file for the GPU machine (the input of finetune/train.py).

Run from tools/:
    nice uv run python -m finetune.pack --work DIR --out DATA.npz

The file holds `<system>.train` and `<system>.val` from DIR/tok (uint16 token ids) and
`token_bytes` (the byte length of each GPT-2 token id, for bits per byte). It holds no
evaluation data: no tail and no 2k sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from baseline._paths import sha256_file
from baseline.bpe import Tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tok = Tokenizer()
    arrays = {"token_bytes": np.array([len(b) for b in tok.token_bytes], dtype=np.uint8)}
    for p in sorted((args.work / "tok").glob("*.npy")):
        name = p.stem  # "<system>.train" or "<system>.val"
        arrays[name] = np.load(p)
    np.savez_compressed(args.out, **arrays)
    n = sum(len(v) for k, v in arrays.items() if k.endswith(".train"))
    print(f"wrote {args.out}: {args.out.stat().st_size:,} bytes, {n:,} train tokens, "
          f"sha256 {sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
