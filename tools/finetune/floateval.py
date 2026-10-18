"""Float sizes of the Route A evaluation windows: the reference for the integer loss.

Run from tools/:
    nice uv run python -m finetune.floateval --model M.safetensors --windows W.txt --out F.json

The tool runs the float model of tools/convert/model.py (float32, full sequence, np.tanh GELU:
the model that the converter calibrates) on each window, and writes for each window the ideal
size `-log2 p` in bits and the byte count. Windows are independent, so `--workers` processes
share them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from baseline.bpe import EOT_ID, Tokenizer
from convert.model import forward, load_gpt2, log2_probs_of

_STATE: dict = {}


def _init(model: str) -> None:
    _STATE["m"] = load_gpt2(Path(model))
    _STATE["tok"] = Tokenizer()


def _one(item: tuple[str, list[int]]) -> dict:
    label, toks = item
    t = np.array(toks, dtype=np.int64)
    x = np.concatenate([[EOT_ID], t[:-1]])
    logits = forward(_STATE["m"], x)
    bits = float(log2_probs_of(logits, t).sum())
    return {"label": label, "tokens": len(toks), "bytes": len(_STATE["tok"].decode(toks)),
            "ideal_bits": bits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    items = []
    for line in args.windows.read_text().splitlines():
        if line and not line.startswith("#"):
            label, ids = line.split("\t", 1)
            items.append((label, [int(v) for v in ids.split()]))
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.model,)) as pool:
        for i, r in enumerate(pool.imap(_one, items, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(items)} windows ({time.time() - t0:.0f} s)", flush=True)
    args.out.write_text(json.dumps(rows, indent=1))
    b = sum(r["bytes"] for r in rows)
    print(f"{args.model}: {sum(r['ideal_bits'] for r in rows) / b:.4f} bits per byte (ideal, "
          f"float32) on {len(rows)} windows; wrote {args.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
