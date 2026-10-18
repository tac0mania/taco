"""The evaluation windows and the float GPT-2 reference sizes (plan §12, phase 3).

Run from tools/:
    nice uv run python -m convert.evalset [--set eval|calib]

`--set eval` (the default) takes the first window of 1,024 tokens of each of the 19 phase 0
corpus files: 19,456 steps, the phase 1 part A set. `--set calib` takes the calibration
windows. The tool writes:

- tools/data/out/<set>_windows.txt: one window on each line, `label<TAB>ids`, the input of
  the HardShell example `eval_bits`;
- tools/data/out/<set>_float.json: the float GPT-2 sizes of each window (float32, full
  sequence, as phase 0): the ideal size `-log2 p` and the size with the plan §5.1 tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from baseline._paths import MODEL_DIR, OUT_DIR
from baseline.bpe import Tokenizer
from baseline.gpt2_np import GPT2
from baseline.rangecoder import TOTAL, tables_from_logits
from convert.windows import calibration_windows, eval_windows


def float_sizes(model: GPT2, w) -> dict:
    logits = model.forward(w.model_input)
    t = np.array(w.tokens)
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    ideal = float(((lse - z[np.arange(len(t)), t]) / np.log(2.0)).sum())
    table = 0.0
    for i in range(0, len(t), 64):
        f, _ = tables_from_logits(logits[i : i + 64])
        ft = f[np.arange(len(f)), t[i : i + 64]].astype(np.float64)
        table += float(-np.log2(ft / TOTAL).sum())
    return {"label": w.label, "tokens": len(w.tokens), "bytes": len(w.data),
            "ideal_bits": ideal, "table_bits": table}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="eval", choices=["eval", "calib"])
    args = ap.parse_args()
    tok = Tokenizer()
    windows = eval_windows(tok) if args.set == "eval" else calibration_windows(tok)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.set}_windows.txt"
    with path.open("w") as f:
        f.write(f"# {args.set} windows: label, tab, token ids (convert.evalset)\n")
        for w in windows:
            f.write(w.label + "\t" + " ".join(map(str, w.tokens)) + "\n")
    print(f"wrote {path}: {len(windows)} windows, {sum(len(w.tokens) for w in windows)} tokens")
    model = GPT2(np.float32, MODEL_DIR)
    rows = []
    for w in windows:
        t0 = time.time()
        rows.append(float_sizes(model, w))
        r = rows[-1]
        print(f"  {w.label}: {r['ideal_bits'] / r['bytes']:.4f} bpb ideal, "
              f"{r['table_bits'] / r['bytes']:.4f} table ({time.time() - t0:.1f} s)", flush=True)
    out = OUT_DIR / f"{args.set}_float.json"
    out.write_text(json.dumps(rows, indent=1))
    b = sum(r["bytes"] for r in rows)
    print(f"total: {sum(r['ideal_bits'] for r in rows) / b:.4f} bpb ideal, "
          f"{sum(r['table_bits'] for r in rows) / b:.4f} bpb table; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
