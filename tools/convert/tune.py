"""Measure recipes with the integer simulation (`convert.intsim`) on the calibration windows.

Run from tools/:
    nice uv run python -m convert.tune [--set calib|eval] RECIPE[:option,...] ...

Options: `nobeta` (no beta correction), `site=key` (the statistic for one site kind, for
example `gelu=q9999`). The tool prints bits per byte (ideal size) for each recipe and its
change against the float GPT-2 on the same windows. `intsim` agrees with HardShell to 1e-6
bits on a full window (measured; research note 2026-10-18-integer-gpt2).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from baseline._paths import MODEL_DIR, OUT_DIR
from baseline.bpe import Tokenizer
from convert import intsim
from convert import recipe as rc
from convert.build import build
from convert.calibrate import STATS_PATH
from convert.gpt2 import vocabulary_and_merges
from convert.model import load_gpt2
from convert.windows import calibration_windows, eval_windows


def parse(spec: str) -> tuple[dict[str, str], bool]:
    name, _, opts = spec.partition(":")
    r = dict(rc.RECIPES[name])
    beta = True
    for o in filter(None, opts.split(",")):
        if o == "nobeta":
            beta = False
        else:
            k, v = o.split("=")
            r[k] = v
    return r, beta


def main() -> int:
    args = sys.argv[1:]
    which = "calib"
    if args[:1] == ["--set"]:
        which, args = args[1], args[2:]
    specs = args or ["max"]
    stats = rc.load_stats(STATS_PATH)
    tok = Tokenizer()
    windows = calibration_windows(tok) if which == "calib" else eval_windows(tok)
    flt = json.loads((OUT_DIR / f"{which}_float.json").read_text())
    fbits = sum(r["ideal_bits"] for r in flt)
    nbytes = sum(r["bytes"] for r in flt)
    print(f"float: {fbits / nbytes:.4f} bpb ({len(windows)} windows)", flush=True)
    vocab, merges = vocabulary_and_merges(tok)
    fm = load_gpt2(MODEL_DIR / "model.safetensors")
    for spec in specs:
        t0 = time.time()
        r, beta = parse(spec)
        im = intsim.IntModel(build(fm, rc.scales(stats, r), vocab, merges, beta_correction=beta))
        bits = 0.0
        zeros = 0
        per = []
        for w in windows:
            b, z = intsim.bits_of(intsim.forward(im, w.model_input), np.array(w.tokens))
            bits += b
            zeros += z
            per.append(b / len(w.data))
        print(f"{spec:50s} {bits / nbytes:.4f} bpb {100 * (bits / fbits - 1):+6.2f} % "
              f"zero-weight {zeros} ({time.time() - t0:.0f} s)  "
              + " ".join(f"{x:.2f}" for x in per), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
