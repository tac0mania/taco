"""Measure recipes with the integer simulation (`convert.intsim`).

Run from tools/:
    nice uv run python -m convert.tune [--set calib|eval] RECIPE[:option,...] ...

For another GPT-2-architecture model (Route A), give its weights, its statistics, a window file
(label, tab, token ids) and the float sizes of these windows (`finetune.floateval`). Use
calibration or validation windows only when the result selects a recipe:
    nice uv run python -m convert.tune --model M.safetensors --stats S.toml --windows W.txt \
        --float F.json [--workers N] SPEC ...

Options: `nobeta` (no beta correction), `kind=statistic` (the statistic for one site kind, for
example `gelu=q9999`), `site=value` (a range for one site, for example `h.0.gelu=12.3`), and
`i16=gelu+ln_f` (i16 outputs at these site kinds, hsm version 2), and `frac=N` (LayerNorm beta
with N more bits, hsm version 2). The tool prints bits per
byte (ideal size) for each recipe and its change against the float model on the same windows.
`intsim` agrees with HardShell to 1e-6 bits on a full window (measured; research note
2026-10-18-integer-gpt2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from baseline._paths import MODEL_DIR, OUT_DIR
from baseline.bpe import Tokenizer
from convert import intsim
from convert import recipe as rc
from convert.build import build
from convert.calibrate import STATS_PATH, read_windows
from convert.gpt2 import vocabulary_and_merges
from convert.model import load_gpt2
from convert.windows import calibration_windows, eval_windows


def parse(spec: str, stats: dict) -> tuple[dict[str, str], bool, dict[str, Fraction], frozenset]:
    """(recipe, beta correction, site overrides, i16 kinds) of `RECIPE[:option,...]`.

    The option `frac=N` (the LayerNorm beta bits) is in `parse_frac`.
    """
    name, _, opts = spec.partition(":")
    r = dict(rc.RECIPES[name])
    wide = rc.WIDE.get(name, frozenset())
    beta = True
    over: dict[str, Fraction] = {}
    for o in filter(None, opts.split(",")):
        if o == "nobeta":
            beta = False
            continue
        k, v = o.split("=")
        if k == "i16":
            wide = frozenset(v.split("+"))
        elif k == "frac":
            continue
        elif k in r:
            r[k] = v
        elif k in stats["site"]:
            over[k] = Fraction(v)
        else:
            raise ValueError(f"unknown option {o}")
    return r, beta, over, wide


def parse_frac(spec: str) -> int:
    """The LayerNorm beta bits of `RECIPE[:option,...]`: the option `frac=N`, else the recipe's."""
    name, _, opts = spec.partition(":")
    frac = rc.BETA_FRAC.get(name, 0)
    for o in filter(None, opts.split(",")):
        k, _, v = o.partition("=")
        if k == "frac":
            frac = int(v)
    return frac


_W: dict = {}


def _init(im: intsim.IntModel) -> None:
    _W["im"] = im


def _one(w) -> tuple[float, int]:
    return intsim.bits_of(intsim.forward(_W["im"], w.model_input), np.array(w.tokens))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="calib", choices=["calib", "eval"])
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--float", type=Path, default=None, help="float sizes (finetune.floateval)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("specs", nargs="*", default=["max"])
    args = ap.parse_args()
    stats = rc.load_stats(args.stats)
    tok = Tokenizer()
    if args.windows is not None:
        windows = read_windows(args.windows, tok)
    else:
        windows = calibration_windows(tok) if args.set == "calib" else eval_windows(tok)
    fpath = args.float or OUT_DIR / f"{args.set}_float.json"
    flt = {r["label"]: r for r in json.loads(fpath.read_text())}
    fbits = sum(flt[w.label]["ideal_bits"] for w in windows)
    nbytes = sum(len(w.data) for w in windows)
    print(f"float: {fbits / nbytes:.4f} bpb ({len(windows)} windows)", flush=True)
    vocab, merges = vocabulary_and_merges(tok)
    fm = load_gpt2(args.model or MODEL_DIR / "model.safetensors")
    for spec in args.specs:
        t0 = time.time()
        r, beta, over, wide = parse(spec, stats)
        im = intsim.IntModel(build(fm, rc.scales(stats, r, over, wide), vocab, merges,
                                   beta_correction=beta, ln_beta_frac=parse_frac(spec)))
        if args.workers > 1:
            import multiprocessing as mp

            with mp.get_context("fork").Pool(args.workers, _init, (im,)) as pool:
                res = pool.map(_one, windows, chunksize=1)
        else:
            _init(im)
            res = [_one(w) for w in windows]
        bits = sum(b for b, _ in res)
        zeros = sum(z for _, z in res)
        per = [b / len(w.data) for (b, _), w in zip(res, windows, strict=True)]
        print(f"{spec:50s} {bits / nbytes:.4f} bpb {100 * (bits / fbits - 1):+6.2f} % "
              f"zero-weight {zeros} ({time.time() - t0:.0f} s)  "
              + " ".join(f"{x:.2f}" for x in per), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
