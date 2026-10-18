"""Fake quantisation: the float GPT-2 with the quantisation steps of HardShell, one site kind
at a time. This tool finds which part of the integer model loses the most.

Run from tools/:
    nice uv run python -m convert.fq [--recipe NAME] [--set calib|eval] [--modes ...]

Each mode is a set of quantised parts:

- `float`: no quantisation.
- `w`: the i8 weights with one scale for each row (wte, wpe and the linear layers).
- `w+<kind>`: the weights and one site kind (`ln_1`, `q`, `k`, `v`, `att`, `ln_2`, `fc`,
  `gelu`, `ln_f`, `res`).
- `all`: the weights and every site. This mode is near the HardShell model. The differences
  are the integer LayerNorm, the softmax table and the logit table, which ADR 0004 measured
  as small, and the rounding of the biases.

A site value becomes `clamp(round(x / s), -q, q) * s`, with the scale `s` of the recipe and
`q = 127` (or 32,767 for the i16 GELU input). The GELU output uses the quantised GELU input.
The result is the ideal size `-log2 p` (float softmax) in bits per byte.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from baseline._paths import MODEL_DIR, OUT_DIR
from baseline.bpe import Tokenizer
from convert import numerics as nm
from convert import recipe as rc
from convert.build import Scales
from convert.calibrate import STATS_PATH
from convert.model import FModel, forward, load_gpt2, log2_probs_of, with_weights
from convert.windows import calibration_windows, eval_windows

KINDS = (*rc.KINDS, "ln_f", "res")


def dequantised(fm: FModel) -> FModel:
    def f(w: np.ndarray) -> np.ndarray:
        q, s = nm.quantise_rows(w)
        return (q.astype(np.float64) * s[:, None]).astype(w.dtype)

    return with_weights(fm, f)


def kind_of(site: str) -> str:
    if site in ("ln_f", "res"):
        return site
    return site.split(".", 2)[2]


def make_hook(sc: Scales, kinds: set[str]):
    def hook(site: str, x: np.ndarray) -> np.ndarray | None:
        k = kind_of(site)
        if k not in kinds:
            return None
        if site == "res":
            q = np.clip(np.rint(x / sc.res), -nm.RESIDUAL_LIMIT, nm.RESIDUAL_LIMIT)
            return (q * sc.res).astype(x.dtype)
        s = sc.site[site]
        qmax = nm.I16_MAX if k == "fc" else nm.I8_MAX
        return (np.clip(np.rint(x / s), -qmax, qmax) * s).astype(x.dtype)

    return hook


def run(fm: FModel, windows, hook) -> tuple[float, int, list[float]]:
    bits, nbytes, per = 0.0, 0, []
    for w in windows:
        lg = forward(fm, w.model_input, hook=hook)
        b = float(log2_probs_of(lg, np.array(w.tokens)).sum())
        per.append(b)
        bits += b
        nbytes += len(w.data)
    return bits, nbytes, per


def parse_mode(mode: str) -> tuple[bool, set[str]]:
    if mode == "float":
        return False, set()
    if mode == "all":
        return True, set(KINDS)
    parts = mode.split("+")
    if parts[0] != "w":
        raise ValueError(f"unknown mode {mode}")
    kinds = set(parts[1:])
    if not kinds <= set(KINDS):
        raise ValueError(f"unknown kind in {mode}")
    return True, kinds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="max", choices=sorted(rc.RECIPES))
    ap.add_argument("--set", default="calib", choices=["calib", "eval"])
    ap.add_argument("--limit", type=int, default=0, help="use the first N windows only")
    ap.add_argument("--modes", nargs="+",
                    default=["float", "w", *[f"w+{k}" for k in KINDS], "all"])
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    stats = rc.load_stats(STATS_PATH)
    sc = rc.scales(stats, rc.RECIPES[args.recipe])
    tok = Tokenizer()
    windows = calibration_windows(tok) if args.set == "calib" else eval_windows(tok)
    if args.limit:
        windows = windows[: args.limit]
    fm = load_gpt2(MODEL_DIR / "model.safetensors")
    fq = None
    results = {}
    base = None
    for mode in args.modes:
        t0 = time.time()
        wq, kinds = parse_mode(mode)
        if wq and fq is None:
            fq = dequantised(fm)
        bits, nbytes, per = run(fq if wq else fm, windows, make_hook(sc, kinds))
        if base is None:
            base = bits
        results[mode] = {"bits": bits, "bytes": nbytes, "per_window": per}
        print(f"{mode:10s} {bits / nbytes:.4f} bpb  {100 * (bits / base - 1):+7.2f} %  "
              f"({time.time() - t0:.0f} s)", flush=True)
    if args.json:
        (OUT_DIR / args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
