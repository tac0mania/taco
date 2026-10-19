"""Fake quantisation: the float model with the quantisation steps of HardShell, one site kind
at a time. This tool finds which part of the integer model loses the most.

Run from tools/:
    nice uv run python -m convert.fq [--recipe NAME] [--set calib|eval] [--modes ...]

For another GPT-2-architecture model (Route A), give its weights, its statistics and a window
file (label, tab, token ids). Use calibration or validation windows only, never the
evaluation windows, when the result selects a recipe:
    nice uv run python -m convert.fq --model M.safetensors --stats S.toml --windows W.txt

Each mode is a set of quantised parts:

- `float`: no quantisation.
- `w`: the i8 weights with one scale for each row (wte, wpe and the linear layers).
- `wte`, `blk`: only the tied embedding and logit matrix `wte` (and `wpe`), or only the
  matrices of the blocks.
- `w+<part>+...`: the weights and the named parts. A part is a site kind (`ln_1`, `q`, `k`,
  `v`, `att`, `ln_2`, `fc`, `gelu`, `ln_f`, `res`), a site name (for example `h.5.gelu`), or
  `logit` (the logit unit 2^-8 and the `logits_exp` table of ADR 0004 section 5).
- `all`: the weights, every site kind and `logit`. This mode is near the HardShell model. The
  differences are the integer LayerNorm (and its beta rounding), the softmax table and the
  rounding of the biases, which ADR 0004 measured as small.
- `all-<part>-...`: `all` without the named parts (leave one out).

A site value becomes `clamp(round(x / s), -q, q) * s`, with the scale `s` of the recipe and
`q = 127` (or 32,767 for the i16 GELU input). The GELU output uses the quantised GELU input.
The result is the ideal size `-log2 p` in bits per byte.

`--rule kind=statistic` and `--range site=value` change the recipe as in `convert.tune`.

NOTE: a measurement tool with floats, not the prediction path (CLAUDE.md §1 rule 1).
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
from convert import numerics as nm
from convert import recipe as rc
from convert.build import Scales
from convert.calibrate import STATS_PATH, read_windows
from convert.model import FModel, forward, load_gpt2, log2_probs_of
from convert.windows import calibration_windows, eval_windows

KINDS = (*rc.KINDS, "ln_f", "res")
ALL_PARTS = (*KINDS, "logit")


def _deq(w: np.ndarray) -> np.ndarray:
    q, s = nm.quantise_rows(w)
    return (q.astype(np.float64) * s[:, None]).astype(w.dtype)


def dequantised(fm: FModel, emb: bool = True, blocks: bool = True) -> FModel:
    """A copy of `fm` with i8 weights: `emb` selects wte and wpe, `blocks` the linear layers."""
    from dataclasses import replace

    bl = [
        replace(b, attn_w=_deq(b.attn_w), proj_w=_deq(b.proj_w), fc_w=_deq(b.fc_w),
                mlp_w=_deq(b.mlp_w)) if blocks else b
        for b in fm.blocks
    ]
    if emb:
        return replace(fm, wte=_deq(fm.wte), wpe=_deq(fm.wpe), blocks=bl)
    return replace(fm, blocks=bl)


def kind_of(site: str) -> str:
    if site in ("ln_f", "res"):
        return site
    return site.split(".", 2)[2]


def make_hook(sc: Scales, parts: set[str]):
    res, site_scale = sc.floats()  # a measurement tool: float scales are enough here

    def hook(site: str, x: np.ndarray) -> np.ndarray | None:
        k = kind_of(site)
        if k not in parts and site not in parts:
            return None
        if site == "res":
            q = np.clip(np.rint(x / res), -nm.RESIDUAL_LIMIT, nm.RESIDUAL_LIMIT)
            return (q * res).astype(x.dtype)
        s = site_scale[site]
        qmax = nm.I16_MAX if k == "fc" else nm.I8_MAX
        return (np.clip(np.rint(x / s), -qmax, qmax) * s).astype(x.dtype)

    return hook


LOGIT_TABLE = nm.logits_exp_table().astype(np.float64)


def int_logit_bits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """-log2 p with the integer logit rule: units of 2^-8, then the `logits_exp` table."""
    z = np.rint(logits.astype(np.float64) * (1 << -nm.DELTA_LOG2))
    gap = np.clip(z.max(axis=1, keepdims=True) - z, 0, len(LOGIT_TABLE) - 1).astype(np.int64)
    w = LOGIT_TABLE[gap]
    return -np.log2(w[np.arange(len(targets)), targets] / w.sum(axis=1))


def run(fm: FModel, windows, hook, int_logits: bool) -> tuple[float, int, list[float]]:
    bits, nbytes, per = 0.0, 0, []
    for w in windows:
        lg = forward(fm, w.model_input, hook=hook)
        t = np.array(w.tokens)
        b = float((int_logit_bits(lg, t) if int_logits else log2_probs_of(lg, t)).sum())
        per.append(b)
        bits += b
        nbytes += len(w.data)
    return bits, nbytes, per


_W: dict = {}


def _init(model: Path, sc: Scales) -> None:
    _W["fm"] = load_gpt2(model)
    _W["sc"] = sc
    _W["cache"] = {(False, False): _W["fm"]}


def _window(item: tuple[list[str], object]) -> list[float]:
    """All modes on one window (a worker process keeps the quantised models)."""
    modes, w = item
    out = []
    for mode in modes:
        wq, parts = parse_mode(mode)
        if wq not in _W["cache"]:
            _W["cache"][wq] = dequantised(_W["fm"], *wq)
        out.append(run(_W["cache"][wq], [w], make_hook(_W["sc"], parts), "logit" in parts)[0])
    return out


def parse_mode(mode: str) -> tuple[tuple[bool, bool], set[str]]:
    """((quantise wte and wpe, quantise the block matrices), parts)."""
    if mode == "float":
        return (False, False), set()
    if mode == "all" or mode.startswith("all-"):
        drop = set(mode.split("-")[1:])
        if not drop <= set(ALL_PARTS):
            raise ValueError(f"unknown part in {mode}")
        return (True, True), set(ALL_PARTS) - drop
    head, *parts = mode.split("+")
    wq = {"w": (True, True), "wte": (True, False), "blk": (False, True)}.get(head)
    if wq is None:
        raise ValueError(f"unknown mode {mode}")
    for p in parts:
        if p not in ALL_PARTS and not p.startswith("h."):
            raise ValueError(f"unknown part {p} in {mode}")
    return wq, set(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="max", choices=sorted(rc.RECIPES))
    ap.add_argument("--rule", action="append", default=[],
                    help="kind=statistic, for example gelu=chan_q99")
    ap.add_argument("--range", action="append", default=[],
                    help="site=value (a float range), for example ln_f=24")
    ap.add_argument("--set", default="calib", choices=["calib", "eval"])
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--windows", type=Path, default=None,
                    help="a window file (label, tab, ids) instead of --set")
    ap.add_argument("--limit", type=int, default=0, help="use the first N windows only")
    ap.add_argument("--modes", nargs="+",
                    default=["float", "w", *[f"w+{k}" for k in ALL_PARTS], "all"])
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1, help="processes, one window each")
    args = ap.parse_args()
    stats = rc.load_stats(args.stats)
    recipe = dict(rc.RECIPES[args.recipe])
    for r in args.rule:
        k, v = r.split("=")
        recipe[k] = v
    over = {}
    for r in args.range:
        k, v = r.split("=")
        over[k] = Fraction(v)
    sc = rc.scales(stats, recipe, over)
    tok = Tokenizer()
    if args.windows is not None:
        windows = read_windows(args.windows, tok)
    else:
        windows = calibration_windows(tok) if args.set == "calib" else eval_windows(tok)
    if args.limit:
        windows = windows[: args.limit]
    model = args.model or MODEL_DIR / "model.safetensors"
    for mode in args.modes:
        parse_mode(mode)
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers, initializer=_init, initargs=(model, sc)) as pool:
            rows = []
            for i, r in enumerate(pool.imap(_window, [(args.modes, w) for w in windows])):
                rows.append(r)
                print(f"  window {i + 1}/{len(windows)} ({time.time() - t0:.0f} s)", flush=True)
    else:
        _init(model, sc)
        rows = [_window((args.modes, w)) for w in windows]
    nbytes = sum(len(w.data) for w in windows)
    results = {"labels": [w.label for w in windows],
               "bytes": [len(w.data) for w in windows], "modes": {}}
    base = None
    for j, mode in enumerate(args.modes):
        per = [r[j] for r in rows]
        bits = sum(per)
        if base is None:
            base = bits
        results["modes"][mode] = {"bits": bits, "bytes": nbytes, "per_window": per}
        print(f"{mode:24s} {bits / nbytes:.4f} bpb  {100 * (bits / base - 1):+7.2f} %", flush=True)
    if args.json:
        path = args.json if args.json.is_absolute() else OUT_DIR / args.json
        path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
