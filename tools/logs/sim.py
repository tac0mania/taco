"""The mixer simulation on the facts of `logs.gpt2pass`: bits per byte for each block size.

Run from tools/ (after the pass outputs are in tools/data/out/logs/pass):
    uv run python -m logs.sim ceiling [--files ...]     # the table for each block size
    uv run python -m logs.sim tune [--files ...]        # a small grid of constants
    uv run python -m logs.sim final                     # the exact interval, final constants

The match model and the mixer are the integer rules of `logs.matchmodel` and `logs.mixer`
(the mirror of `crates/taco-match`). The GPT-2 facts are exact integer weights. For each token:

- with no prediction, the cost is the exact plan §5.1 table of GPT-2 alone (`fa`);
- with a prediction, `ceiling` and `tune` use the probability of the mixed weights,
  `q / 2^16` for the predicted token and `(m / 2^16) * w[a] / R` for another token, with the
  1-count floor of plan §5.1. `final` computes the exact table bounds (`exact_bits`).

NOTE: GPT-2 restarts its context at each 16 KiB block start and after each 1,024 tokens, for
all block sizes (the facts come from one pass). A larger block has fewer restarts in Taco, so
the GPT-2 numbers of the larger blocks are a little pessimistic (est.).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass

import numpy as np

from logs import corpus
from logs import matchmodel as mm
from logs import mixer as mx

PASS_DIR = corpus.OUT_DIR / "pass"
TOTAL = 1 << 32
V = 50257
SPARE = TOTAL - V
UNITS = {16384: 1, 65536: 4, 262144: 16, 1048576: 64}
DEFAULT_MATCH = mm.MatchParams((32, 16, 8, 4, 2, 1))
DEFAULT_MIXER = mx.MixerParams()


@dataclass
class Facts:
    name: str
    nbytes: int
    ids: np.ndarray
    block_starts: np.ndarray
    W: np.ndarray
    wa: np.ndarray
    fa: np.ndarray
    top_id: np.ndarray
    top_w: np.ndarray
    cand: np.ndarray
    wc: np.ndarray


def load(name: str) -> Facts:
    z = np.load(PASS_DIR / f"{name}.npz")
    return Facts(name, int(z["nbytes"]), z["ids"], z["block_starts"], z["W"], z["wa"], z["fa"],
                 z["top_id"], z["top_w"], z["cand"], z["wc"])


def pred_weight(f: Facts, pred: np.ndarray) -> np.ndarray:
    """The GPT-2 weight of each predicted token, from the candidate columns."""
    hit = f.cand == pred[:, None]
    has = pred >= 0
    if not np.all(hit.any(axis=1) | ~has):
        missing = int((~hit.any(axis=1) & has).sum())
        raise ValueError(f"{f.name}: {missing} predictions are not in the candidate set")
    return np.where(has, (f.wc * hit).sum(axis=1, dtype=np.uint64), 0)


def run_mixer(f: Facts, pred: np.ndarray, plen: np.ndarray, wp: np.ndarray, unit: int,
              mp: mx.MixerParams) -> np.ndarray:
    """q for each predicted position (0 elsewhere). The mixer resets at each block start."""
    n = len(f.ids)
    q = np.zeros(n, dtype=np.int64)
    m = mx.Mixer(mp)
    starts = set(f.block_starts[::unit].tolist())
    ids, pl, pr, wpl, W = f.ids.tolist(), plen.tolist(), pred.tolist(), wp.tolist(), f.W.tolist()
    top = f.top_id[:, 0].tolist()
    for i in range(n):
        if i in starts:
            m.reset()
        if pr[i] < 0:
            continue
        qi = m.prob(int(wpl[i]), int(W[i]), pl[i], pr[i] == top[i])
        q[i] = qi
        m.update(pr[i] == ids[i])
    return q


def approx_bits(f: Facts, pred: np.ndarray, wp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The cost of each token in bits (see the module documentation)."""
    gpt = -np.log2(f.fa.astype(np.float64) / TOTAL)
    has = pred >= 0
    hit = has & (pred == f.ids)
    r = np.maximum(f.W.astype(np.float64) - wp.astype(np.float64), 1.0)
    pa = np.where(hit, q / 65536.0, (65536.0 - q) / 65536.0 * f.wa.astype(np.float64) / r)
    pa = np.maximum(pa, 1.0 / TOTAL)
    return np.where(has, -np.log2(pa), gpt)


def exact_bits(f: Facts, pred: np.ndarray, wp: np.ndarray, q: np.ndarray
               ) -> tuple[float, float]:
    """Lower and upper bounds of the exact table cost (plan §5.1 on the mixed weights).

    The pass does not keep the full rows, so two terms have bounds, not values: the sum of
    the other weights after the shift and the 1-floor (an uncertainty of at most V - 1 units),
    and the remainder of plan §5.1 step 3 (0 to V - 1 counts, only for the largest weight).
    """
    lo = hi = 0.0
    ids, pr, W, wa, wpl, ql = (f.ids.tolist(), pred.tolist(), f.W.tolist(), f.wa.tolist(),
                               wp.tolist(), q.tolist())
    fa, tid, tw = f.fa.tolist(), f.top_id.tolist(), f.top_w.tolist()
    for i in range(len(ids)):
        if pr[i] < 0:
            b = -math.log2(fa[i] / TOTAL)
            lo += b
            hi += b
            continue
        p, a, qi = pr[i], ids[i], ql[i]
        wsum, wpi = int(W[i]), int(wpl[i])
        rr = wsum - wpi
        r = max(rr, 1)
        m = 65536 - qi
        k = max(0, r.bit_length() + 16 - 62)
        wpp = max(1, (r * qi) >> k)
        base = (rr * m) >> k
        o_lo = base - (V - 1) if k > 0 else base
        o_hi = base + (V - 1)
        w_lo, w_hi = max(o_lo, 0) + wpp, o_hi + wpp
        wa2 = wpp if a == p else max(1, (int(wa[i]) * m) >> k)
        f_lo = 1 + wa2 * SPARE // w_hi
        f_hi = 1 + wa2 * SPARE // w_lo
        # The largest weight: p, or the first top GPT-2 token that is not p.
        t = tid[i][0] if tid[i][0] != p else tid[i][1]
        twv = tw[i][0] if tid[i][0] != p else tw[i][1]
        wt = max(1, (int(twv) * m) >> k)
        top = p if (wpp > wt or (wpp == wt and p < t)) else t
        certain = k == 0 or top == p
        if a == top or not certain and wa2 >= wt:
            f_hi += V - 1
        lo += -math.log2(min(f_hi, TOTAL) / TOTAL)
        hi += -math.log2(f_lo / TOTAL)
    return lo, hi


def evaluate(f: Facts, bs: int, mp_match: mm.MatchParams, mp_mix: mx.MixerParams,
             exact: bool = False) -> dict:
    unit = UNITS[bs]
    pred, plen = mm.predict_file(f.ids, f.block_starts, unit, mp_match)
    wp = pred_weight(f, pred)
    q = run_mixer(f, pred, plen, wp, unit, mp_mix)
    bits = approx_bits(f, pred, wp, q)
    gpt = -np.log2(f.fa.astype(np.float64) / TOTAL)
    out = {"name": f.name, "bytes": f.nbytes, "block": bs, "tokens": len(f.ids),
           "gpt2_bits": float(gpt.sum()), "mixed_bits": float(bits.sum()),
           "predicted": int((pred >= 0).sum()), "hits": int(((pred >= 0) & (pred == f.ids)).sum())}
    if exact:
        lo, hi = exact_bits(f, pred, wp, q)
        out["exact_lo"], out["exact_hi"] = lo, hi
    return out


def available(names: list[str] | None) -> list[str]:
    all_names = list(corpus.GPT2_LOGS) + list(corpus.TEXT_FILES)
    have = [n for n in all_names if (PASS_DIR / f"{n}.npz").exists()]
    return [n for n in have if not names or n in names]


def cmd_ceiling(names: list[str], mpm: mm.MatchParams, mpx: mx.MixerParams, exact: bool) -> None:
    rows = []
    for name in names:
        f = load(name)
        for bs in UNITS:
            r = evaluate(f, bs, mpm, mpx, exact)
            rows.append(r)
            ex = (f"  exact [{r['exact_lo'] / r['bytes']:.4f}, {r['exact_hi'] / r['bytes']:.4f}]"
                  if exact else "")
            print(f"{name:18s} {bs // 1024:5d} KiB  gpt2 {r['gpt2_bits'] / r['bytes']:.4f}  "
                  f"mixed {r['mixed_bits'] / r['bytes']:.4f} bpb  "
                  f"({100 * (r['mixed_bits'] / r['gpt2_bits'] - 1):+.1f} %){ex}", flush=True)
    tag = "final" if exact else "ceiling"
    (corpus.OUT_DIR / f"{tag}_{mpm.key}_{mpx.key}.json").write_text(json.dumps(rows, indent=1))


def cmd_tune(names: list[str]) -> None:
    facts = [load(n) for n in names]
    grid_match = [mm.MatchParams((32, 16, 8, 4)), mm.MatchParams((16, 8, 4, 2, 1)),
                  mm.MatchParams((32, 16, 8, 4, 2, 1)), mm.MatchParams((24, 12, 6, 3, 2))]
    grid_mix = [mx.MixerParams(lr=lr, hit_limit=lim, wm_init=wm)
                for lr in (6, 12, 24, 48) for lim in (63, 255, 1023) for wm in (0, 1 << 15)]
    for bs in (16384, 1048576):
        for mpm in grid_match:
            best = None
            for mpx in grid_mix:
                tot = sum(evaluate(f, bs, mpm, mpx)["mixed_bits"] for f in facts)
                nb = sum(f.nbytes for f in facts)
                print(f"{bs // 1024:5d} KiB {mpm.key:22s} {mpx.key:22s} {tot / nb:.4f} bpb",
                      flush=True)
                if best is None or tot < best[0]:
                    best = (tot, mpx.key)
            print(f"best {bs // 1024} KiB {mpm.key}: {best}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["ceiling", "tune", "final"])
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()
    names = available(args.files)
    print(f"files: {names}", flush=True)
    if args.cmd == "tune":
        cmd_tune(names)
    else:
        cmd_ceiling(names, DEFAULT_MATCH, DEFAULT_MIXER, args.cmd == "final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
