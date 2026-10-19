"""The skip rule of mixer rule version 2 (ADR 0017), in Python integers, and its measurement.

Run from tools/ (after the pass outputs of `logs.gpt2pass` exist):
    uv run python -m logs.skip grid [--pass-dir DIR] [--files ...]   # skip share and cost
    uv run python -m logs.skip final [--pass-dir DIR]                 # rule version 2 only
    uv run python -m logs.skip vectors                                # golden vectors

The skip rule (mirror of `crates/taco-match`, `Mixer::skip`): at a position with a match
prediction `(p, L)` and the bucket `b = bucket(L)`, the mixer skips GPT-2 when
`b >= min_bucket`, `count[b] >= min_count` and `hit[b] >= min_hit`. All three values depend
only on earlier tokens, so the encoder and the decoder make the same decision. A skipped
position codes a flag first, in a table of two symbols with the weights `q_s` (hit) and
`2^16 - q_s` (miss), `q_s = clamp(hit[b] >> 6, 1, 65535)`. On a miss, it codes the token in the
GPT-2 table without `p` (the weight of `p` is 0). After the token, only `hit[b]` and `count[b]`
change (the weight sets need the GPT-2 input).

The measurement also counts the sequential HardShell passes that `taco unfold` needs for one
block with speculative decoding (draft length `k`) and with the skip rule (ADR 0017):

- `base`: one pass for each token.
- `spec k`: a pass at position `i` runs `k` rows: the known token and `k - 1` draft tokens (the
  match predictions along the draft). Row `r` is valid if the draft is correct at `i .. i+r-1`.
  Without a second coder query ("redecode"), row `r` gives a token only if the token is also
  the draft token; with it, every valid row gives a token.
- `skip`: a position needs weights unless it is a skipped hit. One pass for each position that
  needs weights (the skipped tokens before it run in the same pass, without outputs).
- `skip+spec k`: both.

NOTE: a measurement tool with floats (the cost in bits), not the prediction path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from logs import matchmodel as mm
from logs import mixer as mx
from logs import sim

TOTAL = 1 << 32
V = sim.V
HIT_SHIFT = mx.HIT_BITS - 16


@dataclass(frozen=True)
class SkipParams:
    """The constants of the skip rule (part of the mixer rule version)."""

    min_hit: int  # the hit probability of the context >= min_hit (22 bits)
    min_count: int  # the count of the context >= min_count (count saturates at 255)
    min_bucket: int  # b >= min_bucket
    ctx: str = "bucket"  # the context of the hit probability: "bucket", "token", "tokbkt"
    table_bits: int = 0  # 0: one entry for each context; else a hashed table of 2^bits

    @property
    def key(self) -> str:
        return (f"{self.ctx}{self.table_bits}_h{self.min_hit / (1 << 22):.4f}_n{self.min_count}"
                f"_b{self.min_bucket}")


# Mixer rule version 2: the constants of ADR 0017 (chosen from the `grid` measurement).
V2 = SkipParams(min_hit=3_774_873, min_count=2, min_bucket=0, ctx="tokbkt", table_bits=16)


def _adapt(h: int, n: int, y: bool, limit: int) -> tuple[int, int]:
    """The hit probability rule of the mixer (ADR 0010 section 3, rule 6)."""
    if y:
        h += (((1 << mx.HIT_BITS) - h) * 2) // (2 * n + 3)
    else:
        h -= (h * 2) // (2 * n + 3)
    return h, min(n + 1, limit)


class MixerV2(mx.Mixer):
    """The mixer of rule version 1 with the skip rule."""

    def __init__(self, p: mx.MixerParams, s: SkipParams) -> None:
        self.s = s
        super().__init__(p)

    def reset(self) -> None:
        super().reset()
        self.skip_b: int | None = None
        self.key: int | None = None
        self.tab: dict[int, tuple[int, int]] = {}

    def skip(self, length: int, token: int = 0) -> int | None:
        """q_s if the rule skips GPT-2 at this position; remembers the context for update."""
        self.pending = None
        self.skip_b = None
        b = mx.bucket(length)
        if self.s.ctx == "bucket":
            self.key = None
            h, n = self.hit[b], self.count[b]
        else:
            self.key = token if self.s.ctx == "token" else token * mx.N_BUCKETS + b
            if self.s.table_bits:
                x = ((self.key + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                self.key = x >> (64 - self.s.table_bits)
            h, n = self.tab.get(self.key, (1 << (mx.HIT_BITS - 1), 0))
        if b < self.s.min_bucket or n < self.s.min_count or h < self.s.min_hit:
            return None
        self.skip_b = b
        return min(max(h >> HIT_SHIFT, 1), mx.P_ONE - 1)

    def update(self, hit: bool) -> None:
        if self.key is not None:
            h, n = self.tab.get(self.key, (1 << (mx.HIT_BITS - 1), 0))
            self.tab[self.key] = _adapt(h, n, hit, self.p.hit_limit)
            self.key = None
        if self.skip_b is None:
            super().update(hit)
            return
        b = self.skip_b
        self.skip_b = None
        self.hit[b], self.count[b] = _adapt(self.hit[b], self.count[b], hit, self.p.hit_limit)


def flag_freqs(qs: int) -> tuple[int, int]:
    """The two frequencies of `FreqTable::from_weights([q_s, 2^16 - q_s], 32)` (plan §5.1)."""
    w = (qs, mx.P_ONE - qs)
    f = [1 + x * (TOTAL - 2) // mx.P_ONE for x in w]
    rem = TOTAL - sum(f)
    f[0 if w[0] >= w[1] else 1] += rem
    return f[0], f[1]


@dataclass
class Result:
    name: str
    nbytes: int
    tokens: int
    bits: float
    skipped: int
    skip_miss: int
    need: np.ndarray  # the position needs GPT-2 weights
    hit: np.ndarray  # the match prediction is correct


_PRED: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def predictions(f: sim.Facts, unit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The match predictions, lengths and predicted GPT-2 weights of a file (cached)."""
    key = (f.name, unit)
    if key not in _PRED:
        pred, plen = mm.predict_file(f.ids, f.block_starts, unit, sim.DEFAULT_MATCH)
        _PRED[key] = (pred, plen, sim.pred_weight(f, pred))
    return _PRED[key]


def simulate(f: sim.Facts, unit: int, s: SkipParams | None,
             mpx: mx.MixerParams = sim.DEFAULT_MIXER) -> Result:
    """The approximate cost (sim.approx_bits) of a file with the skip rule `s` (None: rule 1)."""
    pred, plen, wp = predictions(f, unit)
    n = len(f.ids)
    starts = set(f.block_starts[::unit].tolist())
    ids, pl, pr, wpl, W = f.ids.tolist(), plen.tolist(), pred.tolist(), wp.tolist(), f.W.tolist()
    wa, fa = f.wa.tolist(), f.fa.tolist()
    top = f.top_id[:, 0].tolist()
    m = MixerV2(mpx, s) if s is not None else mx.Mixer(mpx)
    bits = 0.0
    skipped = miss = 0
    need = np.ones(n, dtype=bool)
    hit = (pred >= 0) & (pred == f.ids)
    for i in range(n):
        if i in starts:
            m.reset()
        if pr[i] < 0:
            bits -= math.log2(fa[i] / TOTAL)
            continue
        y = pr[i] == ids[i]
        r = max(int(W[i]) - int(wpl[i]), 1)
        qs = m.skip(pl[i], pr[i]) if s is not None else None
        if qs is not None:
            skipped += 1
            f0, f1 = flag_freqs(qs)
            if y:
                bits -= math.log2(f0 / TOTAL)
                need[i] = False
            else:
                miss += 1
                fa2 = 1 + int(wa[i]) * (TOTAL - V) // r
                bits -= math.log2(f1 / TOTAL) + math.log2(fa2 / TOTAL)
            m.update(y)
            continue
        q = m.prob(int(wpl[i]), int(W[i]), pl[i], pr[i] == top[i])
        pa = q / 65536.0 if y else (65536.0 - q) / 65536.0 * float(wa[i]) / r
        bits -= math.log2(max(pa, 1.0 / TOTAL))
        m.update(y)
    return Result(f.name, f.nbytes, n, bits, skipped, miss, need, hit)


def passes(need: np.ndarray, hit: np.ndarray, starts: list[int], k: int, redecode: bool,
           skip: bool) -> tuple[int, int]:
    """(sequential passes, rows) for the protocol of the module documentation. A block start
    ends every draft (the blocks decode in separate slots)."""
    n = len(need)
    nd = need if skip else np.ones(n, dtype=bool)
    hl = hit.tolist()
    ndl = nd.tolist()
    ends = sorted(set(starts) | {n})
    total_passes = rows = 0
    for bi in range(len(ends) - 1):
        lo, hi = ends[bi], ends[bi + 1]
        i = lo
        prev = lo  # the end of the positions that the last pass decoded
        while i < hi:
            if not ndl[i]:
                i += 1
                continue
            total_passes += 1
            # The skipped tokens since the last pass run as rows without outputs.
            rows += (i - prev) + k
            j = i + 1
            while j - i < k and j < hi and hl[j - 1]:
                if ndl[j] and not redecode and not hl[j]:
                    break
                j += 1
            prev = j
            i = j
    return total_passes, rows


GRID = [SkipParams(int(h * (1 << 22)), c, 0, "tokbkt", tb)
        for tb in (16, 18)
        for h in (0.80, 0.85, 0.90)
        for c in (1, 2)]


def _grid_file(job: tuple[str, int, Path]) -> list[tuple[float, int, int, int, int]]:
    """(bits, skipped, skip misses, tokens, bytes) of one file for rule 1, then each of GRID."""
    name, unit, pass_dir = job
    sim.PASS_DIR = pass_dir
    f = sim.load(name)
    out = []
    for s in [None, *GRID]:
        r = simulate(f, unit, s)
        out.append((r.bits, r.skipped, r.skip_miss, r.tokens, r.nbytes))
    return out


def cmd_grid(names: list[str], units: tuple[int, ...], workers: int) -> None:
    import multiprocessing as mp

    for unit in units:
        with mp.Pool(workers) as pool:
            per_file = pool.map(_grid_file, [(n, unit, sim.PASS_DIR) for n in names])
        b_bits = sum(pf[0][0] for pf in per_file)
        nb = sum(pf[0][4] for pf in per_file)
        nt = sum(pf[0][3] for pf in per_file)
        print(f"block {unit * 16} KiB: rule 1 {b_bits / nb:.4f} bpb, {nt} tokens", flush=True)
        for gi, s in enumerate(GRID, start=1):
            bits = sum(pf[gi][0] for pf in per_file)
            sk = sum(pf[gi][1] for pf in per_file)
            ms = sum(pf[gi][2] for pf in per_file)
            print(f"  {s.key:22s} {bits / nb:.4f} bpb ({100 * (bits / b_bits - 1):+.2f} %)  "
                  f"skipped {100 * sk / nt:5.1f} %  skip misses {100 * ms / max(sk, 1):.3f} %",
                  flush=True)


def _final_file(job: tuple[str, tuple[int, ...], Path]) -> list[dict]:
    name, units, pass_dir = job
    sim.PASS_DIR = pass_dir
    f = sim.load(name)
    rows = []
    for unit in units:
        base = simulate(f, unit, None)
        r = simulate(f, unit, V2)
        starts = f.block_starts[::unit].tolist()
        row = {"name": name, "block_kib": unit * 16, "bytes": f.nbytes, "tokens": r.tokens,
               "bpb_v1": base.bits / f.nbytes, "bpb_v2": r.bits / f.nbytes,
               "bits_v1": base.bits, "bits_v2": r.bits,
               "skipped": r.skipped, "skip_miss": r.skip_miss,
               "hit_share": float(r.hit.mean())}
        for skip in (False, True):
            for k in (1, 4, 8, 16, 32):
                for red in ((False, True) if k > 1 else (False,)):
                    p, rw = passes(r.need, r.hit, starts, k, red, skip)
                    tag = f"{'skip+' if skip else ''}spec{k}{'r' if red else ''}"
                    row[tag] = p
                    row[tag + "_rows"] = rw
        rows.append(row)
    return rows


def cmd_final(names: list[str], units: tuple[int, ...], out: Path | None, workers: int) -> None:
    import multiprocessing as mp

    with mp.Pool(workers) as pool:
        per_file = pool.map(_final_file, [(n, units, sim.PASS_DIR) for n in names])
    rows = [r for pf in per_file for r in pf]
    for row in rows:
        t = row["tokens"]
        print(f"{row['name']:18s} {row['block_kib']:5d} KiB  v1 {row['bpb_v1']:.4f}  "
              f"v2 {row['bpb_v2']:.4f} ({100 * (row['bits_v2'] / row['bits_v1'] - 1):+.2f} %)  "
              f"skipped {100 * row['skipped'] / t:5.1f} %  tokens/pass: spec16 "
              f"{t / row['spec16']:.2f}  spec16r {t / row['spec16r']:.2f}  skip "
              f"{t / row['skip+spec1']:.2f}  skip+spec16r {t / row['skip+spec16r']:.2f}",
              flush=True)
    for unit in units:
        sel = [r for r in rows if r["block_kib"] == unit * 16]
        nb = sum(r["bytes"] for r in sel)
        nt = sum(r["tokens"] for r in sel)
        b1 = sum(r["bits_v1"] for r in sel)
        b2 = sum(r["bits_v2"] for r in sel)
        sk = sum(r["skipped"] for r in sel)
        print(f"ALL {unit * 16} KiB: v1 {b1 / nb:.4f} bpb, v2 {b2 / nb:.4f} bpb "
              f"({100 * (b2 / b1 - 1):+.2f} %), skipped {100 * sk / nt:.1f} %", flush=True)
        for tag in ("spec1", "spec4", "spec4r", "spec8", "spec8r", "spec16", "spec16r", "spec32",
                    "spec32r"):
            for pre in ("", "skip+"):
                p = sum(r[pre + tag] for r in sel)
                rw = sum(r[pre + tag + "_rows"] for r in sel)
                print(f"  {pre + tag:14s} tokens/pass {nt / p:6.2f}  rows/pass {rw / p:6.2f}",
                      flush=True)
    if out is not None:
        out.write_text(json.dumps(rows, indent=1))


# The golden vectors: the inputs of `crates/taco-match/tests/golden.rs` (logs.vectors).
def vectors() -> None:
    from logs import vectors as vec

    toks = vec.tokens(20261018)
    match = mm.MatchParams((32, 16, 8, 4, 2, 1), 20, 64, 65535)
    pred, plen = mm.predict_block_ref(toks, match)
    from convert.rng import SplitMix64

    rng = SplitMix64(20261018 ^ 0xA5A5)
    # "small": a table of 2^6 entries (many collisions) and a low limit.
    for name, s in (("v2", V2), ("small", SkipParams(1 << 21, 1, 2, "tokbkt", 6))):
        rng = SplitMix64(20261018 ^ 0xA5A5)
        m = MixerV2(mx.MixerParams(), s)
        h = hashlib.sha256()
        skipped = 0
        for i, t in enumerate(toks):
            w = vec.weights(rng, t)
            if pred[i] < 0:
                h.update(struct.pack("<BI", 0, 0))
                continue
            qs = m.skip(plen[i], pred[i])
            if qs is not None:
                skipped += 1
                h.update(struct.pack("<BI", 2, qs))
            else:
                top = max(range(vec.VOCAB), key=lambda j: (w[j], -j))
                q = m.prob(w[pred[i]], sum(w), plen[i], top == pred[i])
                h.update(struct.pack("<BI", 1, q))
            m.update(pred[i] == t)
        print(f"{name:6s} skipped {skipped}  decisions {h.hexdigest()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["grid", "final", "vectors"])
    ap.add_argument("--pass-dir", type=Path)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--units", nargs="*", type=int, default=[4, 64])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "vectors":
        vectors()
        return 0
    if args.pass_dir is not None:
        sim.PASS_DIR = args.pass_dir
    names = [n for n in (args.files or sim.corpus.GPT2_LOGS)
             if (sim.PASS_DIR / f"{n}.npz").exists()]
    print(f"files: {names}", flush=True)
    if args.cmd == "grid":
        cmd_grid(names, tuple(args.units), args.workers)
    else:
        cmd_final(names, tuple(args.units), args.out, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
