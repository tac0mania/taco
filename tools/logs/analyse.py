"""Where the bits go: the cost of each token class, with GPT-2 alone and with the mixer.

Run from tools/:
    uv run python -m logs.analyse [--block 1048576] [--files ...]

Token classes (from the bytes of the token without spaces): `digits` (only digits), `hex`
(only hex digits, with at least one letter a-f and one digit), `word` (only letters),
`space` (only white space), `other` (punctuation and mixed). Match states: `none` (no
prediction), `hit`, `miss`. Output: tools/data/out/logs/analyse_<block>.json and a table.
"""

from __future__ import annotations

import argparse
import json
import string
import sys

import numpy as np
from baseline.bpe import Tokenizer

from logs import corpus, sim
from logs import matchmodel as mm

HEX = set(string.hexdigits.lower())


def classes(tok: Tokenizer) -> list[str]:
    out = []
    for b in tok.token_bytes:
        s = b.strip().lower()
        if not s:
            out.append("space")
        elif s.isdigit():
            out.append("digits")
        elif all(chr(c) in HEX for c in s) and any(chr(c).isdigit() for c in s) and any(
                chr(c).isalpha() for c in s):
            out.append("hex")
        elif s.isalpha():
            out.append("word")
        else:
            out.append("other")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", type=int, default=1048576)
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()
    tok = Tokenizer()
    cls = np.array(classes(tok))
    rows = []
    for name in sim.available(args.files):
        f = sim.load(name)
        unit = sim.UNITS[args.block]
        pred, plen = mm.predict_file(f.ids, f.block_starts, unit, sim.DEFAULT_MATCH)
        wp = sim.pred_weight(f, pred)
        q = sim.run_mixer(f, pred, plen, wp, unit, sim.DEFAULT_MIXER)
        mixed = sim.approx_bits(f, pred, wp, q)
        gpt = -np.log2(f.fa.astype(np.float64) / sim.TOTAL)
        c = cls[f.ids]
        state = np.where(pred < 0, "none", np.where(pred == f.ids, "hit", "miss"))
        row = {"name": name, "bytes": f.nbytes, "tokens": len(f.ids), "classes": {},
               "states": {}}
        for k in ("digits", "hex", "word", "space", "other"):
            m = c == k
            row["classes"][k] = {"tokens": int(m.sum()), "gpt2_bytes": float(gpt[m].sum() / 8),
                                 "mixed_bytes": float(mixed[m].sum() / 8)}
        for k in ("none", "hit", "miss"):
            m = state == k
            row["states"][k] = {"tokens": int(m.sum()), "gpt2_bytes": float(gpt[m].sum() / 8),
                                "mixed_bytes": float(mixed[m].sum() / 8)}
        rows.append(row)
        tot = mixed.sum() / 8
        parts = "  ".join(f"{k} {100 * v['mixed_bytes'] / tot:4.1f}%"
                          for k, v in row["classes"].items())
        st = "  ".join(f"{k} {v['tokens'] / len(f.ids):.2f}/{100 * v['mixed_bytes'] / tot:4.1f}%"
                       for k, v in row["states"].items())
        print(f"{name:18s} mixed {tot:9.0f} B | {parts} | {st}", flush=True)
    (corpus.OUT_DIR / f"analyse_{args.block}.json").write_text(json.dumps(rows, indent=1))
    return 0



def numeric_fields(name: str, block: int = 1048576) -> dict:
    """The mixed cost of the digit runs of a file: runs seen before in the block, and new runs.

    A token belongs to a run if its bytes overlap the run. The result estimates how much a
    model of field values (a cache of earlier numbers) could save: the cost of repeated runs.
    """
    import re

    tok = Tokenizer()
    f = sim.load(name)
    unit = sim.UNITS[block]
    pred, plen = mm.predict_file(f.ids, f.block_starts, unit, sim.DEFAULT_MATCH)
    wp = sim.pred_weight(f, pred)
    q = sim.run_mixer(f, pred, plen, wp, unit, sim.DEFAULT_MIXER)
    mixed = sim.approx_bits(f, pred, wp, q)
    lens = np.array([len(tok.token_bytes[t]) for t in f.ids.tolist()])
    ends = np.cumsum(lens)
    starts = ends - lens
    data = b"".join(tok.token_bytes[t] for t in f.ids.tolist())
    seen: set[bytes] = set()
    rep = new = 0.0
    rep_n = new_n = 0
    rep_d = new_d = 0
    for m in re.finditer(rb"\d+", data):
        a, b = m.start(), m.end()
        i0 = int(np.searchsorted(ends, a, side="right"))
        i1 = int(np.searchsorted(starts, b, side="left"))
        cost = float(mixed[i0:i1].sum())
        if m.group() in seen:
            rep += cost
            rep_n += 1
            rep_d += b - a
        else:
            new += cost
            new_n += 1
            new_d += b - a
            seen.add(m.group())
    tot = float(mixed.sum())
    return {"name": name, "total_bytes": tot / 8, "repeated_runs": rep_n, "new_runs": new_n,
            "repeated_bytes": rep / 8, "new_bytes": new / 8,
            "repeated_bits_per_digit": rep / max(rep_d, 1), "new_bits_per_digit": new / max(new_d, 1)}


def hit_runs(name: str, block: int = 1048576) -> dict:
    """Runs of correct predictions: the draft lengths that speculative decoding could accept.

    A run is a maximal sequence of tokens that the match model predicts correctly. The result
    gives the share of the tokens in runs of at least 4, 8 and 16 tokens, and the mean run
    length for a token (a token in a run of length n counts n).
    """
    f = sim.load(name)
    pred, _ = mm.predict_file(f.ids, f.block_starts, sim.UNITS[block], sim.DEFAULT_MATCH)
    hit = (pred == f.ids).astype(np.int8)
    runs = []
    n = 0
    for h in hit.tolist():
        if h:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    r = np.array(runs or [0])
    tot = len(f.ids)
    return {"name": name, "hit_share": float(hit.mean()),
            "in_runs_4": float(r[r >= 4].sum() / tot), "in_runs_8": float(r[r >= 8].sum() / tot),
            "in_runs_16": float(r[r >= 16].sum() / tot),
            "mean_run_per_token": float((r * r).sum() / max(r.sum(), 1))}


if __name__ == "__main__":
    sys.exit(main())
