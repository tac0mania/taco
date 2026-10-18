"""The golden vectors of `crates/taco-match` from the Python mirror (ADR 0010 section 6).

Run from tools/:
    uv run python -m logs.vectors

The tool makes the same inputs as `crates/taco-match/tests/golden.rs` (splitmix64, the same
order of calls), runs `logs.matchmodel` and `logs.mixer`, and prints the SHA-256 values. The
Rust test has the same values. Thus the Python mirror and the Rust crate give the same bits.
"""

from __future__ import annotations

import hashlib
import struct
import sys

from convert.rng import SplitMix64

from logs import matchmodel as mm
from logs import mixer as mx

VOCAB = 1000
TOKENS = 4000
NEWLINE = 198


def tokens(seed: int) -> list[int]:
    """A synthetic log: 8 templates, variable fields after some template tokens."""
    rng = SplitMix64(seed)
    templates = []
    for _ in range(8):
        n = 5 + rng.below(20)
        templates.append([rng.below(VOCAB) for _ in range(n)])
    out: list[int] = []
    while len(out) < TOKENS:
        t = templates[rng.below(8)]
        for tok in t:
            out.append(tok)
            if rng.below(4) == 0:
                for _ in range(1 + rng.below(3)):
                    out.append(rng.below(VOCAB))
        out.append(NEWLINE)
    return out[:TOKENS]


def weights(rng: SplitMix64, actual: int) -> list[int]:
    w = [rng.next_u64() >> (20 + rng.below(44)) for _ in range(VOCAB)]
    if rng.below(2) == 0:
        w[actual] = 1 << 40
    return w


def run(seed: int, match: mm.MatchParams, mixp: mx.MixerParams) -> tuple[str, str]:
    """(SHA-256 of the predictions, SHA-256 of all mixed weights)."""
    toks = tokens(seed)
    pred, plen = mm.predict_block_ref(toks, match)
    hp = hashlib.sha256()
    hw = hashlib.sha256()
    rng = SplitMix64(seed ^ 0xA5A5)
    m = mx.Mixer(mixp)
    for i, t in enumerate(toks):
        hp.update(struct.pack("<II", pred[i] & 0xFFFFFFFF, plen[i]))
        w = weights(rng, t)
        if pred[i] >= 0:
            top = max(range(VOCAB), key=lambda j: (w[j], -j))
            q = m.prob(w[pred[i]], sum(w), plen[i], top == pred[i])
            out = mx.mix_row(w, pred[i], q)
            m.update(pred[i] == t)
        else:
            out = mx.mix_row(w, -1, 0)
        hw.update(struct.pack(f"<{VOCAB}Q", *out))
    return hp.hexdigest(), hw.hexdigest()


def main() -> int:
    sq = hashlib.sha256(struct.pack(f"<{len(mx.SQUASH)}H", *mx.SQUASH)).hexdigest()
    st = hashlib.sha256(struct.pack(f"<{len(mx.STRETCH)}h", *mx.STRETCH)).hexdigest()
    print(f"squash  {sq}")
    print(f"stretch {st}")
    v1 = mm.MatchParams((32, 16, 8, 4, 2, 1), 20, 64, 65535)
    small = mm.MatchParams((8, 4, 2, 1), 6, 16, 40)
    for name, p in (("v1", v1), ("small", small)):
        hp, hw = run(20261018, p, mx.MixerParams())
        print(f"{name:6s} predictions {hp}")
        print(f"{name:6s} weights     {hw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
