"""The tiny conformance model (plan §11.1): a random GPT-2 from a fixed seed, as `tiny.hsm`.

Run from tools/:
    nice uv run python -m convert.tiny

The model has 2 blocks, width 64, 4 heads, an MLP width of 256, a context of 64 tokens and a
vocabulary of 300 tokens: the 256 single bytes (token `b` is the byte `b`), 43 BPE merges, and
`<|endoftext|>` (token 299). Thus any byte string can be tokenized (plan §10). Block 0 has an
`i16` GELU table and block 1 an `i8` GELU table, so the model uses both table types. The other
constants are the constants of GPT-2 (ADR 0004).

DETERMINISM: the weights come from splitmix64 (`convert.rng`), in a fixed order. The
calibration runs a float forward pass. Its last bits can depend on the machine, so every range
is rounded up to the grid 2^(k/16) (`numerics.round_range`) before it becomes a scale. The
committed file is the reference; `conformance/README.md` records its SHA-256.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from convert import numerics as nm
from convert.build import Scales, build
from convert.hsmfile import write
from convert.model import FBlock, FModel, forward
from convert.rng import SplitMix64

SEED = 0x7AC0_2026_1018
N_LAYER, N_HEAD, D_MODEL, D_FF, N_CTX = 2, 4, 64, 256, 64
GELU_BITS = {0: 16, 1: 8}
CALIB_SEQS = 8
OUT = Path(__file__).resolve().parents[2] / "conformance" / "tiny.hsm"

# The merges in rank order, as (left, right) byte strings. Merge i makes token 256 + i.
MERGES = [
    (b"t", b"h"), (b"h", b"e"), (b"i", b"n"), (b"e", b"r"), (b"a", b"n"), (b" ", b"t"),
    (b"r", b"e"), (b"o", b"n"), (b"th", b"e"), (b" t", b"he"), (b"e", b"n"), (b"a", b"t"),
    (b"n", b"d"), (b"an", b"d"), (b"in", b"g"), (b"o", b"f"), (b" ", b"a"), (b" ", b"o"),
    (b" o", b"f"), (b" a", b"nd"), (b"e", b"s"), (b"o", b"r"), (b"i", b"s"), (b" ", b"i"),
    (b" i", b"s"), (b"e", b"d"), (b"t", b"o"), (b" ", b"s"), (b"a", b"r"), (b"l", b"e"),
    (b"\r", b"\n"), (b"\n", b"\n"), (b" ", b" "), (b"  ", b"  "), (b"{", b"\n"),
    (b"(", b")"), (b"0", b"0"), (b"\xe4", b"\xb8"), (b"\xc3", b"\xa9"), (b"i", b"on"),
    (b"t", b"ion"), (b" ", b"w"), (b" w", b"in"),
]
VOCAB = 256 + len(MERGES) + 1
EOT = VOCAB - 1


def vocabulary_and_merges() -> tuple[list[bytes], list[tuple[int, int]]]:
    """The 300 tokens and the merges as id pairs. Every merge part is already a token."""
    vocab = [bytes([b]) for b in range(256)]
    ids = {t: i for i, t in enumerate(vocab)}
    merges = []
    for left, right in MERGES:
        merges.append((ids[left], ids[right]))
        joined = left + right
        if joined in ids:
            raise ValueError(f"merge {joined!r} makes a token that exists")
        ids[joined] = len(vocab)
        vocab.append(joined)
    vocab.append(b"<|endoftext|>")
    return vocab, merges


def _normal(rng: SplitMix64, shape: tuple[int, ...], mean: float, std: float) -> np.ndarray:
    n = int(np.prod(shape))
    return np.array([mean + std * rng.normal() for _ in range(n)]).reshape(shape)


def random_model(rng: SplitMix64) -> FModel:
    """A random float model with the shapes and roughly the statistics of a trained GPT-2."""
    d, f = D_MODEL, D_FF
    sd = 1.0 / np.sqrt(float(d))
    wte = _normal(rng, (VOCAB, d), 0.0, 2.5 * sd)
    wpe = _normal(rng, (N_CTX, d), 0.0, 0.75 * sd)
    blocks = []
    for _ in range(N_LAYER):
        blocks.append(FBlock(
            ln1_g=_normal(rng, (d,), 1.0, 0.1), ln1_b=_normal(rng, (d,), 0.0, 0.02),
            attn_w=_normal(rng, (3 * d, d), 0.0, 1.5 * sd), attn_b=_normal(rng, (3 * d,), 0.0, 0.02),
            proj_w=_normal(rng, (d, d), 0.0, 0.5 * sd), proj_b=_normal(rng, (d,), 0.0, 0.02),
            ln2_g=_normal(rng, (d,), 1.0, 0.1), ln2_b=_normal(rng, (d,), 0.0, 0.02),
            fc_w=_normal(rng, (f, d), 0.0, sd), fc_b=_normal(rng, (f,), 0.0, 0.02),
            mlp_w=_normal(rng, (d, f), 0.0, 0.5 / np.sqrt(float(f))),
            mlp_b=_normal(rng, (d,), 0.0, 0.02),
        ))
    return FModel(wte=wte, wpe=wpe, blocks=blocks, lnf_g=_normal(rng, (d,), 1.0, 0.1),
                  lnf_b=_normal(rng, (d,), 0.0, 0.02), n_head=N_HEAD, eot=EOT)


def calibrate(fm: FModel, rng: SplitMix64) -> Scales:
    """Ranges from the float model on random token sequences: the largest value of each site."""
    ranges: dict[str, float] = {}
    row_std = []

    def hook(site: str, x: np.ndarray) -> None:
        if site == "res":
            row_std.append(float(x.std(axis=1).min()))
        ranges[site] = max(ranges.get(site, 0.0), float(np.abs(x).max()))

    for _ in range(CALIB_SEQS):
        toks = np.array([EOT] + [rng.below(VOCAB - 1) for _ in range(N_CTX - 1)])
        forward(fm, toks, hook=hook, det_gelu_fn=True)
    res_max = ranges.pop("res")
    # ADR 0004 section 1: the largest value near 2^24, and every row std at 2^12 units or more.
    s_res = min(res_max / nm.pow2(24), min(row_std) / nm.pow2(12) / np.sqrt(2.0))
    grid = {k: nm.round_range(v) for k, v in ranges.items()}
    return Scales.from_ranges(nm.round_range(s_res), grid, GELU_BITS)


def main() -> int:
    rng = SplitMix64(SEED)
    fm = random_model(rng)
    sc = calibrate(fm, rng)
    vocab, merges = vocabulary_and_merges()
    hsm = build(fm, sc, vocab, merges, GELU_BITS)
    size, sha = write(hsm, OUT)
    print(f"wrote {OUT}: {size} bytes, sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
