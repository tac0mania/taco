"""The tiny conformance model (plan §11.1): a random GPT-2 from a fixed seed, as `tiny.hsm`.

Run from tools/:
    nice uv run python -m convert.tiny [--new-hash]
    nice uv run python -m convert.tiny --calibrate   # only to make tiny_stats.toml again

The model has 2 blocks, width 64, 4 heads, an MLP width of 256, a context of 64 tokens and a
vocabulary of 300 tokens: the 256 single bytes (token `b` is the byte `b`), 43 BPE merges, and
`<|endoftext|>` (token 299). Thus any byte string can be tokenized (plan §10). Block 0 has an
`i16` GELU table and block 1 an `i8` GELU table, so the model uses both table types. The other
constants are the constants of GPT-2 (ADR 0004).

DETERMINISM: the weights come from splitmix64 (`convert.rng`), in a fixed order, with IEEE 754
binary64 additions and multiplications only (each has one correctly rounded result). The
calibration statistics are data: `--calibrate` runs the float model one time and writes
`tiny_stats.toml` (committed). The conversion reads that file and never runs the float model.
The conversion is exact (`convert.exact`, ADR 0014), and the tool compares its SHA-256 with
`expected.toml` (`convert.selfcheck`). `conformance/README.md` records the SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from convert import exact as ex
from convert.build import Scales, build
from convert.hsmfile import to_bytes, write_bytes
from convert.model import FBlock, FModel
from convert.rng import SplitMix64
from convert.selfcheck import Inputs, SelfCheckError, check

SEED = 0x7AC0_2026_1018
N_LAYER, N_HEAD, D_MODEL, D_FF, N_CTX = 2, 4, 64, 256, 64
GELU_BITS = {0: 16, 1: 8}
CALIB_SEQS = 8
OUT = Path(__file__).resolve().parents[2] / "conformance" / "tiny.hsm"
STATS = Path(__file__).resolve().parent / "tiny_stats.toml"

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


def calibrate(fm: FModel, rng: SplitMix64, path: Path = STATS) -> None:
    """Run the float model on random token sequences and write the statistics file.

    For each site: the largest |x|. For the residual: the largest |x| and the smallest row
    standard deviation. NOTE: this is the only step that runs the float model. Its last bits
    can depend on the machine, so its output is committed data.
    """
    from convert.model import forward

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
    lines = [
        "# Calibration statistics of the tiny conformance model (measured).",
        "# Made by `python -m convert.tiny --calibrate`.",
        "# The converter (`python -m convert.tiny`) reads this file. It does not run the model.",
        f"# {CALIB_SEQS} random sequences of {N_CTX} tokens, float64, `convert.model.forward`.",
        "",
        "[residual]",
        f"max = {res_max!r}",
        f"row_std_min = {min(row_std)!r}",
        "",
    ]
    for site, v in ranges.items():
        lines += [f'[site."{site}"]', f"max = {v!r}", ""]
    path.write_text("\n".join(lines))


def scales_from_stats(stats: dict) -> Scales:
    """The scales from the statistics, exact (ADR 0004 section 1, ADR 0014).

    The residual scale is the grid point at or above min(max / 2^24, row_std_min / 2^12.5):
    the largest value near 2^24 units, and every row std at 2^12 units or more. The argument
    is irrational, so the converter compares squares (`exact.round_range_sq`).
    """
    r = stats["residual"]
    a2 = (Fraction(r["max"]) / (1 << 24)) ** 2
    b2 = Fraction(r["row_std_min"]) ** 2 / (1 << 25)
    grid = {k: ex.round_range(v["max"]) for k, v in stats["site"].items()}
    return Scales.from_ranges(ex.round_range_sq(min(a2, b2)), grid, GELU_BITS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true",
                    help="run the float model and write tiny_stats.toml (no conversion)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--new-hash", action="store_true",
                    help="a deliberate change: write the file and print the new expected entry")
    args = ap.parse_args()
    rng = SplitMix64(SEED)
    fm = random_model(rng)
    if args.calibrate:
        calibrate(fm, rng)
        print(f"wrote {STATS}")
        return 0
    stats_sha = hashlib.sha256(STATS.read_bytes()).hexdigest()
    inputs = Inputs("tiny", f"splitmix64 seed {SEED:#x}", "max", stats_sha)
    with ex.no_float_model():
        sc = scales_from_stats(ex.load_toml_exact(STATS))
        vocab, merges = vocabulary_and_merges()
        data = to_bytes(build(fm, sc, vocab, merges, GELU_BITS))
    size, sha = len(data), hashlib.sha256(data).hexdigest()
    try:
        check(inputs, sha, size, args.new_hash)
    except SelfCheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write_bytes(data, args.out)
    print(f"wrote {args.out}: {size} bytes, sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
