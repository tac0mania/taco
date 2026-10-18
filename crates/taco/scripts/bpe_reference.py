"""Writes the token ids of the Python reference tokenizer, for the Rust canonical test.

The Rust test `crates/taco/tests/tokenizer_gpt2.rs` compares its tokens with these files. Run
from the repository root, after `python -m baseline.fetch`:

    uv run --project tools python -I crates/taco/scripts/bpe_reference.py <out_dir>
    TACO_BPE_REFERENCE_DIR=<out_dir> scripts/mx cargo test -p taco --test tokenizer_gpt2

For each file `<name>` in tools/data/corpus, the script writes `corpus/<name>.ids`. It also
writes generated texts: `generated/<i>.bin` and `generated/<i>.ids`. An `.ids` file holds one
little-endian u32 for each token.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from baseline._paths import CORPUS_DIR  # noqa: E402
from baseline.bpe import Tokenizer  # noqa: E402

# Character pools for the generated texts. Each pool tests one rule of the pre-tokenizer.
POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~",
    "'''''",
    "    \t\n\n\r\x0b\x0c",
    " \u0085        　",
    "हिन्दीमेंपाठकरेंि्ं",
    "中文文本汉字測試",
    "العربيةنصًّ",
    "日本語のテキストカタカナ",
    "😀🎉👍🏽‍❤️",
    "éèêëçñßøåæœ́̈",
    "ⅠⅡⅢⅫ²³¹½¾٣٤۵߁",
    "\U00016ea0\U00016ea1\U0001e5d0\U00010d50Ᲊ",
]
WORDS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", " the", " and", "don't", "  ", "\n\n"]


def generated(rng: random.Random, n: int) -> bytes:
    """A text of about `n` characters from random pools, with some invalid bytes."""
    parts: list[str] = []
    while sum(len(p) for p in parts) < n:
        if rng.random() < 0.2:
            parts.append(rng.choice(WORDS))
        else:
            pool = rng.choice(POOLS)
            parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    data = "".join(parts).encode("utf-8")
    if rng.random() < 0.3:
        b = bytearray(data)
        for _ in range(rng.randint(1, 4)):
            b.insert(rng.randrange(len(b) + 1), rng.randrange(256))
        data = bytes(b)
    return data


def write_ids(path: Path, ids: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(ids)}I", *ids))


def main() -> None:
    out = Path(sys.argv[1])
    (out / "corpus").mkdir(parents=True, exist_ok=True)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    count = 0
    for f in sorted(CORPUS_DIR.iterdir()):
        if f.is_file():
            write_ids(out / "corpus" / f"{f.name}.ids", tok.encode(f.read_bytes()))
            count += 1
    rng = random.Random(20261018)
    for i in range(2000):
        data = generated(rng, rng.choice([1, 5, 20, 100, 400]))
        (out / "generated" / f"{i}.bin").write_bytes(data)
        write_ids(out / "generated" / f"{i}.ids", tok.encode(data))
    print(f"wrote {count} corpus files and 2000 generated texts to {out}")


if __name__ == "__main__":
    main()
