"""Tokens of the measured files, in blocks of 16 KiB (ADR 0007 section 2, plan §10).

The encoder tokenizes each block alone, so a token never crosses a block boundary. All
measurements in this package use the 16 KiB block tokenization. A larger block (64 KiB to
1 MiB) is a sequence of 16 KiB token blocks. NOTE: this is the tokenization of 16 KiB blocks;
a real 64 KiB block can have a few different tokens at the 16 KiB boundaries (est. < 0.1 %).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from baseline.bpe import Tokenizer

BASE_BLOCK = 16384
N_CTX = 1024


@dataclass(frozen=True)
class Tokens:
    """The tokens of one file and the start of each 16 KiB block in the token array."""

    name: str
    nbytes: int
    ids: np.ndarray  # int64 [n]
    block_starts: np.ndarray  # int64 [blocks + 1]; the last entry is n
    block_bytes: np.ndarray  # int64 [blocks]


def tokenize(tok: Tokenizer, name: str, data: bytes) -> Tokens:
    ids: list[int] = []
    starts = [0]
    nb = []
    for off in range(0, len(data), BASE_BLOCK):
        part = data[off : off + BASE_BLOCK]
        t = tok.encode(part)
        if tok.decode(t) != part:
            raise ValueError(f"{name}: block at {off} does not restore")
        ids += t
        starts.append(len(ids))
        nb.append(len(part))
    return Tokens(name, len(data), np.array(ids, dtype=np.int64), np.array(starts, dtype=np.int64),
                  np.array(nb, dtype=np.int64))


def windows(t: Tokens) -> list[tuple[int, int]]:
    """The GPT-2 windows (start, end) in the token array: a new context at each block start
    and after each 1,024 tokens (plan §7.7, ADR 0004 section 9)."""
    out = []
    for b in range(len(t.block_bytes)):
        s, e = int(t.block_starts[b]), int(t.block_starts[b + 1])
        for w in range(s, e, N_CTX):
            out.append((w, min(w + N_CTX, e)))
    return out
