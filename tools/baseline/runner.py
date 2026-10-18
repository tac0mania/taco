"""Windows, model passes and coder jobs that phase 0 and phase 1 share.

PLAN(§7.7): a context window holds at most 1,024 tokens. The model input of a window is
<|endoftext|> followed by the tokens of the window except the last one. Thus the model predicts
each token of the window, and the context starts again after 1,024 tokens.

Each window is one coder stream here (one block in §9 terms), so that windows can be decoded
in parallel. The decoder knows the token count of each window (container metadata).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from baseline.bpe import EOT_ID
from baseline.gpt2_np import GPT2, N_CTX
from baseline.rangecoder import (
    FREQ_BITS,
    Decoder,
    Encoder,
    cumulative,
    log2_prob,
    tables_from_logits,
)

ROW_CHUNK = 64  # Rows of logits that become frequency tables in one NumPy call.


@dataclass
class Window:
    """One context window of one file."""

    file: str
    index: int
    tokens: list[int]
    byte_offset: int  # Offset of the first byte of the window in the measured sample.
    data: bytes  # The bytes of the window.

    @property
    def model_input(self) -> np.ndarray:
        return np.array([EOT_ID, *self.tokens[:-1]], dtype=np.int64)


def make_windows(name: str, ids: list[int], token_bytes: list[bytes]) -> list[Window]:
    """Divide the token ids of one sample into windows of N_CTX tokens."""
    out = []
    offset = 0
    for i in range(0, len(ids), N_CTX):
        toks = ids[i : i + N_CTX]
        data = b"".join(token_bytes[t] for t in toks)
        out.append(Window(name, i // N_CTX, toks, offset, data))
        offset += len(data)
    return out


@dataclass
class WindowResult:
    """The result of one pass over one window."""

    ideal_bits: float = 0.0  # Sum of -log2 p with the float probabilities.
    table_bits: float = 0.0  # Sum of -log2 (f / T) with the integer tables (§5.1).
    floor_tokens: int = 0  # Tokens with float p < 2^-32 (more than 32 bits, float).
    floor_ideal_bits: float = 0.0  # The float bits of those tokens.
    stream: bytes = b""
    table_hashes: list[bytes] = field(default_factory=list)


def account(res: WindowResult, f: int, log2p: float) -> None:
    """Add the float and the table information content of one coded token."""
    res.ideal_bits -= log2p
    res.table_bits += FREQ_BITS - math.log2(f)
    if log2p < -FREQ_BITS:
        res.floor_tokens += 1
        res.floor_ideal_bits -= log2p


def iter_tables(logits: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray, float]]:
    """Yield (freq, logits row, log-sum-exp) for each row of a logits matrix, in chunks."""
    for r in range(0, len(logits), ROW_CHUNK):
        chunk = logits[r : r + ROW_CHUNK]
        freq, lse = tables_from_logits(chunk)
        yield from zip(freq, chunk, lse, strict=True)


def table_hash(freq: np.ndarray) -> bytes:
    """SHA-256 of a frequency table as little-endian u32 values (16 bytes kept)."""
    return hashlib.sha256(freq.astype("<u4").tobytes()).digest()[:16]


def encode_with_tables(
    win: Window, tables: Iterable[tuple[np.ndarray, np.ndarray, float]], hashes: bool
) -> WindowResult:
    """Encode the tokens of a window with one table for each token."""
    enc = Encoder()
    res = WindowResult()
    for tok, (freq, row, lse) in zip(win.tokens, tables, strict=True):
        cum = int(freq[:tok].sum())
        enc.encode(cum, int(freq[tok]))
        account(res, int(freq[tok]), log2_prob(row[tok], lse))
        if hashes:
            res.table_hashes.append(table_hash(freq))
    res.stream = enc.finish()
    return res


def full_logits(model: GPT2, win: Window) -> np.ndarray:
    """The full-sequence path: one causal pass over the window."""
    return model.forward(win.model_input)


# The KV-cache path, with a batch of slots --------------------------------------------------


class Job:
    """A stream that the KV runner advances one token at a time."""

    def first_input(self) -> int:
        return EOT_ID

    def consume(self, logits_row: np.ndarray) -> int | None:
        """Use the logits for the next token. Return the next input token, or None at the end."""
        raise NotImplementedError


class EncodeJob(Job):
    """Teacher forcing on the KV path: encode the known tokens of a window."""

    def __init__(self, win: Window) -> None:
        self.win = win
        self.i = 0
        self.enc = Encoder()
        self.res = WindowResult()

    def consume(self, logits_row: np.ndarray) -> int | None:
        freq, lse = (a[0] for a in tables_from_logits(logits_row[None, :]))
        tok = self.win.tokens[self.i]
        self.enc.encode(int(freq[:tok].sum()), int(freq[tok]))
        account(self.res, int(freq[tok]), log2_prob(logits_row[tok], lse))
        self.res.table_hashes.append(table_hash(freq))
        self.i += 1
        if self.i == len(self.win.tokens):
            self.res.stream = self.enc.finish()
            return None
        return tok


class DecodeJob(Job):
    """A real decode on the KV path: each decoded token is the next model input.

    The job stops at the first token that is not the original token, because all output
    after that point is wrong (and a real decoder cannot know it).
    """

    def __init__(self, win: Window, stream: bytes, token_bytes: list[bytes]) -> None:
        self.win = win
        self.dec = Decoder(stream)
        self.token_bytes = token_bytes
        self.out: list[int] = []
        self.first_bad_byte: int | None = None  # Offset in the window.

    def consume(self, logits_row: np.ndarray) -> int | None:
        freq = tables_from_logits(logits_row[None, :])[0][0]
        tok = self.dec.decode(cumulative(freq))
        i = len(self.out)
        self.out.append(tok)
        if tok != self.win.tokens[i]:
            done = b"".join(self.token_bytes[t] for t in self.out[:-1])
            got = self.token_bytes[tok]
            want = self.win.data[len(done) :]
            common = 0
            while common < min(len(got), len(want)) and got[common] == want[common]:
                common += 1
            self.first_bad_byte = len(done) + common
            return None
        if len(self.out) == len(self.win.tokens):
            return None
        return tok


def run_kv(
    model: GPT2,
    jobs: list[Job],
    slots: int,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Run the jobs on the KV path with a fixed batch of `slots` streams.

    A free slot takes the next job in list order. An idle slot feeds <|endoftext|> at position
    0, so the batch shape stays the same at every step. With the same jobs and the same token
    counts, the schedule is the same in every run.
    """
    state = model.new_state(slots)
    active: list[Job | None] = [None] * slots
    inputs = np.full(slots, EOT_ID, dtype=np.int64)
    queue = list(jobs)
    queue.reverse()
    finished = 0
    steps = 0

    def fill(s: int) -> None:
        if queue:
            job = queue.pop()
            active[s] = job
            inputs[s] = job.first_input()
        else:
            active[s] = None
            inputs[s] = EOT_ID
        state.pos[s] = 0

    for s in range(slots):
        fill(s)
    while any(j is not None for j in active):
        for s in range(slots):
            if active[s] is None:
                state.pos[s] = 0
        logits = model.step(state, inputs)
        steps += 1
        for s in range(slots):
            job = active[s]
            if job is None:
                continue
            nxt = job.consume(logits[s])
            if nxt is None:
                finished += 1
                fill(s)
            else:
                inputs[s] = nxt
        if progress and steps % 256 == 0:
            progress(steps, finished)
