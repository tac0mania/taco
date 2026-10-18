"""GPT-2 byte-level BPE tokenizer: the reference for the Rust tokenizer (plan §10).

The encoder accepts any byte string. It divides the input into maximal runs of well-formed
UTF-8 and single bytes that are not part of well-formed UTF-8. A UTF-8 run goes through the
GPT-2 pre-tokenizer regex and the BPE merges. A byte outside a UTF-8 run becomes its
single-byte token. Every token maps to a fixed byte string, so the decoder only concatenates
the bytes of the tokens, and `decode(encode(x)) == x` for every byte string `x`.

NOTE: for valid UTF-8 input the output is the canonical GPT-2 tokenization. For other input it
is not canonical (plan §10 permits this), because the regex never runs across an invalid byte.
NOTE: the regex classes \\p{L} and \\p{N} depend on the Unicode version of the `regex` module.
The Rust tokenizer must use the same Unicode version, or the token ids can differ (the bytes
still come back exactly).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import regex

from baseline._paths import MODEL_DIR

EOT_ID = 50256
EOT_TEXT = "<|endoftext|>"

# PLAN(§10): the GPT-2 pre-tokenizer pattern, as published in openai/gpt-2 encoder.py.
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Well-formed UTF-8 sequences (Unicode 15, table 3-7). No surrogates, no overlong forms.
UTF8_RUN = re.compile(
    rb"(?:[\x00-\x7f]"
    rb"|[\xc2-\xdf][\x80-\xbf]"
    rb"|\xe0[\xa0-\xbf][\x80-\xbf]"
    rb"|[\xe1-\xec\xee\xef][\x80-\xbf]{2}"
    rb"|\xed[\x80-\x9f][\x80-\xbf]"
    rb"|\xf0[\x90-\xbf][\x80-\xbf]{2}"
    rb"|[\xf1-\xf3][\x80-\xbf]{3}"
    rb"|\xf4[\x80-\x8f][\x80-\xbf]{2})+"
)


def bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 map from each byte value to a printable character."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


class Tokenizer:
    """GPT-2 byte-level BPE. Construction reads vocab.json and merges.txt one time."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        with (model_dir / "vocab.json").open(encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)
        lines = (model_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
        merges = [tuple(line.split(" ")) for line in lines[1:] if line]
        self.byte_encoder = bytes_to_unicode()
        byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.encoder = vocab
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.vocab_size = len(vocab)
        # The decoder table: token id -> bytes. This is the vocabulary of the model file (§8).
        self.token_bytes: list[bytes] = [b""] * self.vocab_size
        for text, i in vocab.items():
            if text == EOT_TEXT:
                continue  # NOTE: the special token has no bytes; the encoder never emits it.
            self.token_bytes[i] = bytes(byte_decoder[c] for c in text)
        self.single_byte = [vocab[self.byte_encoder[b]] for b in range(256)]
        self._cache: dict[str, list[int]] = {}

    def _bpe(self, piece: str) -> list[int]:
        """Apply the merges to one pre-token (in byte-encoder characters)."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        word = list(piece)
        while len(word) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            first, second = word[best_i], word[best_i + 1]
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        ids = [self.encoder[w] for w in word]
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def _encode_text(self, text: str, out: list[int]) -> None:
        for m in GPT2_PATTERN.finditer(text):
            piece = "".join(self.byte_encoder[b] for b in m.group().encode("utf-8"))
            out.extend(self._bpe(piece))

    def encode(self, data: bytes) -> list[int]:
        """Return the token ids for any byte string. Never emits <|endoftext|>."""
        out: list[int] = []
        pos = 0
        for m in UTF8_RUN.finditer(data):
            for b in data[pos : m.start()]:
                out.append(self.single_byte[b])
            self._encode_text(m.group().decode("utf-8"), out)
            pos = m.end()
        for b in data[pos:]:
            out.append(self.single_byte[b])
        return out

    def decode(self, ids: list[int]) -> bytes:
        """Return the bytes of the token ids. <|endoftext|> gives no bytes."""
        tb = self.token_bytes
        return b"".join(tb[i] for i in ids)
