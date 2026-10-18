"""The `.hsm` writer in Python, format version 1 (plan §8, ADR 0004 section 8 and addendum A).

`hardshell::hsm::HsmWriter` is the reference. This writer makes the same bytes for the same
parts: compact JSON with the fields in schema order, the same loop for the header length,
spaces at the end of the header, zero gap bytes and 64-byte alignment. A Rust test
(`crates/hardshell/tests/hsm_merges.rs`) writes the tiny model again with `HsmWriter` and
compares the bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"HSHLMDL\0"
VERSION = 1
ALIGN = 64
PREAMBLE_LEN = 16

_DTYPES = {
    "i8": np.dtype("<i1"),
    "i16": np.dtype("<i2"),
    "i32": np.dtype("<i4"),
    "u64": np.dtype("<u8"),
}


def align64(x: int) -> int:
    return (x + ALIGN - 1) // ALIGN * ALIGN


@dataclass
class Tensor:
    """One tensor: name, dtype name, values (row-major) and the rescale shift (or None)."""

    name: str
    dtype: str
    data: np.ndarray
    shift: int | None = None

    def raw(self) -> bytes:
        return np.ascontiguousarray(self.data, dtype=_DTYPES[self.dtype]).tobytes()


@dataclass
class HsmModel:
    """All parts of a model file. `config` has the fields of `model` in schema order."""

    config: dict
    vocabulary: list[bytes]
    merges: list[tuple[int, int]] = field(default_factory=list)
    tensors: list[Tensor] = field(default_factory=list)

    def add(self, name: str, dtype: str, data: np.ndarray, shift: int | None = None) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"unknown dtype {dtype}")
        if shift is not None and dtype != "i32":
            raise ValueError("only an i32 tensor of rescale multipliers has a shift")
        if shift is not None and not 0 <= shift <= 255:
            raise ValueError("shift must be a u8")
        arr = np.asarray(data)
        info = np.iinfo(_DTYPES[dtype])
        if arr.size and (int(arr.min()) < info.min or int(arr.max()) > info.max):
            raise ValueError(f"{name}: values outside {dtype}")
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"duplicate tensor {name}")
        self.tensors.append(Tensor(name, dtype, arr, shift))

    def tensor(self, name: str) -> Tensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)


def _vocab_bytes(tokens: list[bytes]) -> bytes:
    out = bytearray()
    for t in tokens:
        if len(t) > 0xFFFF:
            raise ValueError("a token has more than 65,535 bytes")
        out += len(t).to_bytes(2, "little") + t
    return bytes(out)


def _merges_bytes(merges: list[tuple[int, int]]) -> bytes:
    if not merges:
        return b""
    return np.asarray(merges, dtype="<u4").reshape(-1, 2).tobytes()


def _header(m: HsmModel, header_len: int, vocab_len: int, merges_len: int, raws: list[bytes]) -> dict:
    vocab_offset = align64(PREAMBLE_LEN + header_len)
    cursor = vocab_offset + vocab_len
    h: dict = {"model": m.config, "vocabulary": {"offset": vocab_offset, "length": vocab_len}}
    if merges_len:
        off = align64(cursor)
        h["merges"] = {"offset": off, "length": merges_len}
        cursor = off + merges_len
    entries = []
    for t, raw in zip(m.tensors, raws, strict=True):
        off = align64(cursor)
        e = {
            "name": t.name,
            "dtype": t.dtype,
            "shape": [int(d) for d in t.data.shape],
            "offset": off,
            "length": len(raw),
        }
        if t.shift is not None:
            e["shift"] = int(t.shift)
        entries.append(e)
        cursor = off + len(raw)
    h["tensors"] = entries
    return h


def to_bytes(m: HsmModel) -> bytes:
    """Lay out and serialise the model file (the same rules as `HsmWriter::finish`)."""
    vocab = _vocab_bytes(m.vocabulary)
    merges = _merges_bytes(m.merges)
    raws = [t.raw() for t in m.tensors]
    header_len = 0
    while True:
        h = _header(m, header_len, len(vocab), len(merges), raws)
        text = json.dumps(h, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(text) <= header_len:
            break
        header_len = len(text)
    out = bytearray(MAGIC)
    out += VERSION.to_bytes(4, "little") + header_len.to_bytes(4, "little")
    out += text
    out += b" " * (PREAMBLE_LEN + header_len - len(out))
    out += b"\0" * (h["vocabulary"]["offset"] - len(out))
    out += vocab
    if merges:
        out += b"\0" * (h["merges"]["offset"] - len(out))
        out += merges
    for e, raw in zip(h["tensors"], raws, strict=True):
        out += b"\0" * (e["offset"] - len(out))
        out += raw
    return bytes(out)


def write_bytes(data: bytes, path) -> None:
    """Write the bytes of a model file (through a `.part` file, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def write(m: HsmModel, path) -> tuple[int, str]:
    """Write the file. Returns (size in bytes, SHA-256 hex)."""
    data = to_bytes(m)
    write_bytes(data, path)
    return len(data), hashlib.sha256(data).hexdigest()


def gpt2_config(
    *, n_layer: int, n_head: int, d_model: int, d_ff: int, vocab_size: int, n_ctx: int,
    eot_token: int, ln_k: int, ln_j: int, ln_eps: int, softmax_p: int,
) -> dict:
    """The `model` object of the header, with the fields in schema order."""
    return {
        "architecture": "gpt2",
        "n_layer": n_layer,
        "n_head": n_head,
        "d_model": d_model,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "eot_token": eot_token,
        "constants": {"ln_k": ln_k, "ln_j": ln_j, "ln_eps": ln_eps, "softmax_p": softmax_p},
    }
