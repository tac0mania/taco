"""Token windows for calibration and evaluation (plan §7.7).

A window has at most 1,024 tokens. The model input of a window is <|endoftext|> and then the
first 1,023 tokens, so the model predicts every token of the window.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baseline._paths import DATA_DIR, check_file
from baseline.bpe import EOT_ID, Tokenizer
from convert.rng import SplitMix64

N_CTX = 1024
CALIB_DIR = DATA_DIR / "calib"
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Window:
    """One window: a label, its tokens and its bytes."""

    label: str
    tokens: list[int]
    data: bytes

    @property
    def model_input(self) -> np.ndarray:
        return np.array([EOT_ID, *self.tokens[:-1]], dtype=np.int64)


def split(label: str, ids: list[int], tok: Tokenizer, count: int) -> list[Window]:
    out = []
    for i in range(count):
        toks = ids[i * N_CTX : (i + 1) * N_CTX]
        if not toks:
            break
        out.append(Window(f"{label}#{i}", toks, tok.decode(toks)))
    return out


def calibration_windows(tok: Tokenizer) -> list[Window]:
    """The calibration windows of calibration.toml. Each file is checked against its hash."""
    with (PACKAGE_DIR / "calibration.toml").open("rb") as f:
        spec = tomllib.load(f)
    out: list[Window] = []
    for s in spec["source"]:
        path = CALIB_DIR / s["name"]
        check_file(path, s["size"], s["sha256"])
        data = path.read_bytes()[s["offset"] : s["offset"] + s["cap"]]
        out += split(f"{s['name']}@{s['offset']}", tok.encode(data), tok, s["windows"])
    r = spec["random"]
    data = SplitMix64(r["seed"]).bytes(r["bytes"])
    out += split(f"random@{r['seed']}", tok.encode(data), tok, r["windows"])
    return out


def eval_windows(tok: Tokenizer) -> list[Window]:
    """The first window of each phase 0 corpus file (19 windows, the phase 1 part A set)."""
    from baseline.corpus import load_samples

    out = []
    for s in load_samples():
        out += split(s.name, tok.encode(s.data), tok, 1)
    return out
