"""Paths and file checks that the baseline tools share."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TOOLS_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
CORPUS_DIR = DATA_DIR / "corpus"
OUT_DIR = DATA_DIR / "out"
PACKAGE_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file as lower-case hex. One pass over the file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_toml(name: str) -> dict:
    """Load a TOML file from the package directory."""
    with (PACKAGE_DIR / name).open("rb") as f:
        return tomllib.load(f)


def check_file(path: Path, size: int, sha256: str) -> None:
    """Raise ValueError if the file does not have the expected size and SHA-256."""
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"{path.name}: size {actual_size}, expected {size}")
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"{path.name}: sha256 {actual}, expected {sha256}")
