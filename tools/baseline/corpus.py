"""Load the capped corpus samples that the phase 0 and phase 1 tools measure."""

from __future__ import annotations

from dataclasses import dataclass

from baseline._paths import CORPUS_DIR, check_file, load_toml


@dataclass(frozen=True)
class Sample:
    """The bytes of one corpus file that the measurement uses."""

    name: str
    cls: str
    data: bytes


def load_samples(cap: int | None = None, names: list[str] | None = None) -> list[Sample]:
    """Return the samples in corpus.toml order. Each file is checked against its SHA-256."""
    spec = load_toml("corpus.toml")
    cap = spec["cap"] if cap is None else cap
    out = []
    for f in spec["file"]:
        if names and f["name"] not in names:
            continue
        path = CORPUS_DIR / f.get("local", f["name"])
        check_file(path, f["size"], f["sha256"])
        data = path.read_bytes()[f["offset"] : f["offset"] + cap]
        out.append(Sample(f["name"], f["class"], data))
    return out
