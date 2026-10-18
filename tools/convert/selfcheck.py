"""The self-check of the converter (ADR 0014 point 5): compare the output with the expected hash.

`expected.toml` (committed) holds one entry for each set of inputs: the model name, the
weights (the SHA-256 of the float weights, or the seed of a generated model), the recipe and
the SHA-256 of the statistics file. Each entry gives the SHA-256 and the size of the model file
that the exact converter makes from these inputs.

The converter calculates the model file in memory, then calls `check`. If the hash is not the
expected hash, or no entry has these inputs, the converter stops with an error and does not
write the file. After a deliberate change, run the converter with `--new-hash`: it writes the
file and prints the entry to put in `expected.toml`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

EXPECTED_PATH = Path(__file__).resolve().parent / "expected.toml"


class SelfCheckError(Exception):
    """The model file is not the expected file, or no expected hash is recorded."""


@dataclass(frozen=True)
class Inputs:
    """The inputs that select one model file."""

    model: str
    weights: str
    recipe: str
    stats_sha256: str


def expected(inputs: Inputs, path: Path = EXPECTED_PATH) -> tuple[str, int] | None:
    """The expected (SHA-256, size) for these inputs, or None."""
    with path.open("rb") as f:
        table = tomllib.load(f)
    for e in table.get("model", []):
        if (e["model"], e["weights"], e["recipe"], e["stats_sha256"]) == (
            inputs.model, inputs.weights, inputs.recipe, inputs.stats_sha256
        ):
            return e["hsm_sha256"], int(e["hsm_size"])
    return None


def entry_text(inputs: Inputs, sha: str, size: int) -> str:
    """The TOML entry for `expected.toml`."""
    return (
        "[[model]]\n"
        f'model = "{inputs.model}"\n'
        f'weights = "{inputs.weights}"\n'
        f'recipe = "{inputs.recipe}"\n'
        f'stats_sha256 = "{inputs.stats_sha256}"\n'
        f'hsm_sha256 = "{sha}"\n'
        f"hsm_size = {size}\n"
    )


def check(inputs: Inputs, sha: str, size: int, new_hash: bool) -> None:
    """Raise SelfCheckError unless (sha, size) is the expected result, or `new_hash` is set.

    With `new_hash`, print the entry for `expected.toml` and return.
    """
    exp = expected(inputs)
    if exp == (sha, size):
        print(f"self-check: the SHA-256 is the expected SHA-256 ({sha[:8]}...{sha[-4:]})")
        return
    if new_hash:
        why = "no expected hash" if exp is None else f"expected {exp[0]}"
        print(f"self-check: new hash ({why}). Put this entry in {EXPECTED_PATH.name}:\n")
        print(entry_text(inputs, sha, size))
        return
    if exp is None:
        raise SelfCheckError(
            f"no expected hash for {inputs}. The output has SHA-256 {sha} ({size} bytes). "
            "If the inputs are new by decision, run again with --new-hash and record the entry."
        )
    raise SelfCheckError(
        f"the model file has SHA-256 {sha} ({size} bytes), expected {exp[0]} ({exp[1]} bytes) "
        f"for {inputs}. The file is not written. If the change is deliberate, run again with "
        "--new-hash and record the new entry."
    )
