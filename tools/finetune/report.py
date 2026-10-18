"""The Route A result tables: models and general compressors on the same bytes.

Run from tools/:
    uv run python -m finetune.report --eval DIR NAME=EVAL.tsv... [--float NAME=F.json...]

DIR is the evalset output (sets.json and bytes/). Each EVAL.tsv is the output of the HardShell
example eval_bits on DIR/windows.txt. Each F.json is the output of finetune.floateval.

The tool prints Markdown tables in bits per byte:

1. For each set: each model (the table size, which the range coder gives to about 0.02 %) and
   xz -9e, zstd -19 and zstd -19 --long=27. A compressor gets the bytes of the windows of one
   set and one system as one file (bytes/<set>/<system>.log). The last column gives xz -9e
   each window alone (about 2.5 KB): the same context as the model, which starts again at
   each window.
2. For each held-out system: the same columns.
3. The float against the integer size (ideal sizes) of each model that has a float file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from baseline.bpe import Tokenizer
from convert.report import read_int

COMPRESSORS = {
    "xz -9e": ["xz", "-9e", "-c"],
    "zstd -19": ["zstd", "-19", "-c", "-q"],
    "zstd -19 --long": ["zstd", "-19", "--long=27", "-c", "-q"],
}


def csize(cmd: list[str], path: Path) -> int:
    return len(subprocess.run([*cmd, str(path)], capture_output=True, check=True).stdout)


def pairs(items: list[str]) -> dict[str, Path]:
    return {k: Path(v) for k, v in (s.split("=", 1) for s in items)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--float", nargs="*", default=[])
    args = ap.parse_args()
    meta = json.loads((args.eval / "sets.json").read_text())
    wins = meta["windows"]
    models = {k: read_int(p) for k, p in pairs(args.models).items()}
    floats = {k: {r["label"]: r for r in json.loads(p.read_text())}
              for k, p in pairs(args.float).items()}
    for name, rows in models.items():
        for w in wins:
            r = rows[w["label"]]
            if r["bytes"] != w["bytes"]:
                raise ValueError(f"{name} {w['label']}: {r['bytes']} bytes, expected {w['bytes']}")
    sets = list(dict.fromkeys(w["set"] for w in wins))
    # xz -9e on each window alone: the compressor gets the same context as the model.
    tok = Tokenizer()
    ids = {}
    for line in (args.eval / "windows.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            label, rest = line.split("\t", 1)
            ids[label] = [int(v) for v in rest.split()]
    per_window = {}
    with tempfile.TemporaryDirectory() as td:
        for w in wins:
            path = Path(td) / "w"
            path.write_bytes(tok.decode(ids[w["label"]]))
            per_window[w["label"]] = csize(COMPRESSORS["xz -9e"], path)
    comp: dict[tuple[str, str], dict[str, int]] = {}
    for kind in sets:
        for s in dict.fromkeys(w["system"] for w in wins if w["set"] == kind):
            path = args.eval / "bytes" / kind / f"{s.replace(':', '.')}.log"
            comp[(kind, s)] = {c: csize(cmd, path) for c, cmd in COMPRESSORS.items()}

    def row(title: str, sel: list[dict]) -> str:
        b = sum(w["bytes"] for w in sel)
        cells = [f"{sum(models[m][w['label']]['table_bits'] for w in sel) / b:.3f}"
                 for m in models]
        keys = dict.fromkeys((w["set"], w["system"]) for w in sel)
        for c in COMPRESSORS:
            cells.append(f"{8 * sum(comp[k][c] for k in keys) / b:.3f}")
        cells.append(f"{8 * sum(per_window[w['label']] for w in sel) / b:.3f}")
        return f"| {title} | {len(sel)} | {b:,} | " + " | ".join(cells) + " |"

    head = ("| Set | Windows | Bytes | " + " | ".join(models) + " | " + " | ".join(COMPRESSORS)
            + " | xz -9e, each window |")
    sep = "|---|---:|---:|" + "---:|" * (len(models) + len(COMPRESSORS) + 1)
    print("Bits per byte (models: integer HardShell, table size; lower is better)\n")
    print(head)
    print(sep)
    for kind in sets:
        print(row(kind, [w for w in wins if w["set"] == kind]))
    print("\nHeld-out systems, one row for each system\n")
    print(head.replace("| Set |", "| System |"))
    print(sep)
    by_sys = defaultdict(list)
    for w in wins:
        if w["set"] == "heldout":
            by_sys[w["system"]].append(w)
    for s, sel in by_sys.items():
        print(row(s, sel))
    print("\nTraining-system tails, one row for each system\n")
    print(head.replace("| Set |", "| System |"))
    print(sep)
    by_sys = defaultdict(list)
    for w in wins:
        if w["set"] == "tail":
            by_sys[w["system"]].append(w)
    for s, sel in by_sys.items():
        print(row(s, sel))
    if floats:
        print("\nFloat (float32) against integer (HardShell), ideal size, bits per byte\n")
        print("| Model | Set | Float | Integer | Loss |")
        print("|---|---|---:|---:|---:|")
        for m, fl in floats.items():
            for kind in [*sets, "all"]:
                sel = [w for w in wins if kind in ("all", w["set"])]
                b = sum(w["bytes"] for w in sel)
                f = sum(fl[w["label"]]["ideal_bits"] for w in sel) / b
                i = sum(models[m][w["label"]]["ideal_bits"] for w in sel) / b
                print(f"| {m} | {kind} | {f:.4f} | {i:.4f} | {100 * (i / f - 1):+.2f} % |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
