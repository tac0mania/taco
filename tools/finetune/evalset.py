"""The Route A evaluation windows (research note 2026-10-19-route-a).

Run from tools/ (after `finetune.prepare`):
    nice uv run python -m finetune.evalset --work DIR

Sets (a window has 1,024 tokens; the model input is <|endoftext|> and the first 1,023 tokens):

- `heldout`: the 2k samples of the held-out systems (never in training), 16 windows each.
- `tail`: the tail of each training system (never in training), 8 windows each.
- `train2k`: the 2k samples of the training systems, 8 windows each. These lines can also be
  in the train part; the tool measures the fraction of lines that are.
- `english` and `structured`: the lines of the `heldout` and `tail` sources, divided by one
  rule: a line is English-heavy if 50 % or more of its words (split at white space) are
  alphabetic words (letters, then optional punctuation). The lines of each source stay in
  their order. 4 windows of each kind from each source.

Output in DIR/eval: windows.txt (the input of the HardShell example eval_bits), sets.json (for
each window: the set, the system and the byte count) and bytes/<set>/<system>.log (the bytes of
the windows of that set and system, joined, for xz and zstd).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np

from baseline._paths import DATA_DIR, check_file
from baseline.bpe import Tokenizer
from finetune.prepare import PACKAGE_DIR, load_spec

N_CTX = 1024
WORD = re.compile(rb"[A-Za-z]+[.,:;!?'\")]*")
COUNTS = {"heldout": 16, "tail": 8, "train2k": 8, "english": 4, "structured": 4}


def english_heavy(line: bytes) -> bool:
    words = line.split()
    if not words:
        return False
    alpha = sum(1 for w in words if WORD.fullmatch(w))
    return 2 * alpha >= len(words)


def samples() -> dict[str, Path]:
    spec = load_spec()["samples"]
    with (PACKAGE_DIR / spec["manifest"]).open("rb") as f:
        files = tomllib.load(f)["file"]
    out = {}
    for s in files:
        if s["group"] != spec["group"]:
            continue
        path = DATA_DIR / "logs" / s["name"]
        check_file(path, s["size"], s["sha256"])
        out[s["name"].removesuffix("_2k.log")] = path
    return out


def windows(tok: Tokenizer, data: bytes, count: int) -> list[list[int]]:
    ids = tok.encode(data)
    return [ids[i * N_CTX : (i + 1) * N_CTX] for i in range(count) if ids[i * N_CTX :]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    held = load_spec()["split"]["held_out"]
    tok = Tokenizer()
    two_k = samples()
    train_sys = sorted(s for s in two_k if s not in held)
    sources: dict[str, dict[str, bytes]] = {"heldout": {}, "tail": {}, "train2k": {}}
    for s in held:
        sources["heldout"][s] = two_k[s].read_bytes()
    for s in train_sys:
        sources["tail"][s] = (args.work / "tail" / f"{s}.log").read_bytes()
        sources["train2k"][s] = two_k[s].read_bytes()
    sources["english"], sources["structured"] = {}, {}
    line_share = {}
    for kind in ("heldout", "tail"):
        for s, data in sources[kind].items():
            lines = data.splitlines(keepends=True)
            en = [ln for ln in lines if english_heavy(ln)]
            st = [ln for ln in lines if not english_heavy(ln)]
            key = f"{kind}:{s}"
            line_share[key] = len(en) / max(1, len(lines))
            if en:
                sources["english"][key] = b"".join(en)
            if st:
                sources["structured"][key] = b"".join(st)
    overlap = {}
    for s in train_sys:
        ids = np.load(args.work / "tok" / f"{s}.train.npy")
        train_lines = set(tok.decode(ids.tolist()).splitlines())
        lines = sources["train2k"][s].splitlines()
        overlap[s] = sum(1 for ln in lines if ln in train_lines) / len(lines)
        print(f"train2k {s}: {100 * overlap[s]:.1f} % of the 2k lines are in the train part")
    out = args.work / "eval"
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    with (out / "windows.txt").open("w") as f:
        f.write("# Route A evaluation windows: label, tab, token ids (finetune.evalset)\n")
        for kind, srcs in sources.items():
            for s, data in srcs.items():
                ws = windows(tok, data, COUNTS[kind])
                joined = b""
                for i, w in enumerate(ws):
                    label = f"{kind}/{s.replace(':', '.')}#{i}"
                    f.write(label + "\t" + " ".join(map(str, w)) + "\n")
                    b = tok.decode(w)
                    joined += b
                    rows.append({"label": label, "set": kind, "system": s,
                                 "tokens": len(w), "bytes": len(b)})
                d = out / "bytes" / kind
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{s.replace(':', '.')}.log").write_bytes(joined)
    meta = {"windows": rows, "english_line_share": line_share, "train2k_overlap": overlap}
    (out / "sets.json").write_text(json.dumps(meta, indent=1))
    for kind in sources:
        r = [x for x in rows if x["set"] == kind]
        print(f"{kind}: {len(r)} windows, {sum(x['tokens'] for x in r):,} tokens, "
              f"{sum(x['bytes'] for x in r):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
