"""The facts of `logs.gpt2pass`, made by hardshell-cuda (`tools/logs/dump`) instead of NumPy.

Run from tools/ on devbox:
    uv run python -m logs.cudapass prep NAME...      # data/out/logs/cuda/NAME.in
    (cd logs/dump && cargo build --release)
    logs/dump/target/release/logs-dump data/out/gpt2.hsm data/out/logs/cuda/NAME.in \\
        data/out/logs/cuda/NAME.out
    uv run python -m logs.cudapass collect NAME...   # data/out/logs/pass/NAME.npz
    uv run python -m logs.cudapass check NAME        # compare with an existing NumPy pass

hardshell-cuda gives the weights of the CPU HardShell bit for bit (its own gate, ADR 0008).
`check` compares the CUDA facts with the facts of `logs.fastsim` for a file that both made.
"""

from __future__ import annotations

import struct
import sys

import numpy as np
from baseline.bpe import Tokenizer

from logs import corpus
from logs.gpt2pass import CAND, PASS_DIR, TOP, candidates
from logs.tokens import tokenize, windows

CUDA_DIR = corpus.OUT_DIR / "cuda"
ROW = np.dtype([("W", "<u8"), ("wa", "<u8"), ("fa", "<u8"), ("top_id", "<u4", TOP),
                ("top_w", "<u8", TOP), ("wc", "<u8", CAND)])


def files(names: list[str]):
    all_files = corpus.load(list(corpus.GPT2_LOGS)) + corpus.load_text()
    return [f for f in all_files if f.name in names]


def prep(names: list[str]) -> None:
    CUDA_DIR.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    for f in files(names):
        t = tokenize(tok, f.name, f.data)
        cand, over = candidates(t)
        ws = windows(t)
        with (CUDA_DIR / f"{f.name}.in").open("wb") as o:
            o.write(struct.pack("<I", len(ws)))
            for s, e in ws:
                o.write(struct.pack("<I", e - s))
                o.write(t.ids[s:e].astype("<u4").tobytes())
                o.write(cand[s:e].astype("<i4").tobytes())
        np.savez(CUDA_DIR / f"{f.name}.tokens.npz", ids=t.ids, block_starts=t.block_starts,
                 block_bytes=t.block_bytes, cand=cand, nbytes=t.nbytes)
        print(f"{f.name}: {len(t.ids)} tokens, {len(ws)} windows, {over} positions over {CAND}",
              flush=True)


def load_out(name: str) -> tuple[dict, np.ndarray]:
    z = np.load(CUDA_DIR / f"{name}.tokens.npz")
    rows = np.fromfile(CUDA_DIR / f"{name}.out", dtype=ROW)
    if len(rows) != len(z["ids"]):
        raise SystemExit(f"{name}: {len(rows)} rows, {len(z['ids'])} tokens")
    return dict(z), rows


def collect(names: list[str]) -> None:
    for name in names:
        z, rows = load_out(name)
        np.savez_compressed(PASS_DIR / f"{name}.npz", ids=z["ids"], block_starts=z["block_starts"],
                            block_bytes=z["block_bytes"], cand=z["cand"], nbytes=z["nbytes"],
                            W=rows["W"], wa=rows["wa"], fa=rows["fa"].astype(np.int64),
                            top_id=rows["top_id"].astype(np.int64), top_w=rows["top_w"],
                            wc=rows["wc"])
        bits = float(-np.log2(rows["fa"] / 2.0**32).sum())
        print(f"saved {name}: GPT-2 alone {bits / int(z['nbytes']):.4f} bits per byte", flush=True)


def check(name: str) -> None:
    _, rows = load_out(name)
    old = np.load(PASS_DIR / f"{name}.npz")
    for k in ("W", "wa", "fa", "top_w", "wc"):
        a = rows[k].astype(np.uint64)
        b = old[k].astype(np.uint64)
        print(f"{name} {k}: {int((a != b).sum())} of {a.size} values differ", flush=True)
    t = rows["top_id"][:, 0].astype(np.int64) != old["top_id"][:, 0]
    print(f"{name} top_id[0]: {int(t.sum())} of {len(t)} differ", flush=True)
    ba = float(-np.log2(rows["fa"] / 2.0**32).sum())
    bb = float(-np.log2(old["fa"] / 2.0**32).sum())
    print(f"{name}: GPT-2 alone {ba:.3f} bits (CUDA) against {bb:.3f} bits (fastsim)", flush=True)


def main() -> int:
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "prep":
        prep(names)
    elif cmd == "collect":
        collect(names)
    elif cmd == "check":
        check(names[0])
    else:
        print(__doc__)
        return 64
    return 0


if __name__ == "__main__":
    sys.exit(main())
