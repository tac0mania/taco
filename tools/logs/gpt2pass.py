"""The integer GPT-2 pass: the facts about each token that the mixer simulation needs.

Run from tools/ (on devbox; about 10 s for each window of 1,024 tokens and worker, measured):
    OPENBLAS_NUM_THREADS=4 nice -n 15 uv run python -m logs.gpt2pass [--workers 3] [names...]

For each file of `corpus.GPT2_LOGS` and `corpus.TEXT_FILES`, the tool tokenizes the 16 KiB
blocks, runs the integer GPT-2 (`logs.fastsim`, bit-equal to `convert.intsim`) on each window,
and writes tools/data/out/logs/pass/<name>.npz with, for each token:

- `W`: the sum of the u64 weights; `wa`: the weight of the actual token;
- `fa`: the frequency of the actual token in the exact plan §5.1 table (GPT-2 alone);
- `top_id`, `top_w`: the 8 tokens with the largest weights;
- `cand`, `wc`: the tokens that the match model predicts in any configuration of `CONFIGS`
  and any block size of `UNITS`, and their weights (-1 pads `cand`).

A file that already has its output is not done again.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np
from baseline.bpe import EOT_ID, Tokenizer

from logs import corpus, fastsim, freq
from logs import matchmodel as mm
from logs.tokens import Tokens, tokenize, windows

PASS_DIR = corpus.OUT_DIR / "pass"
CONFIGS = (
    mm.MatchParams((32, 16, 8, 4)),
    mm.MatchParams((16, 8, 4, 2, 1)),
    mm.MatchParams((32, 16, 8, 4, 2, 1)),
    mm.MatchParams((24, 12, 6, 3, 2)),
)
UNITS = (1, 4, 16, 64)  # block sizes in 16 KiB units: 16 KiB, 64 KiB, 256 KiB, 1 MiB
TOP = 8
CAND = 8

FM: fastsim.FastModel | None = None


def candidates(t: Tokens) -> tuple[np.ndarray, int]:
    """The distinct predicted tokens at each position, padded with -1; and the overflow count."""
    preds = [mm.predict_file(t.ids, t.block_starts, u, p)[0] for p in CONFIGS for u in UNITS]
    stack = np.sort(np.stack(preds, axis=1), axis=1)
    out = np.full((len(t.ids), CAND), -1, dtype=np.int64)
    over = 0
    for i, row in enumerate(stack):
        u = np.unique(row[row >= 0])
        if len(u) > CAND:
            over += 1
            u = u[:CAND]
        out[i, : len(u)] = u
    return out, over


def work(job: tuple[int, int, int, np.ndarray, np.ndarray]) -> tuple[int, int, dict]:
    fi, s, e, ids, cand = job
    assert FM is not None
    model_input = np.concatenate([[EOT_ID], ids[:-1]]).astype(np.int64)
    n = e - s
    res = {
        "W": np.zeros(n, np.uint64), "wa": np.zeros(n, np.uint64), "fa": np.zeros(n, np.int64),
        "top_id": np.zeros((n, TOP), np.int64), "top_w": np.zeros((n, TOP), np.uint64),
        "wc": np.zeros((n, CAND), np.uint64),
    }
    for r0, w in fastsim.forward_chunks(FM, model_input):
        rows = np.arange(r0, r0 + len(w))
        tg = ids[rows]
        local = np.arange(len(w))
        res["W"][rows] = w.sum(axis=1, dtype=np.uint64)
        res["wa"][rows] = w[local, tg]
        res["fa"][rows] = freq.entries(w, tg)
        part = np.argpartition(-w.astype(np.float64), TOP, axis=1)[:, :TOP]
        pw = w[local[:, None], part]
        o = np.lexsort((part, -pw.astype(np.float64)), axis=1)
        res["top_id"][rows] = np.take_along_axis(part, o, axis=1)
        res["top_w"][rows] = np.take_along_axis(pw, o, axis=1)
        c = cand[rows]
        res["wc"][rows] = np.where(c >= 0, w[local[:, None], np.maximum(c, 0)], 0)
    return fi, s, res


def main() -> int:
    global FM
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    PASS_DIR.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    files = corpus.load(list(corpus.GPT2_LOGS)) + corpus.load_text()
    if args.names:
        files = [f for f in files if f.name in args.names]
    todo = []
    for f in files:
        if (PASS_DIR / f"{f.name}.npz").exists():
            print(f"skip {f.name} (done)")
            continue
        t = tokenize(tok, f.name, f.data)
        cand, over = candidates(t)
        todo.append((f, t, cand, over))
        print(f"{f.name}: {len(t.ids)} tokens, {len(windows(t))} windows, "
              f"{int((cand >= 0).sum())} candidates, {over} positions over {CAND}", flush=True)
    jobs = []
    for fi, (_f, t, cand, _o) in enumerate(todo):
        for s, e in windows(t):
            jobs.append((fi, s, e, t.ids[s:e], cand[s:e]))
    print(f"{len(jobs)} windows; loading the model", flush=True)
    FM = fastsim.load()
    store: list[dict[int, dict]] = [{} for _ in todo]
    left = [len(windows(t)) for _, t, _, _ in todo]
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for fi, s, res in pool.imap_unordered(work, jobs):
            store[fi][s] = res
            left[fi] -= 1
            done += 1
            el = time.time() - t0
            if done % 10 == 0 or left[fi] == 0:
                print(f"{done}/{len(jobs)} windows, {el / 60:.1f} min, "
                      f"eta {el / done * (len(jobs) - done) / 60:.1f} min", flush=True)
            if left[fi] == 0:
                f, t, cand, over = todo[fi]
                keys = sorted(store[fi])
                arrs = {k: np.concatenate([store[fi][s][k] for s in keys])
                        for k in store[fi][keys[0]]}
                np.savez_compressed(PASS_DIR / f"{f.name}.npz", ids=t.ids,
                                    block_starts=t.block_starts, block_bytes=t.block_bytes,
                                    cand=cand, nbytes=t.nbytes, **arrs)
                store[fi] = {}
                bits = float(-np.log2(arrs["fa"] / freq.TOTAL).sum())
                print(f"saved {f.name}: GPT-2 alone {bits / t.nbytes:.4f} bits per byte "
                      f"(16 KiB blocks, table ideal)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
