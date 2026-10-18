"""Make the Route A training data from the Loghub archives of data.toml.

Run from tools/ (on devbox; the archives are large, so keep them out of the repository):
    nice uv run python -m finetune.prepare --archives DIR --work DIR [--workers N]

For each training system, the tool:

1. checks the archive (size and SHA-256 of data.toml);
2. reads the members in archive order, and keeps the first `cap` bytes (cut at a line end);
3. splits the bytes into train (first 90 %), val (next 5 %) and tail (last 5 %), at line ends;
4. tokenizes train and val with the GPT-2 BPE of tools/baseline (the Taco tokenizer).

Output in --work: tok/<system>.{train,val}.npy (uint16 token ids), tail/<system>.log (the tail
bytes, for evaluation only), calib_windows.txt (one window of 1,024 tokens from the middle of
each train part, plus the random window of convert/calibration.toml) and prepare.json (counts).

NOTE: the tool tokenizes the train text in parts of about 1 MiB that end at a line end. A
token never crosses a part boundary. Whole-file tokenization can differ at a few boundaries.
That has no effect on the training.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import sys
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path

import numpy as np

from baseline._paths import check_file
from baseline.bpe import Tokenizer
from convert.rng import SplitMix64

PACKAGE_DIR = Path(__file__).resolve().parent
PART = 1 << 20
N_CTX = 1024


def load_spec() -> dict:
    with (PACKAGE_DIR / "data.toml").open("rb") as f:
        return tomllib.load(f)


def _wanted(name: str, members: list[str]) -> bool:
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(base, m) for m in members)


def read_prefix(path: Path, members: list[str], cap: int) -> bytes:
    """The first `cap` bytes of the selected members, joined in archive order.

    A partial tar.gz (an HTTP range of the archive) ends in the middle of a member. The tool
    keeps the bytes that it can decompress before the end.
    """
    out = bytearray()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if _wanted(n, members)]
            if members and "*" not in "".join(members):
                names.sort(key=lambda n: members.index(n.rsplit("/", 1)[-1]))
            for n in names:
                with z.open(n) as f:
                    out += f.read(cap - len(out))
                if len(out) >= cap:
                    break
        return bytes(out)
    try:
        with tarfile.open(path, "r|gz") as tf:
            for m in tf:
                if not m.isfile() or not _wanted(m.name, members):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                while len(out) < cap:
                    chunk = f.read(min(PART, cap - len(out)))
                    if not chunk:
                        break
                    out += chunk
                if len(out) >= cap:
                    break
    except (EOFError, tarfile.ReadError, OSError) as e:
        print(f"  {path.name}: archive ends early ({type(e).__name__}); kept {len(out)} bytes")
    return bytes(out)


def read_system(path: Path, members: list[str], cap: int) -> bytes:
    if path.suffix != ".zip" and len(members) > 1 and "*" not in "".join(members):
        # Named tar members in the data.toml order: one pass for each member.
        out = b""
        for m in members:
            out += read_prefix(path, [m], cap - len(out))
            if len(out) >= cap:
                break
        data = out
    else:
        data = read_prefix(path, members, cap)
    end = data.rfind(b"\n") + 1
    return data[:end]


def line_cut(data: bytes, pos: int) -> int:
    """The first byte after the last line end before `pos`."""
    return data.rfind(b"\n", 0, pos) + 1


def parts(data: bytes) -> list[bytes]:
    out, i = [], 0
    while i < len(data):
        j = data.rfind(b"\n", i, i + PART) + 1 if i + PART < len(data) else len(data)
        if j <= i:
            j = min(len(data), i + PART)
        out.append(data[i:j])
        i = j
    return out


_TOK: Tokenizer | None = None


def _encode(b: bytes) -> list[int]:
    global _TOK
    if _TOK is None:
        _TOK = Tokenizer()
    return _TOK.encode(b)


def tokenize(pool, data: bytes) -> np.ndarray:
    ids: list[int] = []
    for r in pool.imap(_encode, parts(data), chunksize=1):
        ids += r
    return np.array(ids, dtype=np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archives", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    spec = load_spec()
    sp = spec["split"]
    cap = sp["cap"]
    for d in ("tok", "tail"):
        (args.work / d).mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    calib: list[tuple[str, list[int]]] = []
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for a in spec["archive"]:
            sysname = a["system"]
            path = args.archives / a["name"]
            check_file(path, a["size"], a["sha256"])
            data = read_system(path, a["members"], cap)
            n = len(data)
            t_end = line_cut(data, int(n * sp["train_fraction"]))
            v_end = line_cut(data, int(n * (sp["train_fraction"] + sp["val_fraction"])))
            train, val, tail = data[:t_end], data[t_end:v_end], data[v_end:]
            tr = tokenize(pool, train)
            va = tokenize(pool, val)
            np.save(args.work / "tok" / f"{sysname}.train.npy", tr)
            np.save(args.work / "tok" / f"{sysname}.val.npy", va)
            (args.work / "tail" / f"{sysname}.log").write_bytes(tail)
            mid = len(tr) // 2
            calib.append((f"calib/{sysname}@{mid}", tr[mid : mid + N_CTX].tolist()))
            stats[sysname] = {
                "bytes": n, "train_bytes": len(train), "val_bytes": len(val),
                "tail_bytes": len(tail), "train_tokens": len(tr), "val_tokens": len(va),
                "train_lines": train.count(b"\n"),
            }
            print(f"{sysname}: {n:,} bytes, train {len(tr):,} tokens "
                  f"({len(train) / max(1, len(tr)):.2f} bytes/token), val {len(va):,}, "
                  f"tail {len(tail):,} bytes ({time.time() - t0:.0f} s)", flush=True)
    with (PACKAGE_DIR.parent / "convert" / "calibration.toml").open("rb") as f:
        r = tomllib.load(f)["random"]
    rnd = Tokenizer().encode(SplitMix64(r["seed"]).bytes(r["bytes"]))
    for i in range(r["windows"]):
        calib.append((f"random@{r['seed']}#{i}", rnd[i * N_CTX : (i + 1) * N_CTX]))
    with (args.work / "calib_windows.txt").open("w") as f:
        f.write("# Route A calibration windows: label, tab, token ids (finetune.prepare)\n")
        for label, ids in calib:
            f.write(label + "\t" + " ".join(map(str, ids)) + "\n")
    (args.work / "prepare.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {args.work} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
