"""General compressors on the log corpus, for each block size (ADR 0010, research note).

Run from tools/:
    uv run python -m logs.baselines

The tool compresses each block of each file alone with the command-line programs (gzip -9,
bzip2 -9, xz -9e, zstd -19, zstd -19 --long=27) and adds the sizes. Each output is
decompressed and compared with the input. Every file of the measurement is at most 1 MiB, so
the 1 MiB block is the whole file. Output: tools/data/out/logs/baselines.json.
"""

from __future__ import annotations

import json
import subprocess
import sys

from logs import corpus

COMPRESSORS = {
    "gzip -9": (["gzip", "-9", "-c", "-n"], ["gzip", "-d", "-c"]),
    "bzip2 -9": (["bzip2", "-9", "-c"], ["bzip2", "-d", "-c"]),
    "xz -9e": (["xz", "-9e", "-c"], ["xz", "-d", "-c"]),
    "zstd -19": (["zstd", "-19", "-c", "-q"], ["zstd", "-d", "-c", "-q"]),
    "zstd -19 --long=27": (["zstd", "-19", "--long=27", "-c", "-q"],
                           ["zstd", "-d", "--long=27", "-c", "-q"]),
}
BLOCKS = (16384, 65536, 262144, 1048576)


def size_of(data: bytes, comp: list[str], dec: list[str]) -> int:
    out = subprocess.run(comp, input=data, capture_output=True, check=True).stdout
    back = subprocess.run(dec, input=out, capture_output=True, check=True).stdout
    if back != data:
        raise RuntimeError(f"{comp[0]} did not restore the input")
    return len(out)


def main() -> int:
    files = corpus.load() + corpus.load_text()
    rows = []
    for f in files:
        row = {"name": f.name, "group": f.group, "bytes": len(f.data), "sizes": {}}
        for name, (comp, dec) in COMPRESSORS.items():
            for bs in BLOCKS:
                total = sum(size_of(f.data[o : o + bs], comp, dec)
                            for o in range(0, len(f.data), bs))
                row["sizes"][f"{name}@{bs}"] = total
        rows.append(row)
        s = row["sizes"]
        print(f"{f.name:20s} {len(f.data):8d}  xz@16K {s['xz -9e@16384']:7d}  "
              f"zstd--long@1M {s['zstd -19 --long=27@1048576']:7d}", flush=True)
    corpus.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (corpus.OUT_DIR / "baselines.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
