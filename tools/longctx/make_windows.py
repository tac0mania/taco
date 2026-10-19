"""Token windows for the long-context measurement (ADR 0015): one window for each block.

    uv run --project tools python tools/longctx/make_windows.py tools/data/longctx/windows

For each input and each block size B, the input is cut at the byte offsets i * B, and each
block is tokenized alone with the reference BPE (ADR 0007 §2.1). One file for each input and
B: `<name>.b<B>.txt`, one line for each block: label, a tab, the token ids.
"""

import sys
from pathlib import Path

from baseline.bpe import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "tools/data/corpus"
L = ROOT / "tools/data/logs"
MIB = 1 << 20
INPUTS = {
    "alice29": (C / "alice29.txt", 0, None),
    "lcet10": (C / "lcet10.txt", 0, None),
    "cp_html": (C / "cp.html", 0, None),
    "apache_1m": (L / "apache_access.log", 0, MIB),
    "chinese_512k": (C / "pg24264.txt", 0, MIB // 2),
}
BLOCKS = [16 << 10, 64 << 10, 256 << 10, MIB]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    for name, (path, off, n) in INPUTS.items():
        data = path.read_bytes()[off:]
        if n is not None:
            data = data[:n]
        for b in BLOCKS:
            lines = []
            total = 0
            for i in range(0, len(data), b):
                block = data[i : i + b]
                ids = tok.encode(block)
                assert tok.decode(ids) == block
                total += len(ids)
                lines.append(f"{name}.{i // b}\t" + " ".join(map(str, ids)))
            (out / f"{name}.b{b}.txt").write_text("\n".join(lines) + "\n")
            print(f"{name} B={b}: {len(data)} bytes, {len(lines)} blocks, {total} tokens", flush=True)


if __name__ == "__main__":
    main()
