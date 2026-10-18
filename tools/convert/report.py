"""The quality table: float GPT-2 against the integer HardShell model on the same windows.

Run from tools/:
    nice uv run python -m convert.report [INT.tsv] [FLOAT.json]

The defaults are tools/data/out/eval_int_t6.tsv (the output of the HardShell example
`eval_bits`) and tools/data/out/eval_float.json (`convert.evalset`). The tool prints a Markdown
table in bits per byte. "Loss" is the relative change of the integer size against the float
size; a negative loss means that the integer model is smaller.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from baseline._paths import OUT_DIR

BINARY = {"kennedy.xls#0", "ptt5#0", "sum#0"}


def read_int(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("TOTAL"):
            continue
        f = line.split("\t")
        out[f[0]] = {"tokens": int(f[1]), "bytes": int(f[2]), "ideal_bits": float(f[3]),
                     "zero": int(f[4]), "table_bits": float(f[5])}
    return out


def main() -> int:
    ipath = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "eval_int_t6.tsv"
    fpath = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR / "eval_float.json"
    ints = read_int(ipath)
    flt = {r["label"]: r for r in json.loads(fpath.read_text())}
    print("| Window | Bytes | Float ideal | Integer ideal | Loss | Float table | Integer table "
          "| Loss | Integer w = 0 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    groups = {"16 files, not binary": [], "3 binary files": [], "All 19 files": []}
    for label, fr in flt.items():
        ir = ints[label]
        if ir["bytes"] != fr["bytes"]:
            raise ValueError(f"{label}: byte counts differ")
        b = fr["bytes"]
        fi, ii = fr["ideal_bits"] / b, ir["ideal_bits"] / b
        ft, it = fr["table_bits"] / b, ir["table_bits"] / b
        print(f"| {label[:-2]} | {b:,} | {fi:.3f} | {ii:.3f} | {100 * (ii / fi - 1):+.1f} % "
              f"| {ft:.3f} | {it:.3f} | {100 * (it / ft - 1):+.1f} % | {ir['zero']} |")
        groups["All 19 files"].append((fr, ir))
        groups["3 binary files" if label in BINARY else "16 files, not binary"].append((fr, ir))
    for name, rows in groups.items():
        b = sum(f["bytes"] for f, _ in rows)
        fi = sum(f["ideal_bits"] for f, _ in rows) / b
        ii = sum(i["ideal_bits"] for _, i in rows) / b
        ft = sum(f["table_bits"] for f, _ in rows) / b
        it = sum(i["table_bits"] for _, i in rows) / b
        z = sum(i["zero"] for _, i in rows)
        print(f"| **{name}** | {b:,} | {fi:.3f} | {ii:.3f} | {100 * (ii / fi - 1):+.1f} % "
              f"| {ft:.3f} | {it:.3f} | {100 * (it / ft - 1):+.1f} % | {z} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
