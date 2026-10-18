"""The tables of the research note from the outputs of `logs.baselines` and `logs.sim final`.

Run from tools/:
    uv run python -m logs.report > data/out/logs/report.md

All sizes are in bits per byte. Taco sizes are the exact table cost (the upper bound of
`logs.sim.exact_bits`) plus the container (est.: 88 bytes, and 9 bytes for each block for the
block table, the token count and the coder flush).
"""

from __future__ import annotations

import json
import sys

from logs import corpus

BLOCKS = (16384, 65536, 262144, 1048576)
GROUPS = {"loghub": "Loghub", "access": "Apache access", "jsonl": "nginx JSON lines",
          "text": "Text"}


def load() -> tuple[dict, dict]:
    base = {r["name"]: r for r in json.loads((corpus.OUT_DIR / "baselines.json").read_text())}
    finals = sorted(corpus.OUT_DIR.glob("final_*.json"), key=lambda p: p.stat().st_mtime)
    rows = json.loads(finals[-1].read_text())
    taco: dict = {}
    for r in rows:
        blocks = -(-r["bytes"] // r["block"])
        over = 88 + 9 * blocks
        taco.setdefault(r["name"], {})[r["block"]] = {
            "gpt2": r["gpt2_bits"] / 8 + over, "mixed": r["exact_hi"] / 8 + over,
            "mixed_ideal": r["exact_hi"] / 8, "gpt2_ideal": r["gpt2_bits"] / 8,
            "hits": r["hits"], "predicted": r["predicted"], "tokens": r["tokens"]}
    return base, taco


def bpb(x: float, n: int) -> str:
    return f"{8 * x / n:.3f}"


def main() -> int:
    base, taco = load()
    names = [n for n in taco]
    print("## Per file, 1 MiB blocks (the whole file)\n")
    print("| File | Bytes | gzip -9 | bzip2 -9 | xz -9e | zstd -19 | zstd -19 --long=27 | "
          "GPT-2 alone | GPT-2 + match | Change against the best other |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in names:
        b = base[n]
        nb = b["bytes"]
        s = b["sizes"]
        t = taco[n][1048576]
        others = {k: s[f"{k}@1048576"] for k in ("gzip -9", "bzip2 -9", "xz -9e", "zstd -19",
                                                 "zstd -19 --long=27")}
        best = min(others.values())
        print(f"| {n} | {nb:,} | " + " | ".join(bpb(v, nb) for v in others.values())
              + f" | {bpb(t['gpt2'], nb)} | **{bpb(t['mixed'], nb)}** | "
              f"{100 * (t['mixed'] / best - 1):+.1f} % |")
    print()
    for group, label in GROUPS.items():
        members = [n for n in names if base[n]["group"] == group]
        if not members:
            continue
        nb = sum(base[n]["bytes"] for n in members)
        files = "file" if len(members) == 1 else "files"
        print(f"### {label} ({len(members)} {files}): {nb:,} bytes\n")
        print("| Block | bzip2 -9 | xz -9e | zstd -19 --long=27 | GPT-2 alone | GPT-2 + match | "
              "Match against GPT-2 | against zstd --long | against xz | against bzip2 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for bs in BLOCKS:
            xz = sum(base[n]["sizes"][f"xz -9e@{bs}"] for n in members)
            bz = sum(base[n]["sizes"][f"bzip2 -9@{bs}"] for n in members)
            zl = sum(base[n]["sizes"][f"zstd -19 --long=27@{bs}"] for n in members)
            g = sum(taco[n][bs]["gpt2"] for n in members)
            m = sum(taco[n][bs]["mixed"] for n in members)
            print(f"| {bs // 1024} KiB | {bpb(bz, nb)} | {bpb(xz, nb)} | {bpb(zl, nb)} | "
                  f"{bpb(g, nb)} | **{bpb(m, nb)}** | {100 * (m / g - 1):+.1f} % | "
                  f"{100 * (m / zl - 1):+.1f} % | {100 * (m / xz - 1):+.1f} % | "
                  f"{100 * (m / bz - 1):+.1f} % |")
        print()
    print("## Text files: GPT-2 alone against GPT-2 + match (16 KiB blocks, the Taco default)\n")
    print("| File | Bytes | GPT-2 alone | GPT-2 + match | Change |")
    print("|---|---:|---:|---:|---:|")
    for n in names:
        if base[n]["group"] != "text":
            continue
        nb = base[n]["bytes"]
        t = taco[n][16384]
        print(f"| {n} | {nb:,} | {bpb(t['gpt2_ideal'], nb)} | {bpb(t['mixed_ideal'], nb)} | "
              f"{100 * (t['mixed_ideal'] / t['gpt2_ideal'] - 1):+.2f} % |")
    print("\n## Files with compressor numbers only (1 MiB blocks, the whole file)\n")
    print("| File | Bytes | xz -9e | zstd -19 --long=27 | bzip2 -9 |")
    print("|---|---:|---:|---:|---:|")
    for n, b in base.items():
        if n in taco:
            continue
        s = b["sizes"]
        print(f"| {n} | {b['bytes']:,} | {bpb(s['xz -9e@1048576'], b['bytes'])} | "
              f"{bpb(s['zstd -19 --long=27@1048576'], b['bytes'])} | "
              f"{bpb(s['bzip2 -9@1048576'], b['bytes'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
