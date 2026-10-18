"""Phase 0: bits per byte of general compressors and of float GPT-2 on the corpus (plan §12).

Run: nice -n 10 uv run --project tools python -m baseline.phase0 [--kv-windows N|all]

For each corpus sample (the first `cap` bytes, see corpus.toml) the tool reports:

1. gzip -9, bz2 -9, xz (preset 9 | extreme) and zstd -19 sizes.
2. GPT-2 ideal size: the sum of -log2 p of the actual tokens, full-sequence path, float32.
3. GPT-2 coded size: the Python range coder (plan §5) with §5.1 tables, FREQ_BITS = 32.
4. Decode checks:
   a. coder check: decode each stream with the tables of the encoder (no model);
   b. mixed check: encode with the full-sequence path, decode with the KV-cache path (a real
      decode, model in the loop);
   c. same-path check: encode and decode with the KV-cache path, for the first N windows of
      each sample (the KV path costs about 5 ms for each token on this Mac, measured).

The results go to tools/data/out/phase0.json and to stdout as Markdown tables.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import os
import platform
import sys
import time

import numpy as np
import zstandard

from baseline._paths import OUT_DIR, load_toml
from baseline.bpe import Tokenizer
from baseline.corpus import load_samples
from baseline.gpt2_np import GPT2
from baseline.rangecoder import Decoder, cumulative
from baseline.runner import (
    DecodeJob,
    EncodeJob,
    Window,
    encode_with_tables,
    full_logits,
    iter_tables,
    make_windows,
    run_kv,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classic_sizes(data: bytes) -> dict[str, int]:
    """Sizes of the general compressors. Each result is decompressed and checked."""
    out = {
        "gzip-9": gzip.compress(data, compresslevel=9, mtime=0),
        "bz2-9": bz2.compress(data, 9),
        "xz-9e": lzma.compress(data, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME),
        "zstd-19": zstandard.ZstdCompressor(level=19).compress(data),
    }
    assert gzip.decompress(out["gzip-9"]) == data
    assert bz2.decompress(out["bz2-9"]) == data
    assert lzma.decompress(out["xz-9e"]) == data
    assert zstandard.ZstdDecompressor().decompress(out["zstd-19"]) == data
    return {k: len(v) for k, v in out.items()}


def coder_check(win: Window, stream: bytes, freq_rows: np.ndarray) -> bool:
    """Decode a stream with the tables that encoded it. True if all tokens come back."""
    dec = Decoder(stream)
    for i, tok in enumerate(win.tokens):
        if dec.decode(cumulative(freq_rows[i].astype(np.uint64))) != tok:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap", type=int, default=None, help="bytes per sample (default: corpus.toml)")
    ap.add_argument("--files", nargs="*", default=None, help="corpus names (default: all)")
    ap.add_argument("--kv-windows", default="1", help="windows per sample for check c, or 'all'")
    ap.add_argument("--slots", type=int, default=12, help="batch size of the KV path")
    ap.add_argument("--out", default=str(OUT_DIR / "phase0.json"))
    args = ap.parse_args(argv)

    t_start = time.time()
    timing: dict[str, float] = {}
    tok = Tokenizer()
    samples = load_samples(args.cap, args.files)
    cap = args.cap or load_toml("corpus.toml")["cap"]
    log(f"{len(samples)} samples, cap {cap} bytes")

    rows: dict[str, dict] = {}
    windows: list[Window] = []
    t0 = time.time()
    for s in samples:
        ids = tok.encode(s.data)
        assert tok.decode(ids) == s.data, f"tokenizer round trip failed: {s.name}"
        ws = make_windows(s.name, ids, tok.token_bytes)
        windows.extend(ws)
        rows[s.name] = {
            "class": s.cls,
            "bytes": len(s.data),
            "tokens": len(ids),
            "windows": len(ws),
            "classic": classic_sizes(s.data),
        }
    timing["tokenize_and_classic_s"] = time.time() - t0
    n_tokens = sum(len(w.tokens) for w in windows)
    log(f"{len(windows)} windows, {n_tokens} tokens")

    t0 = time.time()
    model = GPT2(np.float32)
    timing["load_model_s"] = time.time() - t0

    kv_n = len(windows) if args.kv_windows == "all" else int(args.kv_windows)

    # 1. Full-sequence path: ideal size, coded size, coder check.
    t0 = time.time()
    full: dict[tuple[str, int], dict] = {}
    for k, win in enumerate(windows):
        logits = full_logits(model, win)
        freq_rows = np.empty((len(win.tokens), logits.shape[1]), dtype=np.uint32)

        def keep(tables, rows=freq_rows):
            for i, t in enumerate(tables):
                rows[i] = t[0]
                yield t

        res = encode_with_tables(win, keep(iter_tables(logits)), hashes=win.index < kv_n)
        del logits
        ok = coder_check(win, res.stream, freq_rows)
        full[(win.file, win.index)] = {
            "ideal_bits": res.ideal_bits,
            "table_bits": res.table_bits,
            "floor_tokens": res.floor_tokens,
            "floor_ideal_bits": res.floor_ideal_bits,
            "stream": res.stream,
            "coder_ok": ok,
            "hashes": res.table_hashes,
        }
        if not ok:
            log(f"CODER CHECK FAILED: {win.file} window {win.index}")
        if (k + 1) % 10 == 0 or k + 1 == len(windows):
            el = time.time() - t0
            log(f"full path: {k + 1}/{len(windows)} windows, {el:.0f} s, "
                f"est. {el / (k + 1) * (len(windows) - k - 1):.0f} s left")
    timing["full_path_encode_and_coder_check_s"] = time.time() - t0

    # 2. Mixed check: full-sequence encode, KV-cache decode.
    t0 = time.time()
    mixed = [DecodeJob(w, full[(w.file, w.index)]["stream"], tok.token_bytes) for w in windows]
    run_kv(model, mixed, args.slots,
           lambda st, fin: log(f"mixed decode: step {st}, {fin}/{len(mixed)} windows done"))
    timing["mixed_decode_s"] = time.time() - t0

    # 3. Same-path check: KV encode, KV decode, first N windows of each sample.
    t0 = time.time()
    kv_wins = [w for w in windows if w.index < kv_n]
    enc_jobs = [EncodeJob(w) for w in kv_wins]
    n_kv = sum(len(w.tokens) for w in kv_wins)
    log(f"KV encode: {len(kv_wins)} windows, {n_kv} tokens")
    run_kv(model, enc_jobs, args.slots,
           lambda st, fin: log(f"KV encode: step {st}, {fin}/{len(enc_jobs)} windows done"))
    timing["kv_encode_s"] = time.time() - t0
    t0 = time.time()
    dec_jobs = [DecodeJob(j.win, j.res.stream, tok.token_bytes) for j in enc_jobs]
    run_kv(model, dec_jobs, args.slots,
           lambda st, fin: log(f"KV decode: step {st}, {fin}/{len(dec_jobs)} windows done"))
    timing["kv_decode_s"] = time.time() - t0

    # Collect the results for each sample.
    for name, row in rows.items():
        fw = [full[(w.file, w.index)] for w in windows if w.file == name]
        mj = [j for j in mixed if j.win.file == name]
        row["gpt2_ideal_bytes"] = sum(f["ideal_bits"] for f in fw) / 8
        row["gpt2_coded_bytes"] = sum(len(f["stream"]) for f in fw)
        row["gpt2_table_ideal_bytes"] = sum(f["table_bits"] for f in fw) / 8
        row["floor_tokens"] = sum(f["floor_tokens"] for f in fw)
        row["floor_ideal_bytes"] = sum(f["floor_ideal_bits"] for f in fw) / 8
        row["coded_sha256"] = hashlib.sha256(b"".join(f["stream"] for f in fw)).hexdigest()
        row["coder_check_ok"] = all(f["coder_ok"] for f in fw)
        bad = [j for j in mj if j.first_bad_byte is not None]
        row["mixed_ok_windows"] = len(mj) - len(bad)
        row["mixed_first_bad_byte"] = (
            None if not bad else bad[0].win.byte_offset + bad[0].first_bad_byte
        )
        row["mixed_first_bad_token_in_window"] = [
            (j.win.index, len(j.out) - 1) for j in bad
        ]
        ej = [j for j in enc_jobs if j.win.file == name]
        dj = [j for j in dec_jobs if j.win.file == name]
        row["kv_windows"] = len(ej)
        row["kv_coded_bytes"] = sum(len(j.res.stream) for j in ej)
        row["kv_ideal_bytes"] = sum(j.res.ideal_bits for j in ej) / 8
        row["kv_full_coded_bytes_same_windows"] = sum(
            len(full[(j.win.file, j.win.index)]["stream"]) for j in ej
        )
        row["kv_decode_ok"] = all(j.first_bad_byte is None for j in dj) and all(
            b"".join(tok.token_bytes[t] for t in j.out) == j.win.data for j in dj
        )
        steps = diff = 0
        for j in ej:
            h_full = full[(j.win.file, j.win.index)]["hashes"]
            steps += len(h_full)
            diff += sum(a != b for a, b in zip(h_full, j.res.table_hashes, strict=True))
        row["kv_vs_full_table_steps"] = steps
        row["kv_vs_full_tables_differ"] = diff

    timing["total_s"] = time.time() - t_start
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "machine": f"{platform.machine()} {platform.platform()}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cap": cap,
        "kv_windows": args.kv_windows,
        "slots": args.slots,
        "threads_env": {k: os.environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "timing": timing,
        "samples": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print_tables(result)
    return 0


def bpb(n: float, total: int) -> str:
    return f"{8 * n / total:.3f}"


def print_tables(r: dict) -> None:
    s = r["samples"]
    print("\n| File | Class | Bytes | Tokens | gzip -9 | bz2 -9 | xz -9e | zstd -19 | GPT-2 ideal | GPT-2 table ideal | GPT-2 coded | tokens > 32 bits |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = {"bytes": 0, "gzip-9": 0, "bz2-9": 0, "xz-9e": 0, "zstd-19": 0, "ideal": 0.0,
           "table": 0.0, "coded": 0, "tokens": 0, "floor": 0}
    for name, row in s.items():
        n = row["bytes"]
        c = row["classic"]
        print(f"| {name} | {row['class']} | {n} | {row['tokens']} | " + " | ".join(
            bpb(c[k], n) for k in ("gzip-9", "bz2-9", "xz-9e", "zstd-19"))
            + f" | {bpb(row['gpt2_ideal_bytes'], n)} | {bpb(row['gpt2_table_ideal_bytes'], n)}"
            f" | {bpb(row['gpt2_coded_bytes'], n)} | {row['floor_tokens']} |")
        tot["bytes"] += n
        tot["tokens"] += row["tokens"]
        for k in ("gzip-9", "bz2-9", "xz-9e", "zstd-19"):
            tot[k] += c[k]
        tot["ideal"] += row["gpt2_ideal_bytes"]
        tot["coded"] += row["gpt2_coded_bytes"]
        tot["table"] += row["gpt2_table_ideal_bytes"]
        tot["floor"] += row["floor_tokens"]
    n = tot["bytes"]
    print(f"| **All** | | {n} | {tot['tokens']} | " + " | ".join(
        bpb(tot[k], n) for k in ("gzip-9", "bz2-9", "xz-9e", "zstd-19"))
        + f" | {bpb(tot['ideal'], n)} | {bpb(tot['table'], n)} | {bpb(tot['coded'], n)}"
        f" | {tot['floor']} |")
    print(f"\ncoded - table ideal: {tot['coded'] - tot['table']:.1f} bytes "
          f"({100 * (tot['coded'] / tot['table'] - 1):.4f} %); "
          f"coded - float ideal: {tot['coded'] - tot['ideal']:.1f} bytes")
    print("\n| File | Coder check | Mixed: windows OK | Mixed: first wrong byte | KV windows | KV decode | KV coded | full coded (same windows) | tables differ (KV vs full) |")
    print("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for name, row in s.items():
        print(f"| {name} | {'pass' if row['coder_check_ok'] else 'FAIL'} | "
              f"{row['mixed_ok_windows']}/{row['windows']} | {row['mixed_first_bad_byte']} | "
              f"{row['kv_windows']} | {'pass' if row['kv_decode_ok'] else 'FAIL'} | "
              f"{row['kv_coded_bytes']} | {row['kv_full_coded_bytes_same_windows']} | "
              f"{row['kv_vs_full_tables_differ']}/{row['kv_vs_full_table_steps']} |")
    print("\ntiming (s):", {k: round(v, 1) for k, v in r["timing"].items()})


if __name__ == "__main__":
    sys.exit(main())
