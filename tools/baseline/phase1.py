"""Phase 1: float drift between computation paths and between machines (plan §12).

A path is one way to compute the GPT-2 logits: full sequence or KV cache, float32 or float64,
BLAS thread count, KV batch size, and the machine. For two paths, the tool reports for each
window of 1,024 tokens:

- the number of steps whose frequency table (§5.1, FREQ_BITS = 32) differs;
- the maximum absolute difference of the float64 probabilities;
- whether a stream that path A encoded decodes with path B, and the reverse. If not, the first
  wrong byte.

NOTE: the cross decode uses the tables of path B along the original tokens. Up to and
including the first wrong token, this is exactly what a real decoder on path B computes,
because its model input is the same. After the first wrong token the output is wrong anyway.

Commands:

  python -m baseline.phase1 parta [--windows N]   # Part A: paths on this machine.
  python -m baseline.phase1 dump --path full --dtype f32 --out DIR [--threads 1]
  python -m baseline.phase1 compare DIR_A DIR_B   # Any two dumps, for example Mac and devbox.

Part B (Mac against devbox), when devbox is available:

  scripts/dx baseline uv run --project tools python -m baseline.fetch
  scripts/dx baseline uv run --project tools python -m baseline.phase1 dump \\
      --path full --dtype f32 --out tools/data/out/phase1/devbox-full-f32
  (copy that directory to the Mac, then on the Mac:)
  uv run --project tools python -m baseline.phase1 dump --path full --dtype f32 \\
      --out tools/data/out/phase1/mac-full-f32
  uv run --project tools python -m baseline.phase1 compare \\
      tools/data/out/phase1/mac-full-f32 tools/data/out/phase1/devbox-full-f32
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from baseline._paths import OUT_DIR, TOOLS_DIR
from baseline.bpe import EOT_ID, Tokenizer
from baseline.corpus import load_samples
from baseline.rangecoder import Decoder, Encoder, cumulative, from_weights, probabilities
from baseline.rangecoder import weights_from_logits
from baseline.runner import ROW_CHUNK, Window, make_windows

THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS")
PHASE1_DIR = OUT_DIR / "phase1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_windows(windows_per_file: int, files: list[str] | None) -> list[Window]:
    tok = Tokenizer()
    out = []
    for s in load_samples(names=files):
        ws = make_windows(s.name, tok.encode(s.data), tok.token_bytes)
        out.extend(ws[:windows_per_file])
    return out


def blas_name() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        return str(cfg["Build Dependencies"]["blas"].get("name", "unknown"))
    except Exception:  # noqa: BLE001 - only a label in the metadata.
        return "unknown"


# dump -------------------------------------------------------------------------------------


def cmd_dump(args: argparse.Namespace) -> int:
    from baseline.gpt2_np import GPT2  # Imported here, after the thread variables are set.

    dtype = {"f32": np.float32, "f64": np.float64}[args.dtype]
    windows = select_windows(args.windows, args.files)
    if args.only is not None:
        windows = [windows[i] for i in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = GPT2(dtype)
    t_load = time.time() - t0
    t0 = time.time()
    if args.path == "full":
        for w in windows:
            np.save(out / f"{w.file}.w{w.index}.npy", model.forward(w.model_input))
    else:
        slots = args.slots or len(windows)
        for c in range(0, len(windows), slots):
            dump_kv(model, windows[c : c + slots], slots, out)
    meta = {
        "path": args.path,
        "dtype": args.dtype,
        "slots": args.slots,
        "machine": f"{platform.machine()} {platform.platform()}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "blas": blas_name(),
        "threads_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "load_s": t_load,
        "compute_s": time.time() - t0,
        "windows": [{"file": w.file, "index": w.index, "tokens": w.tokens} for w in windows],
    }
    (out / "meta.json").write_text(json.dumps(meta))
    log(f"dump {args.path}-{args.dtype}: {len(windows)} windows in {meta['compute_s']:.1f} s")
    return 0


def dump_kv(model, windows: list[Window], slots: int, out: Path) -> None:
    """KV path in lockstep: slot s holds window s. Unused slots feed <|endoftext|> at 0."""
    state = model.new_state(slots)
    n = max(len(w.tokens) for w in windows)
    logits = [np.empty((len(w.tokens), 50257), dtype=model.dtype) for w in windows]
    model_inputs = [w.model_input for w in windows]
    inputs = np.full(slots, EOT_ID, dtype=np.int64)
    for i in range(n):
        for s in range(slots):
            if s >= len(windows) or i >= len(windows[s].tokens):
                state.pos[s] = 0
                inputs[s] = EOT_ID
            else:
                inputs[s] = model_inputs[s][i]
        row = model.step(state, inputs)
        for s, w in enumerate(windows):
            if i < len(w.tokens):
                logits[s][i] = row[s]
    for w, lg in zip(windows, logits, strict=True):
        np.save(out / f"{w.file}.w{w.index}.npy", lg)


# compare ----------------------------------------------------------------------------------


def chunk_tables(lg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, _lse = weights_from_logits(lg)
    return from_weights(w), probabilities(lg)


def cross_decode(stream: bytes, logits: np.ndarray, win: Window, token_bytes) -> int | None:
    """Decode `stream` with the tables from `logits`. Return the first wrong byte, or None."""
    dec = Decoder(stream)
    done = 0
    for r in range(0, len(win.tokens), ROW_CHUNK):
        freq, _ = chunk_tables(logits[r : r + ROW_CHUNK])
        for i, f in enumerate(freq):
            want = win.tokens[r + i]
            got = dec.decode(cumulative(f))
            if got != want:
                a, b = token_bytes[got], win.data[done:]
                common = 0
                while common < min(len(a), len(b)) and a[common] == b[common]:
                    common += 1
                return done + common
            done += len(token_bytes[want])
    return None


def compare_window(la: np.ndarray, lb: np.ndarray, win: Window, token_bytes) -> dict:
    diff_steps = 0
    first_diff = None
    max_p = 0.0
    max_logit = 0.0
    enc_a, enc_b = Encoder(), Encoder()
    for r in range(0, len(win.tokens), ROW_CHUNK):
        ca, cb = la[r : r + ROW_CHUNK], lb[r : r + ROW_CHUNK]
        max_logit = max(max_logit, float(np.max(np.abs(ca.astype(np.float64) - cb))))
        fa, pa = chunk_tables(ca)
        fb, pb = chunk_tables(cb)
        rows_differ = np.any(fa != fb, axis=1)
        if first_diff is None and rows_differ.any():
            first_diff = r + int(np.argmax(rows_differ))
        diff_steps += int(rows_differ.sum())
        max_p = max(max_p, float(np.max(np.abs(pa - pb))))
        for i in range(len(fa)):
            t = win.tokens[r + i]
            enc_a.encode(int(fa[i, :t].sum()), int(fa[i, t]))
            enc_b.encode(int(fb[i, :t].sum()), int(fb[i, t]))
    sa, sb = enc_a.finish(), enc_b.finish()
    return {
        "file": win.file,
        "window": win.index,
        "steps": len(win.tokens),
        "steps_tables_differ": diff_steps,
        "first_step_tables_differ": first_diff,
        "max_abs_prob_diff": max_p,
        "max_abs_logit_diff": max_logit,
        "a_to_b_first_bad_byte": cross_decode(sa, lb, win, token_bytes),
        "b_to_a_first_bad_byte": cross_decode(sb, la, win, token_bytes),
        "bytes": len(win.data),
        "stream_a_bytes": len(sa),
        "stream_b_bytes": len(sb),
    }


def compare_dirs(dir_a: Path, dir_b: Path) -> list[dict]:
    tok = Tokenizer()
    meta_a = json.loads((dir_a / "meta.json").read_text())
    meta_b = json.loads((dir_b / "meta.json").read_text())
    by_key = {(w["file"], w["index"]): w for w in meta_b["windows"]}
    samples = {s.name: s for s in load_samples()}
    rows = []
    for wa in meta_a["windows"]:
        key = (wa["file"], wa["index"])
        wb = by_key.get(key)
        if wb is None:
            continue
        if wa["tokens"] != wb["tokens"]:
            raise ValueError(f"{key}: the two dumps have different tokens")
        ws = make_windows(wa["file"], tok.encode(samples[wa["file"]].data), tok.token_bytes)
        win = ws[wa["index"]]
        if win.tokens != wa["tokens"]:
            raise ValueError(f"{key}: the dump tokens differ from this tokenizer")
        la = np.load(dir_a / f"{key[0]}.w{key[1]}.npy")
        lb = np.load(dir_b / f"{key[0]}.w{key[1]}.npy")
        rows.append(compare_window(la, lb, win, tok.token_bytes))
    return rows


def summarise(rows: list[dict]) -> dict:
    steps = sum(r["steps"] for r in rows)
    diff = sum(r["steps_tables_differ"] for r in rows)
    return {
        "windows": len(rows),
        "steps": steps,
        "steps_tables_differ": diff,
        "fraction_tables_differ": diff / steps if steps else 0.0,
        "max_abs_prob_diff": max((r["max_abs_prob_diff"] for r in rows), default=0.0),
        "max_abs_logit_diff": max((r["max_abs_logit_diff"] for r in rows), default=0.0),
        "a_to_b_decoded_windows": sum(r["a_to_b_first_bad_byte"] is None for r in rows),
        "b_to_a_decoded_windows": sum(r["b_to_a_first_bad_byte"] is None for r in rows),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    rows = compare_dirs(Path(args.a), Path(args.b))
    result = {"a": args.a, "b": args.b, "summary": summarise(rows), "windows": rows}
    out = Path(args.out) if args.out else Path(args.b) / "compare.json"
    out.write_text(json.dumps(result, indent=1))
    print_pair(f"{args.a} vs {args.b}", result["summary"], rows)
    return 0


# part A -----------------------------------------------------------------------------------

PAIRS = {
    # name: (reference dump options, variant dump options)
    "full_vs_kv": ({}, {"path": "kv", "slots": 4}),
    "kv_batch_4_vs_1": ({"path": "kv", "slots": 4}, {"path": "kv", "slots": 1}),
    "f32_vs_f64": ({}, {"dtype": "f64"}),
    "threads_default_vs_1": ({}, {"threads": "1"}),
}


def run_dump(opts: dict, only: list[int], windows: int, files: list[str] | None, out: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    if opts.get("threads"):
        env.update({k: opts["threads"] for k in THREAD_VARS})
    cmd = [sys.executable, "-m", "baseline.phase1", "dump",
           "--path", opts.get("path", "full"), "--dtype", opts.get("dtype", "f32"),
           "--windows", str(windows), "--out", str(out), "--only", *map(str, only)]
    if files:
        cmd += ["--files", *files]
    if opts.get("slots"):
        cmd += ["--slots", str(opts["slots"])]
    subprocess.run(cmd, check=True, env=env, cwd=TOOLS_DIR)


def cmd_parta(args: argparse.Namespace) -> int:
    t_start = time.time()
    windows = select_windows(args.windows, args.files)
    pairs = args.pairs or list(PAIRS)
    log(f"part A: {len(windows)} windows, pairs {pairs}, chunks of {args.chunk}")
    work = PHASE1_DIR / "work"
    results: dict[str, list[dict]] = {p: [] for p in pairs}
    timing: dict[str, float] = {}
    for c in range(0, len(windows), args.chunk):
        only = list(range(c, min(c + args.chunk, len(windows))))
        dumps: dict[str, Path] = {}

        def get(opts: dict, only=only, dumps=dumps) -> Path:
            key = json.dumps(opts, sort_keys=True)
            if key not in dumps:
                d = work / f"d{len(dumps)}"
                shutil.rmtree(d, ignore_errors=True)
                t0 = time.time()
                run_dump(opts, only, args.windows, args.files, d)
                timing[f"dump {key}"] = timing.get(f"dump {key}", 0) + time.time() - t0
                dumps[key] = d
            return dumps[key]

        for k, p in enumerate(pairs):
            ref_opts, var_opts = PAIRS[p]
            ref_dir, var_dir = get(ref_opts), get(var_opts)
            t0 = time.time()
            rows = compare_dirs(ref_dir, var_dir)
            timing["compare"] = timing.get("compare", 0) + time.time() - t0
            results[p].extend(rows)
            s = summarise(rows)
            log(f"chunk {c // args.chunk + 1}: {p}: tables differ {s['steps_tables_differ']}"
                f"/{s['steps']}, decoded A->B {s['a_to_b_decoded_windows']}/{s['windows']}")
            # Remove the dumps that the remaining pairs do not need (disk is short).
            need = {json.dumps(o, sort_keys=True) for q in pairs[k + 1 :] for o in PAIRS[q]}
            for key in [key for key in dumps if key not in need]:
                shutil.rmtree(dumps.pop(key), ignore_errors=True)
    timing["total_s"] = time.time() - t_start
    out = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "machine": f"{platform.machine()} {platform.platform()}",
        "numpy": np.__version__,
        "blas": blas_name(),
        "windows_per_file": args.windows,
        "pairs": {p: {"summary": summarise(r), "windows": r} for p, r in results.items()},
        "timing": timing,
    }
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE1_DIR / "parta.json").write_text(json.dumps(out, indent=1))
    for p, r in results.items():
        print_pair(p, summarise(r), r)
    print("\ntiming (s):", {k: round(v, 1) for k, v in timing.items()})
    return 0


def print_pair(name: str, s: dict, rows: list[dict]) -> None:
    print(f"\n### {name}\n")
    print(f"tables differ in {s['steps_tables_differ']} of {s['steps']} steps "
          f"({100 * s['fraction_tables_differ']:.2f} %); max |dp| {s['max_abs_prob_diff']:.3e}; "
          f"max |dlogit| {s['max_abs_logit_diff']:.3e}; decoded A->B "
          f"{s['a_to_b_decoded_windows']}/{s['windows']}, B->A {s['b_to_a_decoded_windows']}"
          f"/{s['windows']}\n")
    print("| File | Window | Steps | Tables differ | First different step | max abs dp | "
          "A->B first wrong byte | B->A first wrong byte |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['file']} | {r['window']} | {r['steps']} | {r['steps_tables_differ']} | "
              f"{r['first_step_tables_differ']} | {r['max_abs_prob_diff']:.2e} | "
              f"{r['a_to_b_first_bad_byte']} | {r['b_to_a_first_bad_byte']} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="write the logits of one path for the selected windows")
    d.add_argument("--path", choices=["full", "kv"], default="full")
    d.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    d.add_argument("--windows", type=int, default=1, help="windows per file")
    d.add_argument("--files", nargs="*", default=None)
    d.add_argument("--only", nargs="*", type=int, default=None, help="indices of windows")
    d.add_argument("--slots", type=int, default=None, help="KV batch size (default: all)")
    d.add_argument("--threads", default=None,
                   help="BLAS threads; sets the variables and runs again in a new process")
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two dumps")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", default=None)
    a = sub.add_parser("parta", help="part A: compare paths on this machine")
    a.add_argument("--windows", type=int, default=1, help="windows per file")
    a.add_argument("--chunk", type=int, default=4, help="windows per dump")
    a.add_argument("--pairs", nargs="*", choices=list(PAIRS), default=None)
    a.add_argument("--files", nargs="*", default=None, help="corpus names (default: all)")
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        if args.threads and any(os.environ.get(k) != args.threads for k in THREAD_VARS):
            env = dict(os.environ, **{k: args.threads for k in THREAD_VARS})
            return subprocess.run([sys.executable, "-m", "baseline.phase1", *sys.argv[1:]],
                                  env=env, cwd=TOOLS_DIR).returncode
        return cmd_dump(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    return cmd_parta(args)


if __name__ == "__main__":
    sys.exit(main())
