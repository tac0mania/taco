"""Head-to-head benchmark: Taco against general compressors (README table).

The script measures, for each compressor and each input file:

- the compressed size (bits per byte, headers included),
- the compression time and the decompression time (wall clock),
- the peak memory (maximum resident set size, from GNU time),
- the 1-minute load average of the machine before each timing.

Each output is decompressed and compared with the input byte for byte. A difference stops the
run. The script uses the Python standard library only.

Two input sets:

- ``samples``: the phase 0 samples (tools/baseline/corpus.toml: ``offset`` and the 32 KiB cap),
  and the first 32 KiB of four more Loghub 2k logs (BGL, Linux, OpenSSH, Spark).
- ``whole``: the whole corpus files and the six whole Loghub 2k logs.

Commands (run on devbox, 4 cores, lowest priority; see docs/research/2026-10-19-benchmark.md):

  scripts/dx bench "taskset -c 10-13 nice -n 19 python3 -I tools/bench/run.py measure \\
      --work ~/bench/work --out ~/bench/results.json \\
      --paq8px ~/bench/src/paq8px/build/paq8px \\
      --taco ~/bench/target/release/taco --model tools/data/out/gpt2.hsm"
  python3 -I tools/bench/run.py table ~/bench/results.json [more results files] > table.md

``measure`` appends to the results file and skips (set, compressor, file) entries that the file
already holds, so a stopped run continues where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_TOML = REPO / "tools" / "baseline" / "corpus.toml"
CORPUS_DIR = REPO / "tools" / "data" / "corpus"
LOG_DIRS = (REPO / "tools" / "data" / "logs", REPO / "tools" / "data" / "bench")
CAP = 32768

# The six Loghub 2k samples. HDFS and Apache are also in corpus.toml. Sizes and SHA-256 were
# measured on 2026-10-18 (the same values are in tools/logs/logs.toml).
LOGS = {
    "HDFS_2k.log": (287848, "7c967000980c086ed55fa6544ba4f05fe66d44622795e890c68caf8bbb635035"),
    "Apache_2k.log": (171239, "c7efa3eb686e3a96bd2f8f4457b2a7887e9cf2f3649327f1b4e87af841363ce8"),
    "BGL_2k.log": (317150, "2a819ea540909db682005c9cf948387a40729b5c2e9f19d430e29ce704825496"),
    "Linux_2k.log": (216485, "b3e20bc1afe732ab1bf3ed1de4bf9c809e4194e02f7dea911d918e5342e8e173"),
    "OpenSSH_2k.log": (225216, "1e4912727fa88245113d41b16a0cd25ceadba7f931e1c406542885b91254264f"),
    "Spark_2k.log": (196268, "2e8b9a37fc5c238253e0b8e18a8bd5e489671def91767ae1192d28c8e1f95901"),
}
LOGHUB_URL = "https://raw.githubusercontent.com/logpai/loghub/master/{d}/{n}"

CLASSES = {
    "English": ["alice29.txt", "asyoulik.txt", "lcet10.txt", "plrabn12.txt"],
    "Other languages": ["pg22367.txt", "pg4650.txt", "pg24264.txt"],
    "Code": ["fields.c", "grammar.lsp", "textwrap.py", "option.rs", "xargs.1"],
    "HTML": ["cp.html"],
    "Logs": list(LOGS),
    "JSON": ["twitter.json"],
    "Binary": ["kennedy.xls", "ptt5", "sum"],
}
CLASS_OF = {f: c for c, fs in CLASSES.items() for f in fs}

# name: (compress argv, decompress argv, output suffix). {i} input, {o} compressed, {r} restored.
# stdout ">" means that the program writes to standard output and the script stores it.
COMPRESSORS: dict[str, tuple[list[str], list[str], str]] = {
    "gzip -9": (["gzip", "-9", "-n", "-c", "{i}", ">"], ["gzip", "-d", "-c", "{o}", ">"], ".gz"),
    "bzip2 -9": (["bzip2", "-9", "-c", "{i}", ">"], ["bzip2", "-d", "-c", "{o}", ">"], ".bz2"),
    "xz -9e": (["xz", "-9e", "-T1", "-c", "{i}", ">"], ["xz", "-d", "-c", "{o}", ">"], ".xz"),
    "zstd -19": (["zstd", "-19", "-q", "-c", "{i}", ">"], ["zstd", "-d", "-q", "-c", "{o}", ">"],
                 ".zst"),
    "zstd --ultra -22 --long=27": (
        ["zstd", "--ultra", "-22", "--long=27", "-q", "-c", "{i}", ">"],
        ["zstd", "-d", "--long=27", "-q", "-c", "{o}", ">"], ".zst"),
    "brotli -q 11 -w 24": (["brotli", "-q", "11", "-w", "24", "-c", "{i}", ">"],
                           ["brotli", "-d", "-c", "{o}", ">"], ".br"),
    "7z PPMd o16 256m": (
        ["7z", "a", "-t7z", "-m0=PPMd:mem=256m:o=16", "-bd", "-bso0", "-bsp0", "{o}", "{i}"],
        ["7z", "e", "-so", "-bd", "{o}", ">"], ".7z"),
    "paq8px -8": (["{paq8px}", "-8", "{i}", "{o}"], ["{paq8px}", "-d", "{o}", "{r}"], ".paq8px"),
    "taco": (["{taco}", "fold", "{i}", "-o", "{o}", "--model", "{model}", "--threads", "{threads}",
              "-f"],
             ["{taco}", "unfold", "{o}", "-o", "{r}", "--model", "{model}", "--threads",
              "{threads}", "-f"], ".taco"),
}
FAST = {"gzip -9", "bzip2 -9", "xz -9e", "zstd -19", "zstd --ultra -22 --long=27",
        "brotli -q 11 -w 24", "7z PPMd o16 256m"}
SLOW_SET = {"paq8px -8", "taco"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def find_log(name: str, size: int, digest: str, fetch_dir: Path) -> Path:
    for d in (*LOG_DIRS, CORPUS_DIR):
        p = d / name
        if p.is_file() and p.stat().st_size == size and sha256(p) == digest:
            return p
    fetch_dir.mkdir(parents=True, exist_ok=True)
    p = fetch_dir / name
    url = LOGHUB_URL.format(d=name.split("_")[0], n=name)
    log(f"download {url}")
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 - fixed Loghub URL.
        p.write_bytes(r.read())
    if p.stat().st_size != size or sha256(p) != digest:
        raise SystemExit(f"{name}: size or SHA-256 differs from the manifest")
    return p


def prepare(work: Path) -> dict[str, list[Path]]:
    """Write the input files of each set to work/in/<set>/ and return them in a fixed order."""
    spec = tomllib.loads(CORPUS_TOML.read_text())
    sets: dict[str, list[Path]] = {"samples": [], "whole": []}
    for s in sets:
        (work / "in" / s).mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, Path, int]] = []
    for f in spec["file"]:
        p = CORPUS_DIR / f.get("local", f["name"])
        if p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
            raise SystemExit(f"{p}: size or SHA-256 differs from corpus.toml")
        sources.append((f["name"], p, f["offset"]))
    known = {n for n, _, _ in sources}
    for n, (size, digest) in LOGS.items():
        if n not in known:
            sources.append((n, find_log(n, size, digest, REPO / "tools" / "data" / "bench"), 0))
    for name, path, offset in sources:
        data = path.read_bytes()
        for s, part in (("samples", data[offset: offset + CAP]), ("whole", data)):
            out = work / "in" / s / name
            if not out.exists() or out.read_bytes() != part:
                out.write_bytes(part)
            sets[s].append(out)
    return sets


def fill(argv: list[str], ctx: dict[str, str]) -> list[str]:
    return [a.format(**ctx) for a in argv]


def timed(argv: list[str], stdout_path: Path | None, tdir: Path) -> tuple[float, int]:
    """Run argv once. Return the wall time in seconds and the peak RSS in KiB."""
    tf = tdir / "time.txt"
    cmd = ["/usr/bin/time", "-f", "%M", "-o", str(tf), *argv]
    out = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL  # noqa: SIM115
    try:
        t0 = time.perf_counter()
        r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, check=False)
        wall = time.perf_counter() - t0
    finally:
        if stdout_path:
            out.close()
    if r.returncode != 0:
        raise RuntimeError(f"{argv[0]} failed ({r.returncode}): {r.stderr[-400:]!r}")
    return wall, int(tf.read_text().split()[-1])


def one(comp: str, src: Path, tdir: Path, ctx: dict[str, str], repeat: int) -> dict:
    """Compress and decompress one file. Keep the smallest wall time of `repeat` runs."""
    c_argv, d_argv, suffix = COMPRESSORS[comp]
    tdir.mkdir(parents=True, exist_ok=True)
    o = tdir / (src.name + suffix)
    r = tdir / (src.name + ".restored")
    ctx = {**ctx, "i": str(src), "o": str(o), "r": str(r)}
    rec: dict = {"compressor": comp, "file": src.name, "bytes": src.stat().st_size}
    best_c = best_d = None
    for _ in range(repeat):
        for p in (o, r):
            p.unlink(missing_ok=True)
        la_c = load1()
        argv = fill(c_argv, ctx)
        tc, mc = timed([a for a in argv if a != ">"], o if argv[-1] == ">" else None, tdir)
        la_d = load1()
        argv = fill(d_argv, ctx)
        td, md = timed([a for a in argv if a != ">"], r if argv[-1] == ">" else None, tdir)
        if r.read_bytes() != src.read_bytes():
            raise SystemExit(f"{comp}: {src.name} did not restore byte for byte")
        if best_c is None or tc < best_c[0]:
            best_c = (tc, mc, la_c)
        if best_d is None or td < best_d[0]:
            best_d = (td, md, la_d)
        rec["compressed"] = o.stat().st_size
    rec.update(c_s=best_c[0], c_rss_kib=best_c[1], c_load1=best_c[2],
               d_s=best_d[0], d_rss_kib=best_d[1], d_load1=best_d[2], repeat=repeat,
               restored="byte for byte")
    for p in (o, r):
        p.unlink(missing_ok=True)
    return rec


def versions(ctx: dict[str, str]) -> dict[str, str]:
    def first(argv: list[str]) -> str:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, check=False)
        except OSError as e:
            return f"unavailable: {e}"
        lines = [x for x in (r.stdout + r.stderr).splitlines() if x.strip()]
        return lines[0].strip() if lines else ""

    v = {
        "gzip": first(["gzip", "--version"]), "bzip2": first(["sh", "-c", "bzip2 -h 2>&1"]),
        "xz": first(["xz", "--version"]), "zstd": first(["zstd", "--version"]),
        "brotli": first(["brotli", "--version"]),
        "7z": first(["sh", "-c", "7z | sed -n 2p"]),
        "paq8px": first([ctx["paq8px"]]) if ctx.get("paq8px") else "",
        "python": sys.version.split()[0], "machine": platform.platform(),
        "cpu": next((x.split(":", 1)[1].strip() for x in
                     Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")),
                    ""),
        "affinity": ",".join(map(str, sorted(os.sched_getaffinity(0)))),
        "nice": str(os.nice(0)),
    }
    if ctx.get("model"):
        v["model_sha256"] = sha256(Path(ctx["model"]))
    return v


def cmd_measure(a: argparse.Namespace) -> int:
    work = Path(a.work).expanduser()
    out = Path(a.out).expanduser()
    sets = prepare(work)
    ctx = {"paq8px": a.paq8px or "", "taco": a.taco or "", "model": a.model or "",
           "threads": str(a.taco_threads)}
    res = json.loads(out.read_text()) if out.exists() else {"runs": [], "env": []}
    res["env"].append({"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), **versions(ctx)})
    done = {(r["set"], r["compressor"], r["file"]) for r in res["runs"]}

    def save() -> None:
        out.write_text(json.dumps(res, indent=1))

    comps = a.compressors or [c for c in COMPRESSORS
                              if (c != "paq8px -8" or a.paq8px) and (c != "taco" or a.taco)]
    for s in a.sets:
        for comp in comps:
            files = sets[s]
            if comp == "taco" and s == "whole":
                files = [f for f in files if f.stat().st_size <= a.taco_whole_max
                         or f.name in (a.taco_whole or [])]
            todo = [f for f in files if (s, comp, f.name) not in done]
            if comp in SLOW_SET and s == "whole":
                # A whole file that is not longer than the cap is the same input as its sample.
                prev = {r["file"]: r for r in res["runs"]
                        if r["set"] == "samples" and r["compressor"] == comp}
                for f in list(todo):
                    p = prev.get(f.name)
                    if p and p["bytes"] == f.stat().st_size and \
                            (work / "in" / "samples" / f.name).read_bytes() == f.read_bytes():
                        res["runs"].append({**p, "set": "whole", "reused_from": "samples"})
                        todo.remove(f)
                save()
            if not todo:
                continue
            log(f"{s} / {comp}: {len(todo)} files, load {load1():.2f}")
            repeat = a.repeat if comp in FAST else 1
            jobs = {"taco": a.taco_jobs, "paq8px -8": a.paq8px_jobs}.get(comp, 1)
            t0 = time.perf_counter()

            def task(i_f: tuple[int, Path], comp: str = comp, s: str = s,
                     repeat: int = repeat) -> dict:
                i, f = i_f
                rec = one(comp, f, work / "tmp" / str(i % 64), ctx, repeat)
                rec["set"] = s
                rec["class"] = CLASS_OF[f.name]
                return rec

            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for rec in ex.map(task, enumerate(todo)):
                    rec["parallel_jobs"] = jobs
                    res["runs"].append(rec)
                    save()
                    log(f"  {rec['file']}: {rec['bytes']} -> {rec['compressed']} "
                        f"({8 * rec['compressed'] / max(rec['bytes'], 1):.3f} bpb), "
                        f"c {rec['c_s']:.2f} s, d {rec['d_s']:.2f} s, "
                        f"rss {rec['c_rss_kib'] // 1024}/{rec['d_rss_kib'] // 1024} MiB, "
                        f"load {rec['c_load1']:.1f}")
            res.setdefault("passes", []).append(
                {"set": s, "compressor": comp, "files": len(todo), "jobs": jobs,
                 "wall_s": time.perf_counter() - t0, "load1_end": load1()})
            save()
    shutil.rmtree(work / "tmp", ignore_errors=True)
    log("done")
    return 0


# Tables --------------------------------------------------------------------------------------

ORDER = list(COMPRESSORS)


def bpb(rows: list[dict]) -> float:
    return 8 * sum(r["compressed"] for r in rows) / sum(r["bytes"] for r in rows)


def table_sizes(runs: list[dict], s: str, names: list[str]) -> list[str]:
    comps = [c for c in ORDER if any(r["compressor"] == c and r["set"] == s for r in runs)]
    out = ["| File | Bytes | " + " | ".join(comps) + " |",
           "|---|---:|" + "---:|" * len(comps)]
    by = {(r["compressor"], r["file"]): r for r in runs if r["set"] == s}

    def line(label: str, files: list[str]) -> str | None:
        cells, vals = [], {}
        for c in comps:
            rows = [by[(c, f)] for f in files if (c, f) in by]
            if len(rows) == len(files) and rows:
                vals[c] = bpb(rows)
        best = min(vals.values()) if vals else None
        nbytes = sum(by[(comps[0], f)]["bytes"] for f in files if (comps[0], f) in by)
        if not nbytes:
            return None
        for c in comps:
            if c in vals:
                v = f"{vals[c]:.3f}"
                cells.append(f"**{v}**" if vals[c] == best else v)
            else:
                cells.append("n.m.")
        return f"| {label} | {nbytes:,} | " + " | ".join(cells) + " |"

    for f in names:
        ln = line(f, [f])
        if ln:
            out.append(ln)
    if len(names) > 1:
        ln = line("**Class**", names)
        if ln:
            out.append(ln)
    return out


def kbs(nbytes: int, seconds: float) -> str:
    v = nbytes / seconds / 1000
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def pct(a: float, b: float) -> str:
    """The change from b to a, in percent, with a sign."""
    return f"{100 * (a / b - 1):+.1f} %".replace("-", "−")


def table_compare(runs: list[dict], s: str) -> list[str]:
    """Taco against the best general compressor and paq8px, for each class and overall.

    A class that Taco measured only in part gets a row for the files that Taco measured. Every
    compressor in a row uses the same files.
    """
    by = {(r["compressor"], r["file"]): r for r in runs if r["set"] == s}
    allf = [f for fs in CLASSES.values() for f in fs]
    text = [f for f in allf if CLASS_OF[f] != "Binary"]
    groups = [*CLASSES.items(), ("All, not binary", text), ("All files", allf)]
    out = ["| Group | Files | Bytes | Best general compressor | paq8px -8 | Taco | "
           "Taco against the best general | Taco against paq8px |",
           "|---|---:|---:|---|---:|---:|---:|---:|"]
    for label, files in groups:
        have = [f for f in files if ("taco", f) in by]
        if not have:
            continue
        if len(have) < len(files) and label.startswith("All"):
            label = {"All files": "**All files that Taco measured**",
                     "All, not binary": "**All that Taco measured, not binary**"}[label]
        elif len(have) < len(files):
            label = f"{label} (only {', '.join(have)})" if len(have) <= 2 else \
                f"{label} ({len(have)} of {len(files)} files)"
        elif label.startswith("All"):
            label = f"**{label}**"
        gen = {c: bpb([by[(c, f)] for f in have]) for c in FAST
               if all((c, f) in by for f in have)}
        best = min(gen, key=gen.get)
        taco = bpb([by[("taco", f)] for f in have])
        paq_ok = all(("paq8px -8", f) in by for f in have)
        paq = bpb([by[("paq8px -8", f)] for f in have]) if paq_ok else None
        out.append(f"| {label} | {len(have)} | {sum(by[('taco', f)]['bytes'] for f in have):,} "
                   f"| {gen[best]:.3f} ({best}) | {f'{paq:.3f}' if paq else 'n.m.'} | "
                   f"{taco:.3f} | {pct(taco, gen[best])} | {pct(taco, paq) if paq else 'n.m.'} |")
    return out


def table_class_speed(runs: list[dict], s: str) -> list[str]:
    """Speed of paq8px and Taco for each class. Records reused from the samples are left out."""
    out = ["| Class | Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) |",
           "|---|---|---:|---:|---:|---:|"]
    for cls in CLASSES:
        for c in sorted(SLOW_SET):
            rows = [r for r in runs if r["compressor"] == c and r["set"] == s
                    and r["class"] == cls and "reused_from" not in r]
            if not rows:
                continue
            b = sum(r["bytes"] for r in rows)
            out.append(f"| {cls} | {c} | {len(rows)} | {b:,} | "
                       f"{kbs(b, sum(r['c_s'] for r in rows))} | "
                       f"{kbs(b, sum(r['d_s'] for r in rows))} |")
    return out


def table_speed(runs: list[dict], s: str) -> list[str]:
    out = ["| Compressor | Files | Bytes | Compress (KB/s) | Decompress (KB/s) | "
           "Peak memory, compress (MiB) | Peak memory, decompress (MiB) | Load average (1 min) |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in ORDER:
        rows = [r for r in runs if r["compressor"] == c and r["set"] == s
                and "reused_from" not in r]
        if not rows:
            continue
        b = sum(r["bytes"] for r in rows)
        cs = sum(r["c_s"] for r in rows)
        ds = sum(r["d_s"] for r in rows)
        loads = [r["c_load1"] for r in rows] + [r["d_load1"] for r in rows]
        out.append(f"| {c} | {len(rows)} | {b:,} | {kbs(b, cs)} | {kbs(b, ds)} | "
                   f"{max(r['c_rss_kib'] for r in rows) / 1024:,.0f} | "
                   f"{max(r['d_rss_kib'] for r in rows) / 1024:,.0f} | "
                   f"{min(loads):.1f} to {max(loads):.1f} |")
    return out


def para(text: str) -> None:
    print(textwrap.fill(text, 96))


TITLES = {"samples": "4. Samples of 32 KiB: bits per byte (measured)",
          "whole": "5. Whole files: bits per byte (measured)"}


def cmd_table(a: argparse.Namespace) -> int:
    """Print the Markdown tables of sections 4 to 6 of the benchmark note."""
    runs = [r for f in a.results for r in json.loads(Path(f).expanduser().read_text())["runs"]]
    sets = [s for s in ("samples", "whole") if any(r["set"] == s for r in runs)]
    allf = [f for fs in CLASSES.values() for f in fs]
    text = [f for f in allf if CLASS_OF[f] != "Binary"]
    for s in sets:
        print(f"\n## {TITLES[s]}\n")
        para("The best value in each row is in bold. n.m. means not measured. A class row "
             "and an overall row need all files of the row.")
        for k, (cls, names) in enumerate(CLASSES.items(), 1):
            print(f"\n### {TITLES[s][0]}.{k} {cls}\n")
            print("\n".join(table_sizes(runs, s, names)))
        print(f"\n### {TITLES[s][0]}.{len(CLASSES) + 1} Overall\n")
        rows = table_sizes(runs, s, allf)
        rows_t = table_sizes(runs, s, text)
        print("\n".join(rows[:2]))
        print(rows_t[-1].replace("**Class**", "**All, not binary**"))
        print(rows[-1].replace("**Class**", "**All files**"))
        if any(r["compressor"] == "taco" and r["set"] == s for r in runs):
            print(f"\n### {TITLES[s][0]}.{len(CLASSES) + 2} Taco against the others\n")
            para("Percentages are the change of the Taco size against the other size. A "
                 "negative value means that Taco is smaller.")
            print()
            print("\n".join(table_compare(runs, s)))
    print("\n## 6. Speed and memory (measured)")
    print()
    para("A whole file of 32 KiB or less is the same input as its sample. Its record is "
         "reused from the samples, and the speed tables of the set `whole` leave it out.")
    for k, s in enumerate(sets, 1):
        print(f"\n### 6.{k} Set `{s}`\n")
        print("\n".join(table_speed(runs, s)))
    for k, s in enumerate(sets, len(sets) + 1):
        if any(r["compressor"] in SLOW_SET and r["set"] == s for r in runs):
            print(f"\n### 6.{k} paq8px and Taco for each class, set `{s}`\n")
            print("\n".join(table_class_speed(runs, s)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--work", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--sets", nargs="*", default=["samples", "whole"])
    m.add_argument("--compressors", nargs="*", default=None, choices=list(COMPRESSORS))
    m.add_argument("--repeat", type=int, default=3, help="runs of each fast compressor")
    m.add_argument("--paq8px", default=None)
    m.add_argument("--paq8px-jobs", type=int, default=1, help="files that paq8px codes at once")
    m.add_argument("--taco", default=None)
    m.add_argument("--model", default=None)
    m.add_argument("--taco-threads", type=int, default=2)
    m.add_argument("--taco-jobs", type=int, default=2, help="files that taco codes at once")
    m.add_argument("--taco-whole-max", type=int, default=CAP,
                   help="whole files up to this size go to taco")
    m.add_argument("--taco-whole", nargs="*", default=None, help="more whole files for taco")
    t = sub.add_parser("table")
    t.add_argument("results", nargs="+", help="one or more results files of `measure`")
    a = ap.parse_args(argv)
    return {"measure": cmd_measure, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
