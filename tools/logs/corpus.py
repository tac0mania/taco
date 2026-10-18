"""The log corpus of `logs.toml`: download, hash check, and load.

Run from tools/:
    uv run python -m logs.corpus fetch

A file that is already present with the correct SHA-256 is not downloaded again. A wrong hash
is an error. The tool stores the files as data only; it never runs them.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from baseline._paths import CORPUS_DIR, DATA_DIR, check_file, load_toml
from baseline.fetch import fetch

LOGS_DIR = DATA_DIR / "logs"
OUT_DIR = DATA_DIR / "out" / "logs"
PACKAGE_DIR = Path(__file__).resolve().parent

# The phase 0 files that are not logs and not binary. The mixer must not make them worse by
# more than about 0.5 % (the brief of ADR 0010).
TEXT_FILES = ("alice29.txt", "asyoulik.txt", "cp.html", "fields.c", "grammar.lsp", "lcet10.txt",
              "plrabn12.txt", "xargs.1", "pg22367.txt", "pg4650.txt", "pg24264.txt",
              "textwrap.py", "option.rs", "twitter.json")
# The log files of the GPT-2 measurement: all files of logs.toml.
GPT2_LOGS = ("HDFS_2k.log", "Apache_2k.log", "BGL_2k.log", "Linux_2k.log", "OpenSSH_2k.log",
             "Spark_2k.log", "Zookeeper_2k.log", "Android_2k.log", "Windows_2k.log", "Mac_2k.log",
             "HealthApp_2k.log", "Proxifier_2k.log", "Hadoop_2k.log", "HPC_2k.log",
             "Thunderbird_2k.log", "OpenStack_2k.log", "apache_access.log", "nginx_json.log")


@dataclass(frozen=True)
class LogFile:
    """The bytes of one corpus file that the measurement uses."""

    name: str
    group: str
    data: bytes


def spec() -> list[dict]:
    with (PACKAGE_DIR / "logs.toml").open("rb") as f:
        return tomllib.load(f)["file"]


def load(names: list[str] | None = None) -> list[LogFile]:
    """The log files in `logs.toml` order, each checked against its SHA-256."""
    out = []
    for f in spec():
        if names and f["name"] not in names:
            continue
        path = LOGS_DIR / f["name"]
        check_file(path, f["size"], f["sha256"])
        data = path.read_bytes()
        if f["cap"]:
            data = data[: f["cap"]]
        out.append(LogFile(f["name"], f["group"], data))
    return out


def load_text(cap: int = 32768) -> list[LogFile]:
    """The phase 0 text files of TEXT_FILES (the first `cap` bytes, as phase 0)."""
    out = []
    for f in load_toml("corpus.toml")["file"]:
        if f["name"] not in TEXT_FILES:
            continue
        path = CORPUS_DIR / f.get("local", f["name"])
        check_file(path, f["size"], f["sha256"])
        data = path.read_bytes()[f["offset"] : f["offset"] + cap]
        out.append(LogFile(f["name"], "text", data))
    return out


def main() -> int:
    if sys.argv[1:] != ["fetch"]:
        print(__doc__)
        return 64
    for f in spec():
        fetch(f["url"], LOGS_DIR / f["name"], f["size"], f["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
