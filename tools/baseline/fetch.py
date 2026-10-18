"""Download the GPT-2 model files and the quality corpus, and check their hashes.

Run: uv run --project tools python -m baseline.fetch
A file that is already present with the correct SHA-256 is not downloaded again. A wrong hash
is an error. The tool only stores the files; it never runs them.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from baseline._paths import CORPUS_DIR, MODEL_DIR, check_file, load_toml


def fetch(url: str, dest: Path, size: int, sha256: str) -> None:
    try:
        check_file(dest, size, sha256)
        print(f"ok      {dest.name}")
        return
    except ValueError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"fetch   {dest.name} <- {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    check_file(dest, size, sha256)
    print(f"ok      {dest.name}")


def main() -> int:
    manifest = load_toml("manifest.toml")
    base = manifest["source"]["base_url"]
    for f in manifest["file"]:
        fetch(base + f["name"], MODEL_DIR / f["name"], f["size"], f["sha256"])
    for f in load_toml("corpus.toml")["file"]:
        fetch(f["url"], CORPUS_DIR / f.get("local", f["name"]), f["size"], f["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
