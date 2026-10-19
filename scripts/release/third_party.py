"""Write the licences of the crates that the `taco` binary contains, for one target.

    python3 -I scripts/release/third_party.py --manifest Cargo.toml --target <triple>

The output (stdout) lists each crate (name, version, licence expression, source) in the
normal-dependency graph of `taco-cli` for <target>, then each licence text of those crates.
Identical texts are printed one time, with the names of the crates that use them. Build
dependencies (for example `cc`) and dev-dependencies are not in the binary and are left out.
Proc-macro crates are kept: their output is in the binary.

The texts come from the crate sources in the Cargo registry (`cargo metadata` downloads a
crate that is missing). A crate with no licence file gets a warning on stderr, and the release
checklist says to resolve each warning before a release (docs/design/release.md).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

LICENCE_PREFIXES = ("license", "licence", "copying", "notice", "copyright", "unlicense")
# Bundled C sources whose licence is in a subdirectory of the crate.
EXTRA_DIRS = {"zstd-sys": ["zstd"]}


def licence_files(pkg):
    root = os.path.dirname(pkg["manifest_path"])
    found = []
    if pkg.get("license_file"):
        found.append(os.path.join(root, pkg["license_file"]))
    for sub in [""] + EXTRA_DIRS.get(pkg["name"], []):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if os.path.isfile(p) and n.lower().startswith(LICENCE_PREFIXES) and p not in found:
                found.append(p)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--root", default="taco-cli", help="the package of the binary")
    a = ap.parse_args()
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked", "--manifest-path", a.manifest,
             "--filter-platform", a.target],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = next(p["id"] for p in meta["packages"] if p["name"] == a.root and p["source"] is None)
    seen, todo = set(), [root]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            if any(k["kind"] is None for k in d["dep_kinds"]):
                todo.append(d["pkg"])
    third = sorted((pkgs[i] for i in seen if pkgs[i]["source"] is not None),
                   key=lambda p: (p["name"], p["version"]))
    ours = sorted(pkgs[i]["name"] for i in seen if pkgs[i]["source"] is None)

    print(f"Third-party software in the taco binary for {a.target}")
    print("=" * 72)
    print()
    print("The taco binary contains the crates of this workspace (Apache-2.0, see LICENSE):")
    print("  " + ", ".join(ours))
    print()
    print(f"It also contains these {len(third)} third-party crates. Their licence texts follow the list.")
    print()
    texts, missing = {}, []
    for p in third:
        files = licence_files(p)
        if not files:
            missing.append(p)
            print(f"third_party.py: WARNING: {p['name']} {p['version']} has no licence file", file=sys.stderr)
        for f in files:
            with open(f, "rb") as fh:
                body = fh.read().decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
            key = hashlib.sha256(body.encode()).hexdigest()
            texts.setdefault(key, {"body": body, "users": []})["users"].append(
                f"{p['name']} {p['version']} ({os.path.basename(f)})")
        print(f"  {p['name']} {p['version']}  licence: {p.get('license') or 'see file'}  "
              f"source: {p.get('repository') or p['source']}")
    print()
    if missing:
        print("These crates have no licence file in their published source. Their licence is the")
        print("expression in the list; the standard text of each licence is below (from another")
        print("crate), and the copyright holders are the authors of the crate:")
        for p in missing:
            authors = ", ".join(p.get("authors") or []) or "see the source"
            print(f"  {p['name']} {p['version']}  licence: {p.get('license')}  authors: {authors}")
        print()
    for n, t in enumerate(sorted(texts.values(), key=lambda t: t["users"][0]), 1):
        print("-" * 72)
        print(f"Text {n}. Used by: " + "; ".join(t["users"]))
        print("-" * 72)
        print(t["body"])
        print()


if __name__ == "__main__":
    main()
