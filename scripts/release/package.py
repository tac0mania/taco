"""Make one release archive from a staging directory, with fixed metadata.

    python3 -I scripts/release/package.py --dir <stage>/<name> --format tar.gz|zip \
        --epoch <SOURCE_DATE_EPOCH> --out <out dir>

The script writes <stage>/<name>/SHA256SUMS (the files of the archive), then
<out>/<name>.tar.gz, .tar or .zip, then <out>/SHA256SUMS (every archive in <out>).

The archive is deterministic for the same files, the same epoch and the same zlib: the
members are sorted, the time of each member is the epoch, the owner is 0, and the modes are
0755 for the program and 0644 for the other files. Only the standard library is used.
"""

import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
import time
import zipfile

PROGRAMS = {"taco", "taco.exe"}
# The model archive is not compressed: the int8 weights do not get much smaller (est.).


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def members(stage):
    """Sorted file names of the staging directory (flat; no subdirectories)."""
    names = sorted(os.listdir(stage))
    for n in names:
        if not os.path.isfile(os.path.join(stage, n)):
            sys.exit(f"package.py: {n} is not a file; the archive is flat")
    return names


def write_sums(stage):
    lines = [f"{sha256(os.path.join(stage, n))}  {n}\n" for n in members(stage) if n != "SHA256SUMS"]
    with open(os.path.join(stage, "SHA256SUMS"), "w", newline="\n") as f:
        f.writelines(lines)


def mode_of(name):
    return 0o755 if name in PROGRAMS else 0o644


def make_tar(stage, name, epoch, out):
    """Writes the tar stream of the staging directory to the file object `out`."""
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        d = tarfile.TarInfo(name)
        d.type, d.mode, d.mtime = tarfile.DIRTYPE, 0o755, epoch
        tar.addfile(d)
        for n in members(stage):
            src = os.path.join(stage, n)
            info = tarfile.TarInfo(f"{name}/{n}")
            info.size = os.path.getsize(src)
            info.mode, info.mtime = mode_of(n), epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(src, "rb") as f:
                tar.addfile(info, f)


def make_tar_gz(stage, name, epoch, path):
    raw = io.BytesIO()
    make_tar(stage, name, epoch, raw)
    with open(path, "wb") as out:
        # NOTE: filename "" and a fixed mtime keep the gzip header the same on every run.
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=epoch, compresslevel=9) as gz:
            gz.write(raw.getvalue())


def make_zip(stage, name, epoch, path):
    # The zip time has no time zone and starts in 1980.
    t = time.gmtime(max(epoch, 315532800))[:6]
    with zipfile.ZipFile(path, "w") as z:
        for n in members(stage):
            info = zipfile.ZipInfo(f"{name}/{n}", date_time=t)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so that the mode bits are kept.
            info.external_attr = (0o100000 | mode_of(n)) << 16
            with open(os.path.join(stage, n), "rb") as f:
                z.writestr(info, f.read(), compresslevel=9)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", required=True, help="the staging directory; its name is the archive name")
    p.add_argument("--format", required=True, choices=["tar.gz", "tar", "zip"])
    p.add_argument("--epoch", required=True, type=int, help="SOURCE_DATE_EPOCH")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    stage = os.path.abspath(a.dir)
    name = os.path.basename(stage)
    os.makedirs(a.out, exist_ok=True)
    write_sums(stage)
    path = os.path.join(a.out, f"{name}.{a.format}")
    if a.format == "tar.gz":
        make_tar_gz(stage, name, a.epoch, path)
    elif a.format == "tar":
        with open(path, "wb") as out:
            make_tar(stage, name, a.epoch, out)
    else:
        make_zip(stage, name, a.epoch, path)
    archives = sorted(
        n for n in os.listdir(a.out) if n.startswith("taco") and n.endswith((".tar.gz", ".zip", ".tar"))
    )
    with open(os.path.join(a.out, "SHA256SUMS"), "w", newline="\n") as f:
        for n in archives:
            f.write(f"{sha256(os.path.join(a.out, n))}  {n}\n")
    print(f"archive: {sha256(path)}  {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
