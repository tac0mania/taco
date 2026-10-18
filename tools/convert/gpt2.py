"""Convert GPT-2 small to `gpt2.hsm` (plan §6, §8, ADR 0004, ADR 0014).

Run from tools/ (after `python -m baseline.fetch`):
    nice uv run python -m convert.gpt2 [--recipe NAME] [--out PATH] [--new-hash]

Another model of the GPT-2 architecture with the GPT-2 vocabulary (for example distilgpt2,
6 blocks) needs its own weights and its own calibration statistics:
    nice uv run python -m convert.gpt2 --model M.safetensors --stats S.toml --out M.hsm

Inputs: the published float32 weights (tools/data/model), the GPT-2 vocabulary and merges, and
the calibration statistics in tools/convert/calib_stats.toml. The statistics are data: the tool
reads them and never runs the float model (ADR 0014 point 4; `exact.no_float_model` checks it).

DETERMINISM (ADR 0014): the conversion is exact (`convert.exact`, `convert.build`). The same
inputs give the same bytes on every machine. The tool compares the SHA-256 of the result with
the expected SHA-256 in `expected.toml` (`convert.selfcheck`). On a mismatch it stops with an
error and does not write the file. After a deliberate change, `--new-hash` writes the file and
prints the new entry.

Output: tools/data/out/gpt2.hsm (not committed) and tools/data/out/gpt2.ranges.json (the
selected ranges and scales, and the SHA-256 of each input). The defaults give `gpt2.hsm`
SHA-256 aaa03eb9...a5c2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from baseline._paths import MODEL_DIR, OUT_DIR, check_file, sha256_file
from baseline.bpe import EOT_ID, EOT_TEXT, Tokenizer
from convert import exact as ex
from convert import recipe as rc
from convert.build import build
from convert.calibrate import STATS_PATH
from convert.hsmfile import to_bytes, write_bytes
from convert.model import load_gpt2
from convert.selfcheck import Inputs, SelfCheckError, check

MODEL_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
MODEL_SIZE = 548105171


def vocabulary_and_merges(tok: Tokenizer) -> tuple[list[bytes], list[tuple[int, int]]]:
    """The GPT-2 vocabulary (token id order) and the merges as id pairs in rank order.

    NOTE: the encoder never emits <|endoftext|>, so its bytes are never decoded. The file
    stores its text, as the HardShell test models do.
    """
    vocab = list(tok.token_bytes)
    vocab[EOT_ID] = EOT_TEXT.encode("ascii")
    lines = (MODEL_DIR / "merges.txt").read_text(encoding="utf-8").split("\n")
    pairs = [tuple(line.split(" ")) for line in lines[1:] if line]
    merges = [(tok.encoder[a], tok.encoder[b]) for a, b in pairs]
    return vocab, merges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="clip", choices=sorted(rc.RECIPES))
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "gpt2.hsm")
    ap.add_argument("--model", type=Path, default=None,
                    help="a GPT-2-architecture safetensors file (default: GPT-2 small)")
    ap.add_argument("--model-sha256", default=None, help="the expected SHA-256 of --model")
    ap.add_argument("--name", default=None, help="the model name in expected.toml")
    ap.add_argument("--new-hash", action="store_true",
                    help="a deliberate change: write the file and print the new expected entry")
    args = ap.parse_args()
    t0 = time.time()
    if args.model is None:
        args.model = MODEL_DIR / "model.safetensors"
        check_file(args.model, MODEL_SIZE, MODEL_SHA256)
        model_sha = MODEL_SHA256
        name = args.name or "gpt2"
    else:
        model_sha = sha256_file(args.model)
        if args.model_sha256 is not None and model_sha != args.model_sha256:
            raise ValueError(f"{args.model}: sha256 {model_sha}, expected {args.model_sha256}")
        print(f"model {args.model}: sha256 {model_sha}")
        name = args.name or args.model.stem
    stats_sha = rc.stats_sha256(args.stats)
    inputs = Inputs(name, model_sha, args.recipe, stats_sha)
    with ex.no_float_model():
        stats = rc.load_stats(args.stats)
        sc = rc.scales(stats, rc.RECIPES[args.recipe])
        tok = Tokenizer()
        vocab, merges = vocabulary_and_merges(tok)
        fm = load_gpt2(args.model)
        hsm = build(fm, sc, vocab, merges)
        data = to_bytes(hsm)
    size, sha = len(data), hashlib.sha256(data).hexdigest()
    try:
        check(inputs, sha, size, args.new_hash)
    except SelfCheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    write_bytes(data, args.out)
    shifts = {t.name: t.shift for t in hsm.tensors if t.shift is not None}
    res, site = sc.floats()
    record = {
        "converter": "exact (ADR 0014)",
        "inputs": {"model": name, "weights_sha256": model_sha, "recipe": args.recipe,
                   "stats": str(args.stats.name), "stats_sha256": stats_sha},
        "recipe": args.recipe,
        "recipe_rules": rc.RECIPES[args.recipe],
        "residual_scale": res,
        "residual_scale_exact": str(sc.res),
        "residual_max_units": float(stats["residual"]["max"] / sc.res),
        "residual_min_row_std_units": float(stats["residual"]["row_std_min"] / sc.res),
        "ln_eps": hsm.config["constants"]["ln_eps"],
        "scales": site,
        "scales_exact": {k: str(v) for k, v in sc.site.items()},
        "shifts": shifts,
        "table_tie_distance_min": str(ex.TieStats.smallest),
        "size": size,
        "sha256": sha,
    }
    rec_path = args.out.with_suffix(".ranges.json")
    rec_path.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}: {size} bytes, sha256 {sha} ({time.time() - t0:.1f} s)")
    print(f"wrote {rec_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
