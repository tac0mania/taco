"""Convert GPT-2 small to `gpt2.hsm` (plan §6, §8, ADR 0004).

Run from tools/ (after `python -m baseline.fetch`):
    nice uv run python -m convert.gpt2 [--recipe NAME] [--out PATH]

Inputs: the published float weights (tools/data/model), the GPT-2 vocabulary and merges, and
the calibration statistics in tools/convert/calib_stats.toml. The tool does not run the model.
Thus the same inputs give the same file on every machine (`convert.numerics`, DETERMINISM).

Output: tools/data/out/gpt2.hsm (not committed) and tools/data/out/gpt2.ranges.json (the
selected ranges and scales). The tool prints the size and the SHA-256 of the model file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from baseline._paths import MODEL_DIR, OUT_DIR, check_file
from baseline.bpe import EOT_ID, EOT_TEXT, Tokenizer
from convert import recipe as rc
from convert.build import build
from convert.calibrate import STATS_PATH
from convert.hsmfile import write
from convert.model import load_gpt2

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
    ap.add_argument("--recipe", default="max", choices=sorted(rc.RECIPES))
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "gpt2.hsm")
    args = ap.parse_args()
    t0 = time.time()
    check_file(MODEL_DIR / "model.safetensors", MODEL_SIZE, MODEL_SHA256)
    stats = rc.load_stats(args.stats)
    sc = rc.scales(stats, rc.RECIPES[args.recipe])
    tok = Tokenizer()
    vocab, merges = vocabulary_and_merges(tok)
    fm = load_gpt2(MODEL_DIR / "model.safetensors")
    hsm = build(fm, sc, vocab, merges)
    size, sha = write(hsm, args.out)
    shifts = {t.name: t.shift for t in hsm.tensors if t.shift is not None}
    record = {
        "recipe": args.recipe,
        "recipe_rules": rc.RECIPES[args.recipe],
        "residual_scale": sc.res,
        "residual_max_units": stats["residual"]["max"] / sc.res,
        "residual_min_row_std_units": stats["residual"]["row_std_min"] / sc.res,
        "ln_eps": hsm.config["constants"]["ln_eps"],
        "scales": sc.site,
        "shifts": shifts,
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
