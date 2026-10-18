"""Calibration: measure the range of each activation tensor of GPT-2 (ADR 0004 section 7).

Run from tools/:
    nice uv run python -m convert.calibrate --fetch

The tool runs the float GPT-2 (float32, full sequence) over the windows of calibration.toml and
records, for each site of `convert.model`: the largest absolute value, high quantiles of the
absolute value, and the largest absolute value of each channel. For the residual stream it
also records the standard deviation of each row (the LayerNorm input), because ADR 0004
section 1 needs 2^12 units or more.

Output: tools/convert/calib_stats.toml (committed; the converter reads it) and
tools/data/out/calib_channels.npz (the channel maxima; not committed).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

from baseline._paths import MODEL_DIR, OUT_DIR, check_file
from baseline.bpe import Tokenizer
from baseline.fetch import fetch
from convert.model import FModel, forward, load_gpt2
from convert.windows import CALIB_DIR, PACKAGE_DIR, calibration_windows

STATS_PATH = PACKAGE_DIR / "calib_stats.toml"
CHANNELS_PATH = OUT_DIR / "calib_channels.npz"
QUANTILES = (0.999, 0.9999, 0.99999, 0.999999)
BINS_PER_OCTAVE = 32
LOG2_MIN = -40


class Observer:
    """Collects the statistics of each site. Histograms of |x| on a log2 grid give quantiles
    with a resolution of 2^(1/32) (about 2.2 %)."""

    def __init__(self) -> None:
        self.max: dict[str, float] = {}
        self.hist: dict[str, np.ndarray] = {}
        self.chan: dict[str, np.ndarray] = {}
        self.count: dict[str, int] = {}
        self.res_pos0_max = 0.0
        self.res_rest_max = 0.0
        self.row_std: list[np.ndarray] = []
        self.row_std_pos0: list[float] = []
        self.layer = 0

    def __call__(self, site: str, x: np.ndarray) -> None:
        a = np.abs(x.astype(np.float64))
        if site == "res":
            self.res_pos0_max = max(self.res_pos0_max, float(a[0].max()))
            if len(a) > 1:
                self.res_rest_max = max(self.res_rest_max, float(a[1:].max()))
            sd = x.astype(np.float64).std(axis=1)
            self.row_std.append(sd)
            self.row_std_pos0.append(float(sd[0]))
        self.max[site] = max(self.max.get(site, 0.0), float(a.max()))
        flat = a.reshape(-1)
        nz = flat[flat > 0]
        idx = np.floor(np.log2(nz) * BINS_PER_OCTAVE).astype(np.int64) - LOG2_MIN * BINS_PER_OCTAVE
        idx = np.clip(idx, 0, 64 * BINS_PER_OCTAVE - 1)
        h = np.bincount(idx, minlength=64 * BINS_PER_OCTAVE)
        self.hist[site] = self.hist.get(site, 0) + h
        self.count[site] = self.count.get(site, 0) + flat.size
        cm = a.reshape(-1, a.shape[-1]).max(axis=0)
        self.chan[site] = np.maximum(self.chan.get(site, cm), cm)

    def quantile(self, site: str, q: float) -> float:
        """The upper edge of the histogram bin that holds the q quantile of |x|."""
        h = self.hist[site]
        n = self.count[site]
        zeros = n - int(h.sum())
        need = q * n - zeros
        if need <= 0:
            return 0.0
        k = int(np.searchsorted(np.cumsum(h), need))
        upper = 2.0 ** ((k + 1) / BINS_PER_OCTAVE + LOG2_MIN)
        return min(upper, self.max[site])


def run(model: FModel, windows) -> Observer:
    obs = Observer()
    for i, w in enumerate(windows):
        t0 = time.time()
        forward(model, w.model_input, hook=obs)
        print(f"  window {i + 1}/{len(windows)} {w.label}: {time.time() - t0:.1f} s", flush=True)
    return obs


def _f(x: float) -> str:
    return repr(float(x))


def write_stats(obs: Observer, windows, path: Path) -> None:
    sd = np.concatenate(obs.row_std)
    lines = [
        "# Calibration statistics of GPT-2 small (measured; `python -m convert.calibrate`).",
        "# The converter (`python -m convert.gpt2`) reads this file and selects the scales.",
        "# All values are absolute values of activations, in float units.",
        "",
        "[meta]",
        f'date = "{date.today().isoformat()}"',
        f"windows = {len(windows)}",
        f"tokens = {sum(len(w.tokens) for w in windows)}",
        'model = "float32, full sequence, tools/convert/model.py"',
        f"histogram_bins_per_octave = {BINS_PER_OCTAVE}",
        "",
        "[residual]",
        f"max = {_f(obs.max['res'])}",
        f"max_position_0 = {_f(obs.res_pos0_max)}",
        f"max_other_positions = {_f(obs.res_rest_max)}",
        f"row_std_min = {_f(sd.min())}",
        f"row_std_q0001 = {_f(np.quantile(sd, 0.0001))}",
        f"row_std_q001 = {_f(np.quantile(sd, 0.001))}",
        f"row_std_q01 = {_f(np.quantile(sd, 0.01))}",
        f"row_std_median = {_f(np.median(sd))}",
        f"row_std_position_0_min = {_f(min(obs.row_std_pos0))}",
        "",
    ]
    for site in obs.max:
        if site == "res":
            continue
        lines.append(f'[site."{site}"]')
        lines.append(f"max = {_f(obs.max[site])}")
        for q in QUANTILES:
            key = "q" + f"{q:.6f}".split(".")[1].rstrip("0")
            lines.append(f"{key} = {_f(obs.quantile(site, q))}")
        # The largest value of each channel, then a quantile over the channels.
        for q in (50, 90, 99):
            lines.append(f"chan_q{q} = {_f(np.percentile(obs.chan[site], q))}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download the calibration files")
    args = ap.parse_args()
    if args.fetch:
        import tomllib

        with (PACKAGE_DIR / "calibration.toml").open("rb") as f:
            spec = tomllib.load(f)
        for s in spec["source"]:
            fetch(s["url"], CALIB_DIR / s["name"], s["size"], s["sha256"])
    tok = Tokenizer()
    windows = calibration_windows(tok)
    print(f"calibration: {len(windows)} windows, {sum(len(w.tokens) for w in windows)} tokens")
    check_file(MODEL_DIR / "model.safetensors", 548105171,
               "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707")
    model = load_gpt2(MODEL_DIR / "model.safetensors")
    obs = run(model, windows)
    write_stats(obs, windows, STATS_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CHANNELS_PATH, **{k: v for k, v in obs.chan.items()})
    print(f"wrote {STATS_PATH} and {CHANNELS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
