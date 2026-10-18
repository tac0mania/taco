"""The scale recipes: calibration statistics to activation ranges (ADR 0004 sections 1 and 7).

A recipe selects, for each site, which calibration statistic is the range: the largest value
(`max`, the rule of ADR 0004 section 7) or a quantile of |x| (`q9999` and so on). Values above
the range saturate (clamp). Each range is rounded up to the grid of `exact.GRID` (steps of
about 2^(1/16)), so a calibration on another machine gives the same model file.

The residual rule (ADR 0004 section 1): the residual scale gives the smallest row standard
deviation of the calibration 2^12.5 units (a margin above the 2^12 limit), and the largest
residual value must stay at or below 2^28 units (4 times below the clamp at 2^30).

DETERMINISM (ADR 0014): the statistics are data. `load_stats` reads each number as its exact
binary64 value, and every step here is exact arithmetic on fractions. This module does not run
the float model.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

from convert import exact as ex
from convert.build import Scales

RES_STD_LOG2 = 12.5
RES_MAX_LOG2 = 28

# Site kinds: the part of the site name after the layer prefix.
KINDS = ("ln_1", "q", "k", "v", "att", "ln_2", "fc", "gelu")

RECIPES: dict[str, dict[str, str]] = {
    # ADR 0004 section 7: the largest calibration value maps to 127 (or 32,767).
    "max": {k: "max" for k in (*KINDS, "ln_f")},
    # Fix 1 (measured on the calibration windows, research note 2026-10-18-integer-gpt2):
    # ln_f and the GELU outputs have a few outlier channels. Their range is the median
    # (ln_f) or the 90th percentile (GELU) over the channels of the channel maximum, and the
    # outlier channels saturate.
    "clip": {**{k: "max" for k in KINDS}, "ln_f": "chan_q50", "gelu": "chan_q90"},
}


def load_stats(path: Path) -> dict:
    """Read a statistics file. Each number is an exact fraction (`exact.binary64_of_text`)."""
    return ex.load_toml_exact(path)


def stats_sha256(path: Path) -> str:
    """The SHA-256 of a statistics file. The self-check of the converter uses it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_scale(stats: dict) -> Fraction:
    """The residual scale: the grid point at or above row_std_min / 2^12.5, exact.

    2^12.5 = 2^12 * sqrt(2) is irrational, so the converter rounds the square
    row_std_min^2 / 2^25 (`exact.round_range_sq`).
    """
    r = stats["residual"]
    s = ex.round_range_sq(Fraction(r["row_std_min"]) ** 2 / (1 << 25))
    if Fraction(r["max"]) / s > 1 << RES_MAX_LOG2:
        raise ValueError(
            f"the residual rule fails: max {float(r['max'])} is "
            f"{float(r['max'] / s):.3g} units > 2^28"
        )
    return s


def ranges(stats: dict, recipe: dict[str, str], overrides: dict[str, Fraction] | None = None
           ) -> dict[str, Fraction]:
    """The range of each site. `overrides[site]` replaces the statistic for one site."""
    out = {}
    for site, st in stats["site"].items():
        kind = "ln_f" if site == "ln_f" else site.split(".", 2)[2]
        key = recipe[kind]
        value = st[key] if not key.startswith("x") else st["max"] * Fraction(key[1:])
        out[site] = ex.round_range(value)
    for site, v in (overrides or {}).items():
        out[site] = ex.round_range(Fraction(v))
    return out


def scales(stats: dict, recipe: dict[str, str], overrides: dict[str, Fraction] | None = None
           ) -> Scales:
    return Scales.from_ranges(residual_scale(stats), ranges(stats, recipe, overrides))
