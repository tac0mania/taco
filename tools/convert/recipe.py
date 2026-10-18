"""The scale recipes: calibration statistics to activation ranges (ADR 0004 sections 1 and 7).

A recipe selects, for each site, which calibration statistic is the range: the largest value
(`max`, the rule of ADR 0004 section 7) or a quantile of |x| (`q9999` and so on). Values above
the range saturate (clamp). Each range is rounded up to the grid 2^(k/16), so a calibration on
another machine gives the same model file (`numerics.round_range`).

The residual rule (ADR 0004 section 1): the residual scale gives the smallest row standard
deviation of the calibration 2^12.5 units (a margin above the 2^12 limit), and the largest
residual value must stay at or below 2^28 units (4 times below the clamp at 2^30).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from convert import numerics as nm
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
    with path.open("rb") as f:
        return tomllib.load(f)


def residual_scale(stats: dict) -> float:
    r = stats["residual"]
    s = r["row_std_min"] / nm.pow2(12) / 2.0**0.5  # 2^RES_STD_LOG2 = 2^12 * sqrt(2)
    s = nm.round_range(s)
    if r["max"] / s > nm.pow2(RES_MAX_LOG2):
        raise ValueError(
            f"the residual rule fails: max {r['max']} is {r['max'] / s:.3g} units > 2^28"
        )
    return s


def ranges(stats: dict, recipe: dict[str, str], overrides: dict[str, float] | None = None
           ) -> dict[str, float]:
    """The range of each site. `overrides[site]` replaces the statistic for one site."""
    out = {}
    for site, st in stats["site"].items():
        kind = "ln_f" if site == "ln_f" else site.split(".", 2)[2]
        key = recipe[kind]
        value = st[key] if not key.startswith("x") else st["max"] * float(key[1:])
        out[site] = nm.round_range(value)
    for site, v in (overrides or {}).items():
        out[site] = nm.round_range(v)
    return out


def scales(stats: dict, recipe: dict[str, str], overrides: dict[str, float] | None = None
           ) -> Scales:
    return Scales.from_ranges(residual_scale(stats), ranges(stats, recipe, overrides))
