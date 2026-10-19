"""Fake quantisation on a GPU: the same measurement as `convert.fq`, about 100 times faster.

Run from tools/ with a Python that has PyTorch (on devbox: ~/taco-train/venv/bin/python):
    PYTHONPATH=. python -m convert.fq_gpu --model M.safetensors --stats S.toml --windows W.txt \
        --modes float w all ... --spec clip --spec clip:ln_f=x0.1,gelu=chan_q99 ...

A spec is `RECIPE[:option,...]`. An option is `kind=statistic` (for example `gelu=chan_q99`,
or `ln_f=x0.1`: 0.1 times the largest value), or `site=value` with a float range for one site
(for example `h.5.gelu=6.0`). The modes are the modes of `convert.fq`. The tool prints the
ideal size in bits per byte for each spec and mode, and the change against the first mode.

The float model is float32 with TF32 off. Its sizes agree with `convert.fq` to about 1e-4
bits per byte (float32 sums in another order), so use it to compare recipes, and HardShell
(`eval_bits`) for the published numbers. Use calibration or validation windows only when the
result selects a recipe, never the evaluation windows.

NOTE: a measurement tool with floats, not the prediction path (CLAUDE.md §1 rule 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from baseline.bpe import Tokenizer
from convert import numerics as nm
from convert import recipe as rc
from convert.calibrate import read_windows
from convert.fq import kind_of, parse_mode
from convert.model import LN_EPS, load_gpt2

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEV)


def deq(w: torch.Tensor) -> torch.Tensor:
    """i8 with one scale for each row (`numerics.quantise_rows`), back to float."""
    mx = w.abs().amax(dim=1, keepdim=True)
    s = torch.where(mx > 0, mx / nm.I8_MAX, torch.ones_like(mx))
    return torch.clamp(torch.round(w / s), -nm.I8_MAX, nm.I8_MAX) * s


class GModel:
    """The float weights on the device, with i8 copies made on first use."""

    MATS = ("attn_w", "proj_w", "fc_w", "mlp_w")
    VECS = ("ln1_g", "ln1_b", "attn_b", "proj_b", "ln2_g", "ln2_b", "fc_b", "mlp_b")

    def __init__(self, path: Path) -> None:
        fm = load_gpt2(path)
        self.n_head = fm.n_head
        self.f = {"wte": _t(fm.wte), "wpe": _t(fm.wpe), "lnf_g": _t(fm.lnf_g),
                  "lnf_b": _t(fm.lnf_b)}
        self.blocks = [{k: _t(getattr(b, k)) for k in (*self.MATS, *self.VECS)}
                       for b in fm.blocks]
        self.q: dict[str, torch.Tensor] = {}

    def emb(self, quant: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if not quant:
            return self.f["wte"], self.f["wpe"]
        if "wte" not in self.q:
            self.q["wte"], self.q["wpe"] = deq(self.f["wte"]), deq(self.f["wpe"])
        return self.q["wte"], self.q["wpe"]

    def mat(self, li: int, name: str, quant: bool) -> torch.Tensor:
        if not quant:
            return self.blocks[li][name]
        key = f"{li}.{name}"
        if key not in self.q:
            self.q[key] = deq(self.blocks[li][name])
        return self.q[key]


def _ln(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    xc = x - mean
    var = (xc * xc).mean(dim=-1, keepdim=True)
    return xc / torch.sqrt(var + LN_EPS) * g + b


def _gelu(x: torch.Tensor) -> torch.Tensor:
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x**3)))


def make_hook(res: float, site_scale: dict[str, float], parts: set[str],
              wide: frozenset[str] = frozenset()):
    """`wide` names the site kinds that are i16 (the same range, 32,767 levels)."""

    def hook(site: str, x: torch.Tensor) -> torch.Tensor:
        k = kind_of(site)
        if k not in parts and site not in parts:
            return x
        if site == "res":
            return torch.clamp(torch.round(x / res), -nm.RESIDUAL_LIMIT, nm.RESIDUAL_LIMIT) * res
        s = site_scale[site]
        q = nm.I16_MAX if k == "fc" else nm.I8_MAX
        if k in wide and k != "fc":
            s, q = s * nm.I8_MAX / nm.I16_MAX, nm.I16_MAX
        return torch.clamp(torch.round(x / s), -q, q) * s

    return hook


@torch.no_grad()
def forward(m: GModel, tokens: torch.Tensor, wq: tuple[bool, bool], hook) -> torch.Tensor:
    t = len(tokens)
    wte, wpe = m.emb(wq[0])
    d = wte.shape[1]
    nh = m.n_head
    hd = d // nh
    x = hook("res", wte[tokens] + wpe[:t])
    mask = torch.triu(torch.full((t, t), float("-inf"), device=DEV), diagonal=1)
    for li, b in enumerate(m.blocks):
        p = f"h.{li}."
        a = hook(p + "ln_1", _ln(x, b["ln1_g"], b["ln1_b"]))
        qkv = a @ m.mat(li, "attn_w", wq[1]).T + b["attn_b"]
        q = hook(p + "q", qkv[:, :d])
        k = hook(p + "k", qkv[:, d : 2 * d])
        v = hook(p + "v", qkv[:, 2 * d :])
        qh = q.reshape(t, nh, hd).transpose(0, 1)
        kh = k.reshape(t, nh, hd).permute(1, 2, 0)
        vh = v.reshape(t, nh, hd).transpose(0, 1)
        s = (qh @ kh) / float(np.sqrt(hd)) + mask
        att = torch.softmax(s, dim=-1)
        y = hook(p + "att", (att @ vh).transpose(0, 1).reshape(t, d))
        x = hook("res", x + (y @ m.mat(li, "proj_w", wq[1]).T + b["proj_b"]))
        a = hook(p + "ln_2", _ln(x, b["ln2_g"], b["ln2_b"]))
        f = hook(p + "fc", a @ m.mat(li, "fc_w", wq[1]).T + b["fc_b"])
        g = hook(p + "gelu", _gelu(f))
        x = hook("res", x + (g @ m.mat(li, "mlp_w", wq[1]).T + b["mlp_b"]))
    a = hook("ln_f", _ln(x, m.f["lnf_g"], m.f["lnf_b"]))
    return a @ wte.T


LOGIT_TABLE = torch.from_numpy(nm.logits_exp_table().astype(np.float64))


def bits(logits: torch.Tensor, targets: torch.Tensor, int_logits: bool) -> torch.Tensor:
    """-log2 p of each target: the float softmax, or the integer logit rule (unit 2^-8, table,
    and a smallest probability of 2^-32)."""
    z = logits.double()
    if not int_logits:
        lp = torch.log_softmax(z, dim=-1)
        return -lp[torch.arange(len(targets)), targets] / np.log(2.0)
    zi = torch.round(z * (1 << -nm.DELTA_LOG2))
    gap = torch.clamp(zi.amax(dim=1, keepdim=True) - zi, 0, len(LOGIT_TABLE) - 1).long()
    w = LOGIT_TABLE.to(z.device)[gap]
    # NOTE: the coder gives every token a frequency of 1 or more in a 32-bit table (plan
    # §5.1), so a weight of 0 costs about 32 bits, not an infinite size.
    p = torch.clamp(w[torch.arange(len(targets)), targets] / w.sum(dim=1), min=2.0**-32)
    return -torch.log2(p)


def parse_spec(spec: str, stats: dict
               ) -> tuple[dict[str, str], dict[str, Fraction], frozenset[str]]:
    """A recipe, its site overrides and its i16 site kinds from `RECIPE[:option,...]`.

    The option `i16=gelu+ln_f` makes these site kinds i16 (a proposal for an engine change).
    """
    name, _, opts = spec.partition(":")
    r = dict(rc.RECIPES[name])
    over: dict[str, Fraction] = {}
    wide: frozenset[str] = rc.WIDE.get(name, frozenset())
    for o in filter(None, opts.split(",")):
        k, v = o.split("=")
        if k == "i16":
            wide = frozenset(v.split("+"))
        elif k in r:
            r[k] = v
        elif k in stats["site"]:
            over[k] = Fraction(v)
        else:
            raise ValueError(f"unknown option {o}")
    return r, over, wide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--modes", nargs="+", default=["float", "w", "all"])
    ap.add_argument("--per-window", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    stats = rc.load_stats(args.stats)
    tok = Tokenizer()
    windows = read_windows(args.windows, tok)
    nbytes = sum(len(w.data) for w in windows)
    m = GModel(args.model)
    inputs = [(torch.tensor(w.model_input, device=DEV), torch.tensor(w.tokens, device=DEV))
              for w in windows]
    out = {"labels": [w.label for w in windows], "bytes": [len(w.data) for w in windows],
           "specs": {}}
    base = None
    for spec in args.spec or ["clip"]:
        recipe, over, wide = parse_spec(spec, stats)
        res, site = rc.scales(stats, recipe, over).floats()
        out["specs"][spec] = {}
        for mode in args.modes:
            t0 = time.time()
            wq, parts = parse_mode(mode)
            hook = make_hook(res, site, parts, wide)
            per = [float(bits(forward(m, x, wq, hook), y, "logit" in parts).sum())
                   for x, y in inputs]
            b = sum(per)
            base = b if base is None else base
            out["specs"][spec][mode] = per
            print(f"{spec:44s} {mode:16s} {b / nbytes:.4f} bpb {100 * (b / base - 1):+7.2f} % "
                  f"({time.time() - t0:.1f} s)", flush=True)
            if args.per_window:
                print("   " + " ".join(f"{p / len(w.data):.3f}" for p, w in zip(per, windows,
                                                                            strict=True)))
    if args.json:
        args.json.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
