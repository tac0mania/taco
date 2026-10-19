"""Float GPT-2 (NumPy) bits of a block with the context rule of ADR 0015, window by window.

    uv run --project tools python tools/longctx/float_overlap.py WINDOWS.txt K[,K...]

Same windows as eval_overlap: window 0 = <eot> + first n_ctx - 1 tokens; window w > 0 = <eot>,
the K tokens before it, then n_ctx - 1 - K new tokens. Prints the ideal bits per byte and the
table bits per byte: `32 - log2(1 + p * (2^32 - V))`, the size with a table of plan §5.1 (the
floor of 1 caps the cost of a token that the float model gives almost no probability).
"""
import sys
import time

import numpy as np

from baseline.gpt2_np import GPT2

N_CTX = 1024
EOT = 50256


def log2_softmax_at(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = np.log(np.exp(logits - m).sum(axis=-1)) + m[:, 0]
    return (logits[np.arange(len(targets)), targets] - z) / np.log(2.0)


def main() -> None:
    path, ks = sys.argv[1], [int(k) for k in sys.argv[2].split(",")]
    model = GPT2(np.float32)
    from baseline.bpe import Tokenizer

    tok = Tokenizer()
    blocks = []
    for line in open(path):
        if line.strip():
            ids = [int(x) for x in line.split("\t", 1)[1].split()]
            blocks.append(ids)
    nbytes = sum(len(tok.decode(b)) for b in blocks)
    for k in ks:
        t0 = time.time()
        bits = 0.0
        table = 0.0
        for t in blocks:
            n = len(t)
            u = [EOT] + t[:-1]  # inputs; the output at input j predicts t[j]
            s = min(n, N_CTX)
            wins = [(0, s, 0)]
            while s < n:
                e = min(s + N_CTX - 1 - k, n)
                wins.append((s, e, k))
                s = e
            for s, e, kk in wins:
                if s == 0:
                    inp, tgt, skip = u[:e], t[:e], 0
                else:
                    inp, tgt, skip = [EOT] + u[s - kk : e], t[s:e], 1 + kk
                logits = model.forward(np.array(inp, dtype=np.int64))
                lp = log2_softmax_at(logits[skip:].astype(np.float64), np.array(tgt))
                bits -= float(lp.sum())
                f = 1.0 + np.exp2(lp) * float(2**32 - 50257)
                table += float((32.0 - np.log2(f)).sum())
        print(
            f"{path}\tK {k}\tfloat bpb ideal {bits / nbytes:.6f} table {table / nbytes:.6f}"
            f"\t{time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
