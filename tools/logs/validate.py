"""Check the exact bounds of `logs.sim` against full rows (plan §5.1 on the mixed weights).

Run from tools/ (on devbox, after `logs.gpt2pass` wrote the file):
    OPENBLAS_NUM_THREADS=4 nice -n 15 uv run python -m logs.validate NAME [WINDOWS]

The tool runs the integer GPT-2 again on the first WINDOWS windows of the first 16 KiB block of
NAME, applies the mixer of `logs.mixer` (rule version 1) to each full row, makes the exact
table (`logs.freq`), and adds the cost of the actual tokens. Then it compares the sum with the
bounds of `logs.sim.exact_bits` on the same tokens. The full-row cost must be inside the
bounds.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from baseline.bpe import EOT_ID

from logs import fastsim, freq, sim
from logs import matchmodel as mm
from logs import mixer as mx


def main() -> int:
    name = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    f = sim.load(name)
    end = int(f.block_starts[1])
    stop = min(end, count * 1024)
    pred, plen = mm.predict_file(f.ids, f.block_starts, 1, sim.DEFAULT_MATCH)
    wp = sim.pred_weight(f, pred)
    q = sim.run_mixer(f, pred, plen, wp, 1, sim.DEFAULT_MIXER)
    sub = sim.Facts(f.name, 0, f.ids[:stop], np.array([0, stop]), f.W[:stop], f.wa[:stop],
                    f.fa[:stop], f.top_id[:stop], f.top_w[:stop], f.cand[:stop], f.wc[:stop])
    lo, hi = sim.exact_bits(sub, pred[:stop], wp[:stop], q[:stop])
    fm = fastsim.load()
    full = 0.0
    checked = 0
    for s in range(0, stop, 1024):
        e = min(s + 1024, stop)
        ids = f.ids[s:e]
        model_input = np.concatenate([[EOT_ID], ids[:-1]]).astype(np.int64)
        for r0, w in fastsim.forward_chunks(fm, model_input):
            for r in range(len(w)):
                i = s + r0 + r
                if i >= e:
                    break
                row = w[r]
                if int(row.sum(dtype=np.uint64)) != int(f.W[i]):
                    raise SystemExit(f"row {i}: the weight sum differs from the pass")
                out = mx.mix_row_np(row, int(pred[i]), int(q[i]))
                fa = int(freq.entries(out[None, :], np.array([f.ids[i]]))[0])
                full += -math.log2(fa / freq.TOTAL)
                checked += 1
    inside = lo - 1e-9 <= full <= hi + 1e-9
    print(f"{name}: {checked} tokens, full rows {full:.6f} bits, bounds [{lo:.6f}, {hi:.6f}], "
          f"width {hi - lo:.6f} bits, inside: {inside}", flush=True)
    return 0 if inside else 1


if __name__ == "__main__":
    sys.exit(main())
