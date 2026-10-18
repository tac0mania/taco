"""Check the NumPy GPT-2 against known facts.

Run: uv run --project tools python -m baseline.check_model

1. The top next tokens after "The capital of France is".
2. Greedy generation of 20 tokens from "The dog" on the KV path. The expected text is the
   published output of the GPT-2 greedy test in Hugging Face transformers
   (test_modeling_gpt2.py, `test_lm_generate_gpt2`).
3. The mean loss in nats per token on English prose (first window of alice29.txt,
   asyoulik.txt, lcet10.txt; and alice29.txt with LF line ends instead of CRLF).
"""

from __future__ import annotations

import numpy as np

from baseline.bpe import EOT_ID, Tokenizer
from baseline.corpus import load_samples
from baseline.gpt2_np import GPT2

# UNVERIFIED: the text is from the transformers test as remembered; the test is not downloaded.
HF_GREEDY_TEXT = (b"The dog was found in a field near the intersection of West and West Streets."
                  b"\n\nThe dog")
GREEDY_LEN = 20


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> None:
    tok = Tokenizer()
    m = GPT2(np.float32)
    paris = tok.encode(b" Paris")[0]
    for prefix, label in (([], "no prefix"), ([EOT_ID], "<|endoftext|> prefix")):
        ids = prefix + tok.encode(b"The capital of France is")
        p = softmax(m.forward(np.array(ids))[-1])
        top = np.argsort(-p)[:5]
        rank = int((p > p[paris]).sum())
        print(f"France ({label}): top 5 = "
              + ", ".join(f"{tok.decode([int(i)])!r} {p[i]:.3f}" for i in top)
              + f"; ' Paris' rank {rank} (0 = top), p = {p[paris]:.3f}")
    state = m.new_state(1)
    out = tok.encode(b"The dog")
    logits = None
    for t in out:
        logits = m.step(state, np.array([t]))
    while len(out) < GREEDY_LEN:
        nxt = int(np.argmax(logits[0]))
        out.append(nxt)
        logits = m.step(state, np.array([nxt]))
    same = tok.decode(out) == HF_GREEDY_TEXT
    print(f"greedy 'The dog' (KV path): {tok.decode(out)!r}; ids {out}; "
          f"same text as published: {same}")
    samples = load_samples(names=["alice29.txt", "asyoulik.txt", "lcet10.txt"])
    alice = samples[0]
    samples.append(type(alice)("alice29.txt (CRLF -> LF)", alice.cls, alice.data.replace(b"\r\n", b"\n")))
    for s in samples:
        ids = tok.encode(s.data)[:1024]
        logits = m.forward(np.array([EOT_ID, *ids[:-1]]))
        x = logits.astype(np.float64)
        x -= x.max(axis=1, keepdims=True)
        lse = np.log(np.exp(x).sum(axis=1))
        nll = lse - x[np.arange(len(ids)), ids]
        print(f"{s.name}: mean loss {nll.mean():.3f} nats/token over {len(ids)} tokens "
              f"({nll.mean() / np.log(2) * len(ids) / len(tok.decode(ids)):.3f} bits/byte)")


if __name__ == "__main__":
    main()
