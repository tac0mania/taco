"""Float GPT-2 small in NumPy: the phase 0 baseline model (plan §6, §12).

Two computation paths give the logits:

- `forward(tokens)`: the full-sequence causal pass (teacher forcing). One call gives the
  logits at every position. The encoder uses it for measurement.
- `KVState` and `step(...)`: the incremental pass with a key/value cache. One call adds one
  token to each stream of a batch. A decoder must use a path of this kind, because it knows a
  token only after it decodes it.

The two paths do the same mathematics, but in a different order of float operations. Thus the
last bits of the logits can differ. Phase 1 measures this difference (`baseline.phase1`).

NOTE: this is not the prediction path of Taco. It uses floats, so it is a measurement tool only
(CLAUDE.md §1 rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import safe_open

from baseline._paths import MODEL_DIR

N_LAYER = 12
N_HEAD = 12
D_MODEL = 768
D_HEAD = D_MODEL // N_HEAD
N_CTX = 1024
VOCAB = 50257
LN_EPS = 1e-5


@dataclass
class Block:
    """The weights of one transformer layer."""

    ln1_g: np.ndarray
    ln1_b: np.ndarray
    attn_w: np.ndarray
    attn_b: np.ndarray
    proj_w: np.ndarray
    proj_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    fc_w: np.ndarray
    fc_b: np.ndarray
    mlp_w: np.ndarray
    mlp_b: np.ndarray


class GPT2:
    """GPT-2 small with all weights in one float dtype (float32 or float64)."""

    def __init__(self, dtype: type = np.float32, model_dir: Path = MODEL_DIR) -> None:
        self.dtype = np.dtype(dtype)
        with safe_open(str(model_dir / "model.safetensors"), "np") as f:

            def get(name: str) -> np.ndarray:
                return np.ascontiguousarray(f.get_tensor(name), dtype=self.dtype)

            self.wte = get("wte.weight")
            self.wpe = get("wpe.weight")
            self.lnf_g = get("ln_f.weight")
            self.lnf_b = get("ln_f.bias")
            self.blocks = []
            for i in range(N_LAYER):
                p = f"h.{i}."
                self.blocks.append(
                    Block(
                        get(p + "ln_1.weight"), get(p + "ln_1.bias"),
                        get(p + "attn.c_attn.weight"), get(p + "attn.c_attn.bias"),
                        get(p + "attn.c_proj.weight"), get(p + "attn.c_proj.bias"),
                        get(p + "ln_2.weight"), get(p + "ln_2.bias"),
                        get(p + "mlp.c_fc.weight"), get(p + "mlp.c_fc.bias"),
                        get(p + "mlp.c_proj.weight"), get(p + "mlp.c_proj.bias"),
                    )
                )
        self.wte_t = self.wte.T  # A view; BLAS reads it as a transposed matrix.

    # Shared operations ---------------------------------------------------------------------

    def _ln(self, x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        xc = x - mean
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc / np.sqrt(var + self.dtype.type(LN_EPS)) * g + b

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        # The tanh form of GELU, as in the published GPT-2 code.
        c = self.dtype.type(np.sqrt(2.0 / np.pi))
        return self.dtype.type(0.5) * x * (1 + np.tanh(c * (x + self.dtype.type(0.044715) * x**3)))

    @staticmethod
    def _softmax_last(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _mlp(self, blk: Block, x: np.ndarray) -> np.ndarray:
        h = self._gelu(self._ln(x, blk.ln2_g, blk.ln2_b) @ blk.fc_w + blk.fc_b)
        return h @ blk.mlp_w + blk.mlp_b

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return self._ln(x, self.lnf_g, self.lnf_b) @ self.wte_t

    # Full-sequence path --------------------------------------------------------------------

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Return the logits [T, VOCAB] for a sequence of T <= 1024 tokens (causal)."""
        t = len(tokens)
        if not 0 < t <= N_CTX:
            raise ValueError(f"sequence length {t} is not in 1..{N_CTX}")
        x = self.wte[tokens] + self.wpe[:t]
        scale = self.dtype.type(1.0 / np.sqrt(D_HEAD))
        mask = np.triu(np.full((t, t), -np.inf, dtype=self.dtype), k=1)
        for blk in self.blocks:
            qkv = self._ln(x, blk.ln1_g, blk.ln1_b) @ blk.attn_w + blk.attn_b
            q, k, v = (qkv[:, i * D_MODEL : (i + 1) * D_MODEL] for i in range(3))
            q = q.reshape(t, N_HEAD, D_HEAD).transpose(1, 0, 2)
            k = k.reshape(t, N_HEAD, D_HEAD).transpose(1, 2, 0)
            v = v.reshape(t, N_HEAD, D_HEAD).transpose(1, 0, 2)
            att = self._softmax_last((q @ k) * scale + mask)
            y = (att @ v).transpose(1, 0, 2).reshape(t, D_MODEL)
            x = x + (y @ blk.proj_w + blk.proj_b)
            x = x + self._mlp(blk, x)
        return self._logits(x)

    # Incremental path ----------------------------------------------------------------------

    def new_state(self, batch: int) -> KVState:
        """Return an empty key/value cache for `batch` independent streams."""
        shape = (N_LAYER, batch, N_HEAD, N_CTX, D_HEAD)
        return KVState(
            k=np.zeros(shape, dtype=self.dtype),
            v=np.zeros(shape, dtype=self.dtype),
            pos=np.zeros(batch, dtype=np.int64),
        )

    def step(self, state: KVState, tokens: np.ndarray) -> np.ndarray:
        """Add one token to each stream and return the logits [B, VOCAB] for the next token.

        Stream b puts its token at position `state.pos[b]`, then the position increases by 1.
        To start a new context, set `state.pos[b] = 0` before the call. The attention of each
        stream reads only its own positions 0..pos[b], in a loop, so that the attention result
        of a stream does not depend on the other streams in the batch.
        """
        b = len(tokens)
        pos = state.pos.copy()
        if np.any(pos >= N_CTX):
            raise ValueError("a stream is past the context length; reset it first")
        x = self.wte[tokens] + self.wpe[pos]
        scale = self.dtype.type(1.0 / np.sqrt(D_HEAD))
        for li, blk in enumerate(self.blocks):
            qkv = self._ln(x, blk.ln1_g, blk.ln1_b) @ blk.attn_w + blk.attn_b
            q = qkv[:, :D_MODEL].reshape(b, N_HEAD, D_HEAD)
            kc, vc = state.k[li], state.v[li]
            kc[np.arange(b), :, pos] = qkv[:, D_MODEL : 2 * D_MODEL].reshape(b, N_HEAD, D_HEAD)
            vc[np.arange(b), :, pos] = qkv[:, 2 * D_MODEL :].reshape(b, N_HEAD, D_HEAD)
            y = np.empty((b, N_HEAD, D_HEAD), dtype=self.dtype)
            for s in range(b):
                n = pos[s] + 1
                scores = (kc[s, :, :n] @ q[s][:, :, None])[:, :, 0] * scale  # [H, n]
                att = self._softmax_last(scores)
                y[s] = (att[:, None, :] @ vc[s, :, :n])[:, 0, :]
            x = x + (y.reshape(b, D_MODEL) @ blk.proj_w + blk.proj_b)
            x = x + self._mlp(blk, x)
        state.pos = pos + 1
        return self._logits(x)


@dataclass
class KVState:
    """Key/value cache for a batch of streams: [layer, stream, head, position, d_head]."""

    k: np.ndarray
    v: np.ndarray
    pos: np.ndarray
