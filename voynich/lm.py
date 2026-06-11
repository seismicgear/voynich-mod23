"""
lm.py — smoothed character n-gram language model.

The model is trained on a cleaned reference corpus (lowercase a-z plus
space) and exposes a flat log-probability table indexed by packed n-gram
id, so the annealer can score a whole candidate decipherment with a
single vectorized gather.

Smoothing is recursive interpolation: the conditional distribution of
order o is blended with the order o-1 distribution, bottoming out at a
Laplace-smoothed unigram.  This keeps unseen n-grams finite and ranks
"almost right" decipherments above garbage, which is what the annealer
needs to climb.
"""

from __future__ import annotations

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
A = len(ALPHABET)  # 27
SPACE_ID = ALPHABET.index(" ")
_CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}


def encode_text(text: str) -> np.ndarray:
    """Map text to alphabet ids, dropping other characters and collapsing
    runs of whitespace into single spaces."""
    out = []
    prev_space = True
    for ch in text.lower():
        if ch.isspace():
            if not prev_space:
                out.append(SPACE_ID)
                prev_space = True
            continue
        idx = _CHAR_TO_ID.get(ch)
        if idx is not None:
            out.append(idx)
            prev_space = False
    while out and out[-1] == SPACE_ID:
        out.pop()
    return np.array(out, dtype=np.int64)


def decode_ids(ids: np.ndarray) -> str:
    return "".join(ALPHABET[i] for i in ids)


class CharNgramModel:
    """Interpolated character n-gram model over the 27-symbol alphabet.

    Attributes:
        order: n-gram order (3 or 4 are sensible).
        logp: flat float64 array of size A**order with log2 conditional
              probabilities; index = c0*A^(n-1) + c1*A^(n-2) + ... + c(n-1).
    """

    def __init__(self, order: int = 4, alpha: float = 0.4):
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        self.alpha = alpha
        self.logp: np.ndarray | None = None

    def fit(self, text: str) -> "CharNgramModel":
        ids = encode_text(text)
        if len(ids) < self.order * 10:
            raise ValueError("training text is too short")

        # cond[o] has shape (A**o, A): P(c | previous o chars), built
        # bottom-up so each order interpolates with the one below it.
        uni = np.bincount(ids, minlength=A).astype(np.float64)
        cond = (uni + 1.0) / (uni.sum() + A)  # shape (A,) == (A**0, A)

        n = len(ids)
        for o in range(1, self.order):
            ctx_size = A**o
            # Packed (context, char) index for every position with o chars
            # of history.
            idx = np.zeros(n - o, dtype=np.int64)
            for j in range(o):
                idx = idx * A + ids[j : n - o + j]
            idx = idx * A + ids[o:]
            counts = np.bincount(idx, minlength=ctx_size * A).astype(np.float64)
            counts = counts.reshape(ctx_size, A)
            ctx_totals = counts.sum(axis=1, keepdims=True)

            lower = cond.reshape(-1, A)
            # Context (c0..c_{o-1}) backs off to (c1..c_{o-1}), i.e. the
            # packed context index modulo A**(o-1); tiling the lower-order
            # table vertically reproduces exactly that row layout.
            backoff = np.tile(lower, (ctx_size // lower.shape[0], 1))
            k = self.alpha * A
            cond = (counts + k * backoff) / (ctx_totals + k)

        full = cond.reshape(A ** (self.order - 1), A)
        self.logp = np.log2(full).reshape(-1)
        return self

    # ---- scoring -------------------------------------------------------

    def ngram_indices(self, ids: np.ndarray) -> np.ndarray:
        """Packed n-gram index for every position, padding the start with
        spaces so every character is scored."""
        padded = np.concatenate(
            [np.full(self.order - 1, SPACE_ID, dtype=np.int64), ids]
        )
        n = len(ids)
        idx = np.zeros(n, dtype=np.int64)
        for j in range(self.order):
            idx = idx * A + padded[j : j + n]
        return idx

    def score_ids(self, ids: np.ndarray) -> float:
        """Mean log2 probability per character."""
        if self.logp is None:
            raise RuntimeError("model is not fitted")
        if len(ids) == 0:
            return float("-inf")
        return float(self.logp[self.ngram_indices(ids)].mean())

    def score_text(self, text: str) -> float:
        return self.score_ids(encode_text(text))
