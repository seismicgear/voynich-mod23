"""
metrics.py
Statistical helpers for evaluating a Voynich decoding.

Changes v2
-----------
* Retired single‑character Shannon entropy (invariant under 1‑to‑1 maps).
* Added `kgram_entropy` for k ≥ 2 (digram/trigram entropy).
* Added `gzip_size` as a compression‑based structure check.
"""

import math
import gzip
import io
from collections import Counter
import numpy as np

# -------------------------------------------------------------------------- #
#  n‑gram‑based metrics
# -------------------------------------------------------------------------- #
def kgram_entropy(text: str, k: int = 2) -> float:
    """
    Shannon entropy over overlapping k‑grams.
    k = 2  -> digram entropy
    k = 3  -> trigram entropy
    Returns bits per *k‑gram* (divide by k for bits/char if you like).
    """
    if len(text) < k:
        return 0.0
    grams = Counter(text[i : i + k] for i in range(len(text) - k + 1))
    n = sum(grams.values())
    return -sum((v / n) * math.log2(v / n) for v in grams.values())


def ngram_counter(text: str, n: int = 3) -> Counter:
    """Return a Counter of character n‑grams of length *n*."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    """Cosine similarity between two Counter vectors."""
    keys = set(counter_a) | set(counter_b)
    v1 = np.array([counter_a.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([counter_b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if denom == 0 else float(v1.dot(v2) / denom)


# -------------------------------------------------------------------------- #
#  Compression‑based metric (optional but fast and intuitive)
# -------------------------------------------------------------------------- #
def gzip_size(text: str) -> int:
    """
    Return the byte size of `text` after gzip compression.
    Smaller size → more repeating patterns.
    """
    with io.BytesIO() as buf:
        with gzip.GzipFile(fileobj=buf, mode="w") as f:
            f.write(text.encode())
        return len(buf.getvalue())
