"""metrics.py
Statistical helpers used to evaluate a proposed Voynich decoding.
"""

import math
from collections import Counter
import numpy as np

# --------------------------------------------------------------------------
def shannon_entropy(text: str) -> float:
    """Character‑level Shannon entropy (bits)."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())

# --------------------------------------------------------------------------
def ngram_counter(text: str, n: int = 3) -> Counter:
    """Return Counter of character n‑grams of length *n*."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))

# --------------------------------------------------------------------------
def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    """Cosine similarity between two Counter vectors."""
    keys = set(counter_a) | set(counter_b)
    v1 = np.array([counter_a.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([counter_b.get(k, 0) for k in keys], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if denom == 0 else float(v1.dot(v2) / denom)