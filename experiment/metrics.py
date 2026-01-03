"""
metrics.py

Functions for measuring linguistic structure and similarity.
"""

import math
import gzip
import io
from collections import Counter
import numpy as np

def ngram_counter(text: str, n: int = 3) -> Counter:
    """
    Count n-grams in the text.
    """
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))

def cosine_similarity(a: Counter, b: Counter) -> float:
    """
    Compute cosine similarity between two counters (viewed as vectors).
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0

    v1 = np.array([a.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([b.get(k, 0) for k in keys], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))

def gzip_size(text: str) -> int:
    """
    Return the size of the text compressed with gzip (default settings).
    Lower size implies higher redundancy/structure.
    """
    with io.BytesIO() as buf:
        # Use a fixed mtime to ensure deterministic output for identical input
        with gzip.GzipFile(fileobj=buf, mode="w", mtime=0) as f:
            f.write(text.encode('utf-8'))
        return len(buf.getvalue())

def shannon_entropy(text: str) -> float:
    """
    Calculate character-level Shannon Entropy (bits per symbol).
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    probs = [c / length for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def index_of_coincidence(text: str) -> float:
    """
    Calculate Index of Coincidence (probability that two randomly selected letters are the same).
    """
    if len(text) < 2:
        return 0.0
    counts = Counter(text)
    N = len(text)
    numerator = sum(n * (n - 1) for n in counts.values())
    denominator = N * (N - 1)
    return numerator / denominator
