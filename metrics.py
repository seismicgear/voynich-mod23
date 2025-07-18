import math, gzip, io
from collections import Counter
import numpy as np

# ---------- metrics that CAN move under permutation -----------------
def ngram_counter(text: str, n: int = 3) -> Counter:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))

def cosine_similarity(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    v1 = np.array([a.get(k, 0) for k in keys], float)
    v2 = np.array([b.get(k, 0) for k in keys], float)
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if d == 0 else float(v1.dot(v2) / d)

def gzip_size(text: str) -> int:
    with io.BytesIO() as buf:
        with gzip.GzipFile(fileobj=buf, mode="w") as f:
            f.write(text.encode())
        return len(buf.getvalue())
