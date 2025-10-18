"""
run_experiment.py
  1. Loads an EVA transcription file.
  2. Applies decoder.py mapping (inverse mod‑23).
  3. Scores the output with TWO permutation‑sensitive metrics:
       • gzip compression size
       • trigram cosine similarity vs. a Latin corpus
  4. Builds a Monte‑Carlo baseline with 10 000 random alphabets
     and prints p‑values.

Edit EVA_PATH, LATIN_PATH, or N_ITER as needed.
"""

import random
import pathlib
import re

import decoder as dec
from metrics import ngram_counter, cosine_similarity, gzip_size

# --------------------------------------------------------------------------
EVA_PATH   = "data/voynich_eva_takahashi.txt"   # ← adjust if needed
LATIN_PATH = "data/latin_reference.txt"         # ← adjust if needed
N_ITER     = 10_000

# --------------------------------------------------------------------------
def load_eva(path: str):
    """Return a list of EVA tokens (lowercase a‑z only)."""
    raw = pathlib.Path(path).read_text().split()
    return [w for w in raw if re.fullmatch(r"[a-z]+", w)]

def load_latin(path: str):
    return pathlib.Path(path).read_text().lower()

# --------------------------------------------------------------------------
def main():
    eva_words  = load_eva(EVA_PATH)
    latin_text = load_latin(LATIN_PATH)

    # ---------- decode Voynich --------------------------------------------
    decoded_words = dec.decode_list(eva_words)
    decoded = "".join(decoded_words)

    # gzip size (smaller = more structure)
    size_obs = gzip_size(decoded)

    # trigram cosine similarity vs. Latin (higher = closer)
    tri_obs  = ngram_counter(decoded, 3)
    tri_lat  = ngram_counter(latin_text, 3)
    sim_obs  = cosine_similarity(tri_obs, tri_lat)

    print(f"gzip size (observed)         : {size_obs} bytes")
    print(f"trigram cosine vs. Latin (obs): {sim_obs:.4f}")

    # ---------- Monte‑Carlo baseline --------------------------------------
    latin_alphabet = list(dec.LATIN_23)
    size_null, sim_null = [], []

    for _ in range(N_ITER):
        shuffle_map = dict(
            zip(latin_alphabet, random.sample(latin_alphabet, dec.MOD))
        )

        trans_table = str.maketrans(shuffle_map)
        rand_str = decoded.translate(trans_table)

        size_null.append(gzip_size(rand_str))
        sim_null.append(
            cosine_similarity(ngram_counter(rand_str, 3), tri_lat)
        )

    size_p = (sum(s <= size_obs for s in size_null) + 1) / (N_ITER + 1)
    sim_p  = (sum(c >= sim_obs for c in sim_null) + 1) / (N_ITER + 1)

    print(f"p‑value (gzip smaller)  : {size_p:.4f}")
    print(f"p‑value (cosine higher) : {sim_p:.4f}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
