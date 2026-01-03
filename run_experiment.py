"""
run_experiment.py
  1. Loads an EVA transcription file.
  2. Applies decoder.py mapping (inverse mod‑23).
  3. Scores the output with metrics:
       • gzip compression size
       • trigram cosine similarity vs. a Latin corpus
       • Shannon Entropy
       • Index of Coincidence
  4. Builds a Monte‑Carlo baseline with 10 000 random alphabets
     and prints p‑values.

Edit EVA_PATH, LATIN_PATH, or N_ITER as needed.
"""

import random
import pathlib
import re

import decoder as dec
from metrics import ngram_counter, cosine_similarity, gzip_size, shannon_entropy, index_of_coincidence

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
       """Return the Latin reference corpus as an uppercase, letters-only string."""
       text = pathlib.Path(path).read_text().upper()
       return re.sub(r"[^A-Z]", "", text)
# --------------------------------------------------------------------------
def main():
    print("Starting experiment...")
    eva_words  = load_eva(EVA_PATH)
    print(f"Loaded {len(eva_words)} EVA words.")
    latin_text = load_latin(LATIN_PATH)
    print(f"Loaded Latin text ({len(latin_text)} chars).")

    # ---------- decode Voynich --------------------------------------------
    decoded_words = dec.decode_list(eva_words)
    decoded = "".join(decoded_words).upper()

    # gzip size (smaller = more structure)
    size_obs = gzip_size(decoded)

    # trigram cosine similarity vs. Latin (higher = closer)
    tri_obs  = ngram_counter(decoded, 3)
    tri_lat  = ngram_counter(latin_text, 3)
    sim_obs  = cosine_similarity(tri_obs, tri_lat)

    # Shannon Entropy
    ent_obs = shannon_entropy(decoded)

    # Index of Coincidence
    ioc_obs = index_of_coincidence(decoded)

    print(f"gzip size (observed)         : {size_obs} bytes")
    print(f"trigram cosine vs. Latin (obs): {sim_obs:.4f}")
    print(f"Shannon Entropy (observed)    : {ent_obs:.4f}")
    print(f"Index of Coincidence (observed): {ioc_obs:.4f}")

    # ---------- Monte‑Carlo baseline --------------------------------------
    latin_alphabet = list(dec.LATIN_23)
    size_null, sim_null, ent_null, ioc_null = [], [], [], []

    print(f"Running Monte Carlo with {N_ITER} iterations...")
    for i in range(N_ITER):
        if i % 1000 == 0:
            print(f"Iteration {i}")
        shuffle_map = dict(
            zip(latin_alphabet, random.sample(latin_alphabet, dec.MOD))
        )

        # Handle the 23rd character mapping if it's involved in shuffle
        # In current decoder, LATIN_23 has 23 chars. dec.MOD is 23.
        # But decoded string only contains letters from num_to_latin.
        # num_to_latin[23] is "".
        # So decoded string has characters from A-Z (excluding J,U,W) and potentially V, X, Y, Z?
        # LATIN_23 = "ABCDEFGHIKLMNOPQRSTVXYZ"
        # The shuffle should permute these letters.

        trans_table = str.maketrans(shuffle_map)
        rand_str = decoded.translate(trans_table)

        size_null.append(gzip_size(rand_str))
        sim_null.append(
            cosine_similarity(ngram_counter(rand_str, 3), tri_lat)
        )
        # Entropy and IoC are invariant under monoalphabetic substitution.
        # We skip calculating them in the loop to save time.
        # ent_null.append(shannon_entropy(rand_str))
        # ioc_null.append(index_of_coincidence(rand_str))

    size_p = (sum(s <= size_obs for s in size_null) + 1) / (N_ITER + 1)
    sim_p  = (sum(c >= sim_obs for c in sim_null) + 1) / (N_ITER + 1)

    ent_lat = shannon_entropy(latin_text)
    ioc_lat = index_of_coincidence(latin_text)

    print(f"Shannon Entropy (Latin)       : {ent_lat:.4f}")
    print(f"Index of Coincidence (Latin)  : {ioc_lat:.4f}")

    print(f"p‑value (gzip smaller)  : {size_p:.4f}")
    print(f"p‑value (cosine higher) : {sim_p:.4f}")
    print(f"Shannon Entropy (Latin)       : {ent_lat:.4f}")
    print(f"Index of Coincidence (Latin)  : {ioc_lat:.4f}")

# ...
