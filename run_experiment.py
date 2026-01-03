"""
run_experiment.py
  1. Loads an EVA transcription file.
  2. Applies decoder.py mapping (inverse mod‑23).
  3. Scores the output with metrics:
       • gzip compression size
       • trigram cosine similarity vs. a Latin corpus
       • Shannon Entropy
       • Index of Coincidence
  4. Builds a Monte‑Carlo baseline with random alphabets
     and prints p‑values.

Run with: python run_experiment.py
"""

import random
import pathlib
import re
import argparse
import statistics as stats

import decoder as dec
from decoder import Mod23Decoder
from metrics import ngram_counter, cosine_similarity, gzip_size, shannon_entropy, index_of_coincidence

# --------------------------------------------------------------------------
EVA_PATH   = "data/voynich_eva_takahashi.txt"
LATIN_PATH = "data/latin_reference.txt"
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
def main(eva_path: str, latin_path: str, n_iter: int, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
        print(f"Random seed: {seed}")

    print("Starting experiment...")

    try:
        eva_words  = load_eva(eva_path)
        print(f"Loaded {len(eva_words)} EVA words.")
        latin_text = load_latin(latin_path)
        print(f"Loaded Latin text ({len(latin_text)} chars).")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # ---------- decode Voynich --------------------------------------------
    # Using the new class-based decoder
    decoder = Mod23Decoder()
    decoded_words = [decoder.decode_word(w) for w in eva_words]
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

    # ---------- Monte‑Carlo baselines --------------------------------------
    print(f"Running Monte Carlo with {n_iter} iterations...")

    # Null Model 1: Structure (Shuffle Text)
    # H0: The observed structure (gzip size) is not different from a random permutation of the text.
    size_null = []
    # Entropy and IoC are invariant under shuffling of characters if we just shuffle,
    # but Gzip depends on order.

    decoded_chars = list(decoded)

    # Null Model 2: Latin-ness (Shuffle Alphabet)
    # H0: The observed trigram similarity is not different from a random monoalphabetic substitution.
    sim_null = []
    latin_alphabet = list(Mod23Decoder.ALPHABET)

    for i in range(n_iter):
        if i % 1000 == 0:
            # Simple progress indicator
            pass # print(f"Iteration {i}")

        # --- Null 1: Shuffle Text ---
        # We shuffle the characters of the decoded text to destroy structure.
        # Note: This is computationally expensive for large texts.
        # Optimization: To make it faster, we could shuffle lines or words, but user suggested:
        # "Or shuffle words, or shuffle within lines. That actually destroys sequential structure."
        # Or "Randomize the sequence itself, not just the labels. E.g. random.shuffle(chars)"
        # Let's do random.shuffle(chars) on a copy.

        # Optimization: Shuffling a huge list every time might be slow.
        # But let's follow instructions first.
        shuffled_chars = decoded_chars[:]
        random.shuffle(shuffled_chars)
        rand_text_struct = "".join(shuffled_chars)
        size_null.append(gzip_size(rand_text_struct))

        # --- Null 2: Shuffle Alphabet ---
        shuffle_map = dict(
            zip(latin_alphabet, random.sample(latin_alphabet, dec.MOD))
        )
        trans_table = str.maketrans(shuffle_map)
        # Note: We apply this to the ORIGINAL decoded text to preserve structure but randomize identity.
        rand_text_alpha = decoded.translate(trans_table)

        sim_null.append(
            cosine_similarity(ngram_counter(rand_text_alpha, 3), tri_lat)
        )

    # ---------- Statistics ------------------------------------------------
    def print_stats(name, obs, null_dist, smaller_is_better=False):
        mean = stats.mean(null_dist)
        std = stats.pstdev(null_dist)
        z = (obs - mean) / std if std > 0 else 0

        if smaller_is_better:
            p_val = (sum(s <= obs for s in null_dist) + 1) / (len(null_dist) + 1)
        else:
            p_val = (sum(s >= obs for s in null_dist) + 1) / (len(null_dist) + 1)

        print(f"{name} obs: {obs:.4f} (null mean {mean:.4f} ± {std:.4f})")
        print(f"  Z-score: {z:.2f}")
        print(f"  p-value: {p_val:.4f}")

    print("\n--- Results ---")

    # Entropy/IoC (Invariant under substitution, so we just print observed)
    print(f"Shannon Entropy (observed)    : {ent_obs:.4f}")
    print(f"Index of Coincidence (observed): {ioc_obs:.4f}")

    print("-" * 20)
    print("Null Model 1: Random Character Shuffle (Test for Structure)")
    print_stats("gzip size", size_obs, size_null, smaller_is_better=True) # Intentionally printing float format even for size

    print("-" * 20)
    print("Null Model 2: Random Alphabet Substitution (Test for Latin-likeness)")
    print_stats("trigram cosine", sim_obs, sim_null, smaller_is_better=False)

    # Compare with Latin reference
    ent_lat = shannon_entropy(latin_text)
    ioc_lat = index_of_coincidence(latin_text)

    print("-" * 20)
    print(f"Shannon Entropy (Latin)       : {ent_lat:.4f}")
    print(f"Index of Coincidence (Latin)  : {ioc_lat:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voynich Modular-23 Experiment")
    parser.add_argument("--eva", default=EVA_PATH, help="Path to EVA transcription file")
    parser.add_argument("--latin", default=LATIN_PATH, help="Path to Latin reference file")
    parser.add_argument("--n-iter", type=int, default=N_ITER, help="Number of Monte Carlo iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    main(args.eva, args.latin, args.n_iter, args.seed)
