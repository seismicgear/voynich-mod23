"""
run_experiment.py
  1. Loads an EVA transcription file.
  2. Applies decoder.py mapping.
  3. Computes *digram* entropy & Latin‑trigram similarity.
  4. Performs a Monte‑Carlo shuffle test.

Fill in EVA_PATH / LATIN_PATH and tweak N_ITER as needed.
"""

import random
import pathlib
import re

import decoder as dec
import metrics as met  # uses kgram_entropy, cosine_similarity, etc.

# --------------------------------------------------------------------------
EVA_PATH   = "data/voynich_eva_takahashi.txt"   # TODO: adjust
LATIN_PATH = "data/latin_reference.txt"         # TODO: adjust
N_ITER     = 10_000

# --------------------------------------------------------------------------
def load_eva(path: str):
    """Return a list of EVA tokens (lowercase a‑z)."""
    raw = pathlib.Path(path).read_text().split()
    return [w for w in raw if re.fullmatch(r"[a-z]+", w)]

def load_latin(path: str):
    return pathlib.Path(path).read_text().lower()

# --------------------------------------------------------------------------
def main():
    eva_words  = load_eva(EVA_PATH)
    latin_text = load_latin(LATIN_PATH)

    # --- Decode Voynich -----------------------------------------------------
    decoded = "".join(dec.decode_list(eva_words))

    # digram entropy (k = 2)
    ent_obs = met.kgram_entropy(decoded, k=2)

    # trigram cosine similarity vs Latin
    trig_obs = met.ngram_counter(decoded, 3)
    trig_lat = met.ngram_counter(latin_text, 3)
    sim_obs  = met.cosine_similarity(trig_obs, trig_lat)

    print(f"Observed digram entropy: {ent_obs:.3f} bits")
    print(f"Trigram cosine vs Latin:  {sim_obs:.4f}")

    # --- Monte‑Carlo null distribution -------------------------------------
    latin_alphabet = list(dec.LATIN_23)
    ent_null, sim_null = [], []

    for _ in range(N_ITER):
        shuffle_map = dict(
            zip(range(1, dec.MOD + 1), random.sample(latin_alphabet, dec.MOD))
        )

        def shuffle_decode(nums):
            return "".join(shuffle_map[i] for i in nums)

        rand_str_parts = []
        for w in eva_words:
            nums = [dec.glyph_to_num[g] for g in w if g in dec.glyph_to_num]
            rand_str_parts.append(shuffle_decode(nums))
        rand_str = "".join(rand_str_parts)

        ent_null.append(met.kgram_entropy(rand_str, k=2))
        sim_null.append(
            met.cosine_similarity(met.ngram_counter(rand_str, 3), trig_lat)
        )

    ent_p = (sum(e <= ent_obs for e in ent_null) + 1) / (N_ITER + 1)
    sim_p = (sum(s >= sim_obs for s in sim_null) + 1) / (N_ITER + 1)

    print(f"p‑value (digram entropy lower) : {ent_p:.4f}")
    print(f"p‑value (trigram similarity up): {sim_p:.4f}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
