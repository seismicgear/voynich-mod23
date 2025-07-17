"""run_experiment.py
Skeleton script that:
  1. Loads an EVA transcription file.
  2. Applies decoder.py mapping.
  3. Computes entropy & Latin‑trigram similarity.
  4. Performs Monte‑Carlo shuffle test.
Fill in the paths and tweak N_ITER as desired.
"""

import random, pathlib, re
import numpy as np

import decoder as dec
import metrics as met

# --------------------------------------------------------------------------
EVA_PATH      = 'data/voynich_eva_takahashi.txt'  # TODO: adjust
LATIN_PATH    = 'data/latin_reference.txt'        # TODO: adjust
N_ITER        = 10000

# --------------------------------------------------------------------------
def load_eva(path):
    raw = pathlib.Path(path).read_text().split()
    return [w for w in raw if re.fullmatch(r'[a-z]+', w)]

def load_latin(path):
    return pathlib.Path(path).read_text().lower()

# --------------------------------------------------------------------------
def main():
    eva_words = load_eva(EVA_PATH)
    latin_txt = load_latin(LATIN_PATH)

    decoded = ''.join(dec.decode_list(eva_words))
    ent_obs  = met.shannon_entropy(decoded)

    trigrams_obs = met.ngram_counter(decoded, 3)
    trigrams_lat = met.ngram_counter(latin_txt, 3)
    sim_obs = met.cosine_similarity(trigrams_obs, trigrams_lat)

    print(f'Observed entropy: {ent_obs:.3f} bits/char')
    print(f'Observed trigram‑cosine similarity vs Latin: {sim_obs:.4f}')

    # --- Monte‑Carlo control ---------------------------------------------
    latin_alphabet = list(dec.LATIN_23)
    ent_null, sim_null = [], []

    for _ in range(N_ITER):
        shuffle_map = dict(zip(range(1, dec.MOD + 1),
                               random.sample(latin_alphabet, dec.MOD)))

        def shuffle_decode(num_list):
            return ''.join(shuffle_map[i] for i in num_list)

        rand_chars = []
        for w in eva_words:
            nums = [dec.glyph_to_num[g] for g in w if g in dec.glyph_to_num]
            rand_chars.append(shuffle_decode(nums))
        rand_str = ''.join(rand_chars)

        ent_null.append(met.shannon_entropy(rand_str))
        sim_null.append(met.cosine_similarity(
            met.ngram_counter(rand_str, 3), trigrams_lat))

    ent_p = (sum(e <= ent_obs for e in ent_null) + 1) / (N_ITER + 1)
    sim_p = (sum(s >= sim_obs for s in sim_null) + 1) / (N_ITER + 1)

    print(f'p‑value (entropy lower):  {ent_p:.4f}')
    print(f'p‑value (trigram higher): {sim_p:.4f}')

if __name__ == '__main__':
    main()
