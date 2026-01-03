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
    # Entropy: we usually expect natural language to have lower entropy than random?
    # Or specific entropy.
    # Actually, substitution cipher preserves entropy of the underlying symbols if 1-to-1.
    # But here we are comparing the decoded output (which is fixed) vs randomized alphabets applied to it?
    # Wait.
    # The Monte Carlo shuffles the ALPHABET.
    # If I shuffle the alphabet of the decoded text, I am effectively testing if the specific assignment of letters matters.
    # Entropy depends only on the distribution of frequencies, not on the identity of the symbols.
    # So shuffling the alphabet (substitution) does NOT change Shannon Entropy.
    # `shannon_entropy(rand_str)` should be identical to `shannon_entropy(decoded)`.
    # Let's verify this hypothesis.
    # If entropy is invariant under substitution, then p-value is meaningless (it will be 1.0 or 0.0 depending on float precision).
    # The prompt says: "Add Entropy... Add IoC... To make the experiment publishable/rigorous".
    # IoC is also invariant under monoalphabetic substitution!
    # "If your decoder works, the output's IoC should match Latin (~0.07-0.08) much better than the random shuffles."
    # Wait. Random shuffles of WHAT?
    # Usually, we compare the IoC of the *text* against the IoC of *random text* (uniform distribution).
    # But here the Monte Carlo baseline permutes the alphabet.
    # If I have a text "AAAAABBB", IoC is high.
    # If I swap A and B -> "BBBBBAAA", IoC is the same.
    # So IoC and Entropy are invariant under simple substitution.
    # Why does the prompt ask to add them to the metrics and implies they are better than random shuffles?
    # Maybe "random shuffles" means shuffling the *text* itself?
    # But the code currently does:
    # `shuffle_map = ...; rand_str = decoded.translate(trans_table)`
    # This is a substitution cipher.
    # If the user wants to compare against "random shuffles", maybe they mean we should generate random text or shuffle the words?
    # However, the current code is testing "Is this specific mapping of EVA to Latin better than other mappings?".
    # For metrics like Trigram Cosine Similarity, the mapping matters because "THE" is common but "XQZ" is not.
    # For IoC and Entropy, the mapping does NOT matter. The underlying distribution matters.
    # So these metrics check if the *structure* of Voynich (as decoded) matches Latin *structure*.
    # They should be compared to Latin values, not to the "null hypothesis of random alphabet mapping".
    # They are properties of the EVA text itself (and the tokenizer).
    # Wait, the tokenizer changes depending on `glyph_to_num`?
    # No, `glyph_to_num` maps glyphs to numbers. The tokenizer (regex) splits the string.
    # The tokenizer uses `glyph_to_num.keys()`. If I change the keys, tokenizer changes.
    # But in the Monte Carlo loop, we are only permuting the Latin letters *after* decoding.
    # So the structure (sequence of symbols) is fixed.
    # Thus IoC and Entropy will be constant across all iterations of the Monte Carlo loop.

    # So, printing p-values for Entropy and IoC based on alphabet permutation is useless.
    # Instead, I should just print the observed values and compare them to Latin reference values.
    # The prompt says "If your decoder works, the output's IoC should match Latin (~0.07-0.08) much better than the random shuffles."
    # "much better than the random shuffles" -> This implies the random shuffles *should* have different IoC?
    # That is only possible if "random shuffles" means shuffling the characters in the text (destroying order/frequencies if done uniformly? No).
    # Or maybe "random shuffles" means decoding with a random mapping *before* tokenizing? No, decoding happens first.

    # Maybe the user implies that the "decoded" text might be garbage?
    # If I decode "chedy" -> "H".
    # If I decode with random map, "chedy" -> "X".
    # The IoC of "H...X..." vs "X...H..." is the same.

    # Perhaps the user is confused about IoC invariance, OR I am missing something.
    # UNLESS the 23rd character (Null) handling interacts with it.
    # If I map some glyphs to Null, they disappear.
    # Different mappings might map different glyphs to Null?
    # In my implementation, `glyph_to_num` is fixed. `num_to_latin[23]` is "".
    # `glyph_to_num` maps specific glyphs to specific numbers (1-23).
    # The Monte Carlo loop:
    # `shuffle_map` maps `latin_alphabet` (A-Z...) to `random.sample(latin_alphabet, MOD)`.
    # `decoded` string is already generated using the "correct" mapping.
    # Then `rand_str` substitutes letters in `decoded`.
    # This preserves length and pattern of repeats.
    # So IoC and Entropy are indeed invariant.

    # However, the user asked to "Add Entropy" and "Add IoC".
    # And "Standardize Input".
    # Maybe I should just display them as "Observed" metrics.
    # And maybe calculate Latin IoC and Entropy for comparison.

    # Wait, if I permute the alphabet, trigram scores change (because trigrams depend on specific letters).
    # Gzip size might change slightly because Huffman coding depends on frequencies, but if it's just substitution, the frequency distribution (counts) is the same, so entropy is same. Gzip also uses matching sequences (LZ77). If I map A->B and B->A, the patterns "ABAB" become "BABA", repetitions are preserved. So gzip size should be roughly invariant too?
    # Actually, `gzip_size` variation is small but nonzero due to internal details of Deflate (e.g. fixed Huffman trees vs dynamic, header bits, etc), but effectively it measures Kolmogorov complexity which is invariant under simple substitution.
    # But `gzip` might treat some byte values differently (e.g. ASCII vs others)? No, here it's all text.
    # In previous runs (if I had them), gzip size p-value is used.
    # Let's verify if gzip size varies significantly.

    # Anyway, I will add IoC and Entropy to the output.
    # I will also compute them for the Latin text for comparison.
    # I won't calculate p-values for them because they should be invariant, and it would be confusing to see p=1.0.
    # Unless... "random shuffles" refers to shuffling the *words* or *glyphs*?
    # The `run_experiment.py` code clearly does substitution cipher shuffle.

    # I will implement printing the values. I will also add them to the Monte Carlo loop just to see (or maybe skip to save time if they are constant).
    # I'll stick to the user instructions: "Add Entropy", "Add IoC".

    ent_lat = shannon_entropy(latin_text)
    ioc_lat = index_of_coincidence(latin_text)

    print(f"Shannon Entropy (Latin)       : {ent_lat:.4f}")
    print(f"Index of Coincidence (Latin)  : {ioc_lat:.4f}")

    # I'll comment out p-value calculation for IoC/Entropy or just not do it,
    # as it doesn't make sense for substitution test.
    # But maybe I should check if gzip size varies.

    # Wait, if the user thinks IoC detects "monoalphabetic substitution", then IoC is useful to check if the *input* is monoalphabetic.
    # Comparing it to "random shuffles" is weird.
    # Random text (uniform) has IoC ~ 1/26 = 0.038.
    # English/Latin has ~0.065-0.075.
    # So we compare Obs to Latin.

    # So my plan for `run_experiment.py` is:
    # Print Obs Entropy, Obs IoC.
    # Print Latin Entropy, Latin IoC.
    # (Optional) Print comparison or something.

    # I will modify the loop to NOT calculate IoC/Entropy for nulls, to save compute,
    # unless I want to prove they are invariant.
    # I'll leave them out of the loop for now.

    # Re-reading prompt: "If your decoder works, the output's IoC should match Latin (~0.07-0.08) much better than the random shuffles."
    # This sentence is key. "better than the random shuffles".
    # If "random shuffles" refers to the Monte Carlo substitution, the user might be mistaken about IoC properties.
    # OR, "random shuffles" means something else.
    # But given the existing code does substitution shuffle, I assume the user *thinks* it varies or I am missing a nuance (e.g. 23rd char null interaction).
    # If the mapping shuffles include the Null character...
    # My `num_to_latin` maps 23 -> "".
    # The `decoded` string is formed. It has NO characters from 23.
    # The `latin_alphabet` variable in `run_experiment.py` comes from `dec.LATIN_23`.
    # `LATIN_23` has 23 chars.
    # `decoded` uses a subset (if 23 is used in `glyph_to_num` but maps to "", it's not in `decoded`).
    # If `decoded` only has chars from `LATIN_23` (excluding 'Z' if 'Z' corresponds to 23? No, `LATIN_23[22]` is 'Z'. `num_to_latin[23]` is "").
    # `num_to_latin` uses `LATIN_23` for 1..22.
    # So `decoded` contains `LATIN_23[0..21]`.
    # `LATIN_23` contains `LATIN_23[0..22]`.
    # The shuffle maps `LATIN_23` -> `permuted(LATIN_23)`.
    # If `decoded` has 'A' (index 0), and map sends 'A' -> 'Z' (index 22).
    # 'Z' is in the shuffled output.
    # 'Z' was NOT in the input (if 23 was the only source of 'Z', but wait. `num_to_latin[22]` is `LATIN_23[21]` which is 'Y'. `num_to_latin[23]` is `LATIN_23[22]`? No.
    # `LATIN_23` = "A...Z". Length 23.
    # `num_to_latin`:
    # i=1..22: `LATIN_23[0]` .. `LATIN_23[21]`.
    # `LATIN_23[22]` is 'Z'. It is NOT used in `num_to_latin` 1..22.
    # So 'Z' only appears if `num_to_latin[23]` is used.
    # But `num_to_latin[23]` is "".
    # So 'Z' never appears in `decoded`.
    # But the shuffle map includes 'Z'.
    # So `rand_str` MIGHT contain 'Z'.
    # If `decoded` has 'A', and 'A' maps to 'Z'.
    # Then `rand_str` has 'Z'.
    # Does this change IoC?
    # IoC depends on counts: sum(n*(n-1)).
    # If 'A' appears 10 times, and maps to 'Z' (10 times). The contribution to sum is 10*9.
    # It doesn't matter what character it maps to, as long as it's 1-to-1.
    # So IoC is invariant.

    # I will ignore the "better than random shuffles" part regarding IoC variance and just implement the calculation and display.
    # I will assume the user wants to see the metric.

    print(f"p‑value (gzip smaller)  : {size_p:.4f}")
    print(f"p‑value (cosine higher) : {sim_p:.4f}")
    print(f"Shannon Entropy (Latin)       : {ent_lat:.4f}")
    print(f"Index of Coincidence (Latin)  : {ioc_lat:.4f}")

# ...
