"""
null_models.py

Implements null models for Monte Carlo testing.
"""

import random

def shuffle_text(text: str) -> str:
    """
    Null Model 1: Structure (Shuffle Text).
    Shuffles the characters of the text randomly.
    Destroys sequential structure (n-grams, compressibility) but preserves
    character frequency (Entropy, IoC).
    """
    chars = list(text)
    random.shuffle(chars)
    return "".join(chars)

def shuffle_alphabet_mapping(text: str, alphabet: str) -> str:
    """
    Null Model 2: Latin-likeness (Shuffle Alphabet).
    Performs a random monoalphabetic substitution.
    Preserves the underlying abstract structure of the text (isomorphisms),
    but randomizes the specific letter identities.

    Args:
        text: The text to transform.
        alphabet: The alphabet string (e.g. "ABC...").
                  Must cover all characters in `text`.
    """
    # Create a random permutation of the alphabet
    alphabet_list = list(alphabet)
    shuffled = alphabet_list[:]
    random.shuffle(shuffled)

    shuffle_map = dict(zip(alphabet_list, shuffled))
    trans_table = str.maketrans(shuffle_map)

    return text.translate(trans_table)

def sample_latin_windows(latin_text: str, length: int, n_samples: int, rng: random.Random) -> list[str]:
    """
    Sample random contiguous windows from the Latin corpus to serve as a baseline.
    This helps answer: "Where does the decoded text sit relative to real Latin segments of the same length?"
    """
    if len(latin_text) <= length:
        return [latin_text]

    max_start = len(latin_text) - length
    starts = [rng.randint(0, max_start) for _ in range(n_samples)]
    return [latin_text[s : s + length] for s in starts]

def random_glyph_mapping(base_map: dict[str, int]) -> dict[str, int]:
    """
    Generate a random glyph mapping by shuffling the values (1..23) of the base map.
    The keys (EVA glyphs) remain the same.
    """
    keys = list(base_map.keys())
    values = list(base_map.values())
    random.shuffle(values)
    return dict(zip(keys, values))
