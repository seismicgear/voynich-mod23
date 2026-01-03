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
