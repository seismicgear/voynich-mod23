"""
decode.py

Implements the modular-23 inverse decoder for Voynich EVA glyphs.
"""

import re

class Mod23Decoder:
    """
    Decodes Voynich EVA text using a modular-23 inverse mapping.

    The mapping process:
    1. Tokenize EVA string into glyphs (greedy match).
    2. Map glyphs to integers 1..23.
    3. Compute modular inverse: x -> x^-1 (mod 23).
    4. Map result to Latin alphabet (A..Z, excluding J, K, U/W/etc to fit 23).

    Default alphabet: 'ABCDEFGHIKLMNOPQRSTVXYZ' (23 letters).
    """

    # Standard Latin-23 alphabet for this experiment
    ALPHABET = "ABCDEFGHIKLMNOPQRSTVXYZ"
    MOD = 23

    # Default EVA glyph -> Number mapping (1-based)
    DEFAULT_GLYPH_TO_NUM = {
        # Multigraphs (longest first)
        'qokeedy': 1, # -> A
        'qokedy': 2,  # -> M
        'chedy': 3,   # -> H
        'shedy': 4,   # -> F
        'qoty': 5,    # -> O
        'daiin': 7,   # -> K
        'chol': 9,    # -> S
        'shol': 10,   # -> G
        'cheedy': 11, # -> X
        'chody': 12,  # -> B
        'chedal': 13, # -> Q
        'aiin': 17,   # -> T

        # Single chars / shorter sequences
        'dar': 19,    # -> R
        'air': 18,    # -> I
        'ar': 19,     # -> R

        'oty': 6,     # -> D

        # Singles
        'q':  1, 'o':  2, 'k':  3, 'e':  4, 'd':  5, 'y':  6,
        'a':  7, 'i':  8, 'r':  9, 's': 10, 'h': 11, 'c': 12,
        't': 13, 'l': 14, 'n': 15, 'm': 16, 'p': 17, 'g': 18,
        'f': 19, 'x': 20, 'b': 21, 'v': 22, 'z': 23
    }

    def __init__(self, glyph_to_num: dict[str, int] = None):
        """
        Initialize the decoder.

        Args:
            glyph_to_num: Optional dictionary mapping EVA strings to integers 1..23.
                          If None, uses DEFAULT_GLYPH_TO_NUM.
        """
        self.glyph_to_num = glyph_to_num if glyph_to_num is not None else self.DEFAULT_GLYPH_TO_NUM

        # Sort tokens by length descending for greedy tokenization
        self.sorted_tokens = sorted(self.glyph_to_num.keys(), key=len, reverse=True)

        # Precompute modular inverses
        self._inv_cache = {n: self._safe_mod_inv(n, self.MOD) for n in range(1, self.MOD + 1)}

        # Map 1..23 back to letters
        self.num_to_latin = {i: self.ALPHABET[i - 1] for i in range(1, self.MOD + 1) if i <= len(self.ALPHABET)}
        # Ensure 0 mod 23 cases or out of bound don't crash, though mathematically shouldn't happen with 1..23 inputs

    def _safe_mod_inv(self, n: int, mod: int) -> int:
        """Return the modular inverse of n mod m. If not invertible, return n."""
        # For prime 23, all 1..22 are invertible. 23 (0) is not.
        try:
            return pow(n, -1, mod)
        except ValueError:
            return n

    def tokenize_eva(self, eva_word: str) -> list[str]:
        """
        Tokenize an EVA string into glyphs using greedy longest-match strategy.

        Raises:
            ValueError: If a character cannot be tokenized.
        """
        tokens = []
        i = 0
        while i < len(eva_word):
            matched = False
            for tok in self.sorted_tokens:
                if eva_word.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    matched = True
                    break

            if not matched:
                bad_char = eva_word[i]
                raise ValueError(
                    f"Unknown glyph '{bad_char}' at position {i} in EVA word '{eva_word}'"
                )
        return tokens

    def decode_word(self, eva_word: str) -> str:
        """
        Decode a single EVA word to a Latin string.
        """
        tokens = self.tokenize_eva(eva_word)
        nums = [self.glyph_to_num[t] for t in tokens]
        invs = [self._inv_cache[n] for n in nums]
        return "".join(self.num_to_latin.get(i, '?') for i in invs)
