"""decoder.py
Basic utilities for decoding Voynich EVA words with a provisional mod‑23 inverse cipher.
Edit `glyph_to_num` to match the exact numeric assignments you wish to test.
"""

import re

# --- Provisional glyph→number table ---------------------------------------
# Replace / extend as appropriate for your research.
DEFAULT_GLYPH_TO_NUM = {
    # Multigraphs from Codebook
    'chedy': 3,   # -> H
    'dar': 19,    # -> R
    'shedy': 4,   # -> F
    'air': 18,    # -> I
    'ar': 19,     # -> R
    'qokedy': 2,  # -> M
    'shol': 10,   # -> G
    'chody': 12,  # -> B
    'qoty': 5,    # -> O
    'chol': 9,    # -> S
    'daiin': 7,   # -> K
    'aiin': 17,   # -> T
    'chedal': 13, # -> Q
    'cheedy': 11, # -> X
    'oty': 6,     # -> D
    'qokeedy': 1, # -> A

    # Fallback single letters
    'q':  1, 'o':  2, 'k':  3, 'e':  4, 'd':  5, 'y':  6,
    'a':  7, 'i':  8, 'r':  9, 's': 10, 'h': 11, 'c': 12,
    't': 13, 'l': 14, 'n': 15, 'm': 16, 'p': 17, 'g': 18,
    'f': 19, 'x': 20, 'b': 21, 'v': 22, 'z': 23
}

# Kept for backward compatibility if needed, but usage is discouraged in favor of Mod23Decoder
glyph_to_num = DEFAULT_GLYPH_TO_NUM
LATIN_23 = "ABCDEFGHIKLMNOPQRSTVXYZ"
MOD = 23


class Mod23Decoder:
    MOD = 23
    ALPHABET = LATIN_23

    def __init__(self, glyph_to_num: dict[str, int] = None):
        self.glyph_to_num = glyph_to_num if glyph_to_num is not None else DEFAULT_GLYPH_TO_NUM
        # Sort tokens by length (longest match first)
        self.sorted_tokens = sorted(self.glyph_to_num.keys(), key=len, reverse=True)

        self._inv_cache = {n: self._safe_mod_inv(n, self.MOD) for n in range(1, self.MOD + 1)}
        self.num_to_latin = {i: self.ALPHABET[i - 1] for i in range(1, self.MOD)}
        self.num_to_latin[23] = "" # Null

    def _safe_mod_inv(self, n: int, mod: int) -> int:
        """Return the modular inverse of ``n`` or ``n`` itself if not invertible."""
        try:
            return pow(n, -1, mod)
        except ValueError:
            return n

    def tokenize_eva(self, eva_word: str) -> list[str]:
        """Greedy tokenizer (longest match first)."""
        tokens = []
        i = 0
        while i < len(eva_word):
            for tok in self.sorted_tokens:
                if eva_word.startswith(tok, i):
                    tokens.append(tok)
                    i += len(tok)
                    break
            else:
                bad = eva_word[i]
                raise ValueError(
                    f"Unknown glyph '{bad}' at position {i} in EVA word '{eva_word}'"
                )
        return tokens

    def decode_word(self, eva_word: str) -> str:
        tokens = self.tokenize_eva(eva_word)
        nums   = [self.glyph_to_num[t] for t in tokens]
        invs   = [self._inv_cache[n] for n in nums]
        return "".join(self.num_to_latin[i] for i in invs)


# Functional wrappers for backward compatibility with existing code
_default_decoder = Mod23Decoder()

def decode_word(eva_word: str) -> str:
    return _default_decoder.decode_word(eva_word)

def decode_list(words):
    return [decode_word(w) for w in words]
