"""decoder.py
Basic utilities for decoding Voynich EVA words with a provisional mod‑23 inverse cipher.
Edit `glyph_to_num` to match the exact numeric assignments you wish to test.
"""

import re

MOD = 23
LATIN_23 = "ABCDEFGHIKLMNOPQRSTVXYZ"  # Classical Latin alphabet (J, U, W omitted)

# --- Provisional glyph→number table ---------------------------------------
# Replace / extend as appropriate for your research.
glyph_to_num = {
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

# Compile a regex for "Longest Match First"
sorted_tokens = sorted(glyph_to_num.keys(), key=len, reverse=True)
pattern = re.compile("|".join(map(re.escape, sorted_tokens)))

# Pre‑compute modular inverses
def _safe_mod_inv(n: int, mod: int = MOD) -> int:
    """Return the modular inverse of ``n`` or ``n`` itself if not invertible."""
    try:
        return pow(n, -1, mod)
    except ValueError:
        # 23 has no inverse modulo 23; map it to itself (Null behavior handles it later)
        return n

_inv_cache = {n: _safe_mod_inv(n, MOD) for n in range(1, MOD + 1)}

# Map 1-22 to Latin letters, 23 to empty string (Null)
num_to_latin = {i: LATIN_23[i - 1] for i in range(1, MOD)}
num_to_latin[23] = "" # Option B: Null

# --------------------------------------------------------------------------
def decode_word(eva_word: str) -> str:
    """Tokenizes EVA string and applies modular decoding."""
    tokens = pattern.findall(eva_word)

    # Check if we consumed the whole string (optional validation)
    if "".join(tokens) != eva_word:
        # Handle unmapped garbage or raise error
        # Identify the first unmapped character for the error message
        matched = "".join(tokens)
        # This check is tricky because tokens might be in different order than source if not contiguous
        # But findall returns tokens in order of appearance.
        # So "".join(tokens) should be a subsequence of eva_word.
        # If not equal, there are gaps.

        # We want to raise ValueError to satisfy the requirement "Handle unmapped garbage or raise error"
        # and to pass the existing test.
        # Let's find the first char that is not in the matched tokens?
        # Actually, let's just find where the mismatch starts.
        # Or simpler:
        for char in eva_word:
             # This is a weak check if we just iterate chars.
             pass

        # Re-scan to find the gap
        pos = 0
        for t in tokens:
            # Find t starting at or after pos
            idx = eva_word.find(t, pos)
            if idx == -1:
                # Should not happen if findall works
                raise RuntimeError("Regex logic error")
            if idx > pos:
                # There is a gap between pos and idx
                bad_char = eva_word[pos]
                raise ValueError(f"Unknown glyph '{bad_char}' at position {pos} in EVA word '{eva_word}'")
            pos = idx + len(t)

        if pos < len(eva_word):
             bad_char = eva_word[pos]
             raise ValueError(f"Unknown glyph '{bad_char}' at position {pos} in EVA word '{eva_word}'")

    nums = [glyph_to_num[t] for t in tokens]

    # Apply your inverse logic
    invs = [_inv_cache[n] for n in nums]
    return ''.join(num_to_latin[i] for i in invs)

def decode_list(words):
    """Vectorised convenience wrapper."""
    return [decode_word(w) for w in words]
