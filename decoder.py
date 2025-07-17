"""decoder.py
Basic utilities for decoding Voynich EVA words with a provisional mod‑23 inverse cipher.
Edit `glyph_to_num` to match the exact numeric assignments you wish to test.
"""

MOD = 23
LATIN_23 = "ABCDEFGHIKLMNOPQRSTVXYZ"  # Classical Latin alphabet (J, U, W omitted)

# --- Provisional glyph→number table ---------------------------------------
# Replace / extend as appropriate for your research.
glyph_to_num = {
    'q':  1, 'o':  2, 'k':  3, 'e':  4, 'd':  5, 'y':  6,
    'a':  7, 'i':  8, 'r':  9, 's': 10, 'h': 11, 'c': 12,
    't': 13, 'l': 14, 'n': 15, 'm': 16, 'p': 17, 'g': 18,
    'f': 19, 'x': 20, 'b': 21, 'v': 22, 'z': 23
}

# Pre‑compute modular inverses
_inv_cache = {n: pow(n, -1, MOD) for n in range(1, MOD + 1)}
num_to_latin = {i: LATIN_23[i - 1] for i in range(1, MOD + 1)}

# --------------------------------------------------------------------------
def decode_word(eva_word: str) -> str:
    """Return the mod‑23 inverse decoding of a single EVA token."""
    nums = [glyph_to_num[g] for g in eva_word if g in glyph_to_num]
    invs = [_inv_cache[n] for n in nums]
    return ''.join(num_to_latin[i] for i in invs)

def decode_list(words):
    """Vectorised convenience wrapper."""
    return [decode_word(w) for w in words]
