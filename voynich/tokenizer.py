"""
tokenizer.py — EVA glyph tokenization.

EVA transcribes some single Voynich glyphs as multi-character sequences
(benched gallows 'cth'/'ckh'/'cph'/'cfh', the bench characters 'ch'/'sh').
Those are always treated as atomic.  On top of that, a simple BPE pass
can learn frequent composites ('qo', 'ai', 'dy', 'aiin', ...) so the
solver can hypothesise that they encode single plaintext letters.
"""

from __future__ import annotations

import collections

# Multi-character EVA sequences that represent single glyphs.
CORE_MULTIGRAPHS = ["cth", "ckh", "cph", "cfh", "ch", "sh"]


def base_tokenize(word: str) -> list[str]:
    """Split an EVA word into core glyphs (multigraphs + single chars)."""
    tokens = []
    i = 0
    n = len(word)
    while i < n:
        for mg in CORE_MULTIGRAPHS:
            if word.startswith(mg, i):
                tokens.append(mg)
                i += len(mg)
                break
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def learn_bpe_merges(words: list[str], num_merges: int = 30) -> list[tuple[str, str]]:
    """Learn BPE merges over core-tokenized words.  Returns the ordered
    list of merged pairs."""
    vocab = collections.Counter()
    for w in words:
        vocab[tuple(base_tokenize(w))] += 1

    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        merged = best[0] + best[1]
        new_vocab = collections.Counter()
        for symbols, freq in vocab.items():
            out = []
            i = 0
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == best[0]
                    and symbols[i + 1] == best[1]
                ):
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab
    return merges


class EvaTokenizer:
    """Tokenizes EVA words into glyph tokens using core multigraphs plus
    learned BPE merges, and maintains an integer vocabulary."""

    def __init__(self, merges: list[tuple[str, str]] | None = None):
        self.merges = merges or []
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def train(cls, words: list[str], num_merges: int = 30) -> "EvaTokenizer":
        return cls(learn_bpe_merges(words, num_merges))

    def tokenize(self, word: str) -> list[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = base_tokenize(word)
        for a, b in self.merges:
            merged = a + b
            out = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        self._cache[word] = symbols
        return symbols

    def build_vocab(self, lines: list[list[str]]) -> list[str]:
        """Token vocabulary (sorted by frequency, most common first) over
        all words in the given lines."""
        counts = collections.Counter()
        for line in lines:
            for word in line:
                counts.update(self.tokenize(word))
        return [tok for tok, _ in counts.most_common()]
