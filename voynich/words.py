"""
words.py — word-level (dictionary) scoring and crib locking.

Two ideas live here:

1. The ANAGRAM hypothesis: if scribes wrote each word's letters in some
   canonical or scrambled order, character n-grams are meaningless — but
   a word's letter MULTISET survives.  So decoded words are scored as
   bags of letters against a dictionary built from the reference corpus:
   a word whose sorted letters match a real word scores that word's
   log-frequency; anything else pays a unigram-based mismatch penalty.

2. CRIB LOCKING: after a solve, tokens whose assignments are supported
   by many dictionary-matched words get frozen, and the search re-runs
   on the remaining tokens.  This is the automated version of how human
   cryptanalysts bootstrap from partial cribs — with thresholds, because
   locking junk early poisons everything downstream.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from .lm import A, SPACE_ID, CharNgramModel

N_LETTERS = 26
MISMATCH_PENALTY = -3.0  # extra bits charged for a non-dictionary word


def alphagram_text(text: str) -> str:
    """Reference text with each word's letters sorted — an order-invariant
    representation.  An n-gram LM trained on this gives the anagram
    search a smooth gradient (Hauer & Kondrak's trick), where pure
    dictionary matching is nearly binary and traps the annealer."""
    return " ".join("".join(sorted(w)) for w in text.split())


class WordDictionary:
    """Word frequencies, anagram signatures and letter unigrams from a
    cleaned reference corpus."""

    def __init__(self, text: str, max_words: int = 60_000):
        counts = collections.Counter(text.split())
        total = sum(counts.values())
        self.word_logp: dict[str, float] = {}
        # signature (bytes of sorted letter ids) -> (logp, best word)
        self.sig_best: dict[bytes, tuple[float, str]] = {}
        letter_counts = np.ones(N_LETTERS)

        for word, c in counts.most_common(max_words):
            lp = math.log2(c / total)
            self.word_logp[word] = lp
            sig = bytes(sorted(ord(ch) - 97 for ch in word))
            cur = self.sig_best.get(sig)
            if cur is None or lp > cur[0]:
                self.sig_best[sig] = (lp, word)

        for word, c in counts.items():
            for ch in word:
                letter_counts[ord(ch) - 97] += c
        self.letter_logp = np.log2(letter_counts / letter_counts.sum())

    def expected_word_score(self, text: str) -> float:
        """Mean log2 P(word) per word token of the corpus itself — the
        ceiling for word-level objectives."""
        words = text.split()
        total = 0.0
        n = 0
        for w in words:
            lp = self.word_logp.get(w)
            if lp is not None:
                total += lp
                n += 1
        return total / max(n, 1)


class WordTypes:
    """The corpus reduced to unique word types (token-id tuples) with
    frequencies, plus a token -> types posting index for incremental
    rescoring."""

    def __init__(self, tokenized_lines: list[list[list[int]]]):
        counts: collections.Counter = collections.Counter()
        for line in tokenized_lines:
            for word in line:
                if word:
                    counts[tuple(word)] += 1
        self.types = [np.array(t, dtype=np.int64) for t in counts]
        self.freqs = np.array(list(counts.values()), dtype=np.float64)
        self.n_word_tokens = float(self.freqs.sum())

        postings: dict[int, list[int]] = collections.defaultdict(list)
        for i, t in enumerate(self.types):
            for tok in set(int(x) for x in t):
                postings[tok].append(i)
        self.postings = {
            tok: np.array(ix, dtype=np.int64) for tok, ix in postings.items()
        }


class AnagramScorer:
    """Objective for anagram-hypothesis keys: mean log2 P(word) per word
    token, where a word's identity is its letter multiset.

    A word whose sorted letters match a dictionary word scores that
    word's log-frequency.  Unmatched words fall back to an n-gram LM
    trained on the reference's ALPHAGRAMS (letters sorted within words)
    when one is supplied — a smooth, order-invariant gradient — or to a
    penalized letter-unigram model otherwise.  Supports incremental
    updates so the annealer only rescores word types that contain a
    mutated token."""

    def __init__(
        self,
        word_types: WordTypes,
        dictionary: WordDictionary,
        alphagram_lm: CharNgramModel | None = None,
    ):
        self.wt = word_types
        self.dict = dictionary
        self._scores: np.ndarray | None = None
        self._weighted = 0.0
        self._alpha_logp = alphagram_lm.logp if alphagram_lm is not None else None
        self._alpha_order = alphagram_lm.order if alphagram_lm is not None else 0
        if alphagram_lm is not None:
            mod = A ** (alphagram_lm.order - 1)
            ctx = 0
            for _ in range(alphagram_lm.order - 1):
                ctx = ctx * A + SPACE_ID
            self._alpha_mod = mod
            self._alpha_space_ctx = ctx

    # ---- scoring ---------------------------------------------------------

    def _alphagram_lm_score(self, sorted_letters) -> float:
        """log2 probability of a sorted word (plus its closing space)
        under the alphagram LM, with space context at the word start."""
        logp = self._alpha_logp
        mod = self._alpha_mod
        ctx = self._alpha_space_ctx
        total = 0.0
        for c in sorted_letters:
            total += logp[ctx * A + c]
            ctx = (ctx * A + c) % mod
        total += logp[ctx * A + SPACE_ID]
        return float(total)

    def _type_score(self, key: np.ndarray, ti: int) -> float:
        letters = key[self.wt.types[ti]]
        sorted_letters = sorted(int(x) for x in letters)
        hit = self.dict.sig_best.get(bytes(sorted_letters))
        if hit is not None:
            return hit[0]
        if self._alpha_logp is not None:
            return self._alphagram_lm_score(sorted_letters)
        return float(self.dict.letter_logp[letters].sum()) + MISMATCH_PENALTY

    def reset(self, key: np.ndarray) -> float:
        self._scores = np.array(
            [self._type_score(key, i) for i in range(len(self.wt.types))]
        )
        self._weighted = float(self._scores @ self.wt.freqs)
        return self.objective()

    def objective(self) -> float:
        return self._weighted / self.wt.n_word_tokens

    def score_key(self, key: np.ndarray) -> float:
        """Stateless full evaluation (anchors, held-out sets)."""
        total = 0.0
        for i in range(len(self.wt.types)):
            total += self._type_score(key, i) * self.wt.freqs[i]
        return total / self.wt.n_word_tokens

    # ---- incremental updates for the annealer ----------------------------

    def update(self, key: np.ndarray, changed_tokens: list[int]):
        """Rescore types containing any changed token.  Returns an undo
        state; objective() reflects the new key afterwards."""
        parts = [
            self.wt.postings[t] for t in changed_tokens if t in self.wt.postings
        ]
        if not parts:
            return None
        affected = np.unique(np.concatenate(parts))
        old = self._scores[affected].copy()
        for ti in affected:
            self._scores[ti] = self._type_score(key, int(ti))
        delta = float((self._scores[affected] - old) @ self.wt.freqs[affected])
        self._weighted += delta
        return (affected, old, delta)

    def revert(self, undo) -> None:
        if undo is None:
            return
        affected, old, delta = undo
        self._scores[affected] = old
        self._weighted -= delta

    # ---- reporting --------------------------------------------------------

    def match_info(self, key: np.ndarray, min_len: int = 1):
        """(matched type mask, matched words list aligned with types)."""
        matched = np.zeros(len(self.wt.types), dtype=bool)
        words: list[str | None] = [None] * len(self.wt.types)
        for i, t in enumerate(self.wt.types):
            if len(t) < min_len:
                continue
            letters = key[t]
            hit = self.dict.sig_best.get(bytes(sorted(int(x) for x in letters)))
            if hit is not None:
                matched[i] = True
                words[i] = hit[1]
        return matched, words


def frequency_init_key(
    word_types: WordTypes,
    dictionary: WordDictionary,
    n_tokens: int | None = None,
) -> np.ndarray:
    """Classic frequency-analysis seed: align tokens to letters by rank.
    Letter frequencies survive anagramming exactly, so this start is far
    inside the truth's basin compared to a random key."""
    if n_tokens is None:
        n_tokens = 1 + max(
            (int(t.max()) for t in word_types.types if len(t)), default=0
        )
    tok_freq = np.zeros(n_tokens)
    for t, f in zip(word_types.types, word_types.freqs):
        for tok in t:
            tok_freq[tok] += f
    letter_rank = np.argsort(-dictionary.letter_logp)
    key = np.empty(n_tokens, dtype=np.int64)
    for rank, tok in enumerate(np.argsort(-tok_freq)):
        key[tok] = letter_rank[rank % N_LETTERS]
    return key


def word_match_rate(
    decoded_words: list[tuple[str, int]],
    dictionary: WordDictionary,
    min_len: int = 3,
) -> float:
    """Frequency-weighted share of decoded word tokens (length >= min_len)
    that are real reference-corpus words.  The 'is it language yet?'
    diagnostic, applicable to every hypothesis."""
    hit = 0
    total = 0
    for word, freq in decoded_words:
        if len(word) < min_len:
            continue
        total += freq
        if word in dictionary.word_logp:
            hit += freq
    return hit / total if total else 0.0


def evidence_locks(
    word_types: WordTypes,
    matched: np.ndarray,
    n_tokens: int,
    min_len: int = 3,
    min_support: float = 20.0,
    min_ratio: float = 0.6,
) -> np.ndarray:
    """Boolean mask of tokens whose assignments are corroborated by
    dictionary matches: at least `min_support` occurrences inside words
    of length >= min_len, of which >= `min_ratio` are matched."""
    matched_freq = np.zeros(n_tokens)
    total_freq = np.zeros(n_tokens)
    for i, t in enumerate(word_types.types):
        if len(t) < min_len:
            continue
        f = word_types.freqs[i]
        for tok in set(int(x) for x in t):
            total_freq[tok] += f
            if matched[i]:
                matched_freq[tok] += f
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total_freq > 0, matched_freq / total_freq, 0.0)
    return (total_freq >= min_support) & (ratio >= min_ratio)
