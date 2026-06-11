"""
synthetic.py — solver validation on ciphers with known answers.

"Can this thing solve anything?" is answerable: encrypt a held-out slice
of a real-language corpus with a random substitution into an artificial
glyph alphabet, hand the solver only the ciphertext and a language model
trained on *different* text, and measure how much of the plaintext it
recovers.  This is the honest benchmark behind any claim the machinery
makes about the Voynich text itself.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import collections

from .annealer import (
    anneal,
    anneal_anagram,
    anneal_expansion,
    random_expansion_scores,
    random_key_scores,
)
from .words import (
    AnagramScorer,
    WordDictionary,
    WordTypes,
    alphagram_text,
    frequency_init_key,
)
from .cipher import (
    NO_CHAR,
    EncodedCorpus,
    ExpansionScorer,
    NgramView,
    decode_stream,
    encode_corpus,
    expand_stream,
    expansion_strings,
)
from .lm import ALPHABET, SPACE_ID, CharNgramModel, encode_text


def make_synthetic_cipher(
    plaintext: str, seed: int | None = None
) -> tuple[EncodedCorpus, np.ndarray, np.ndarray]:
    """Encrypt plaintext (a-z + spaces) with a random monoalphabetic
    substitution into fake glyph tokens.

    Returns (encoded_corpus, true_key, plaintext_char_ids) where
    plaintext_char_ids aligns 1:1 with the corpus token stream."""
    rng = np.random.default_rng(seed)
    ids = encode_text(plaintext)
    letters = sorted({int(i) for i in ids if i != SPACE_ID})

    # One fake glyph per plaintext letter, in shuffled order so glyph
    # index carries no information.
    perm = rng.permutation(len(letters))
    vocab = [f"g{j:02d}" for j in range(len(letters))]
    letter_to_token = {letters[i]: int(perm[i]) for i in range(len(letters))}

    # Build "lines" of ~8 words to mirror the Voynich line structure.
    words = plaintext.split()
    lines: list[list[list[str]]] = []
    for i in range(0, len(words), 8):
        chunk = words[i : i + 8]
        line = []
        for w in chunk:
            toks = [
                vocab[letter_to_token[c]]
                for c in (int(x) for x in encode_text(w))
            ]
            if toks:
                line.append(toks)
        if line:
            lines.append(line)

    corpus = encode_corpus(lines, vocab, n_states=1)

    true_key = np.full((1, len(vocab) + 1), SPACE_ID, dtype=np.int64)
    for letter, tok in letter_to_token.items():
        true_key[0, tok] = letter

    plain_ids = decode_stream(true_key, corpus)
    return corpus, true_key, plain_ids


def make_abbreviation_cipher(
    plaintext: str, n_bigram_units: int = 18, seed: int | None = None
) -> tuple[EncodedCorpus, np.ndarray, np.ndarray]:
    """Encrypt plaintext the way an abbreviating scribe would: the most
    frequent letter bigrams become single signs (glyph tokens), remaining
    letters get one sign each, and signs are shuffled into an artificial
    glyph alphabet.

    Returns (encoded_corpus, true_expansion_key, plaintext_char_ids)."""
    rng = np.random.default_rng(seed)
    words = [w for w in plaintext.split() if w]

    bigram_counts = collections.Counter()
    for w in words:
        for i in range(len(w) - 1):
            bigram_counts[w[i : i + 2]] += 1
    bigram_units = [bg for bg, _ in bigram_counts.most_common(n_bigram_units)]
    bigram_set = set(bigram_units)

    def units_of(word: str) -> list[str]:
        out = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i : i + 2] in bigram_set:
                out.append(word[i : i + 2])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    # Inventory: every unit that actually occurs, shuffled into glyphs.
    unit_lines = [[units_of(w) for w in words[i : i + 8]] for i in range(0, len(words), 8)]
    units = sorted({u for line in unit_lines for w in line for u in w})
    perm = rng.permutation(len(units))
    vocab = [f"g{j:02d}" for j in range(len(units))]
    unit_to_token = {units[i]: int(perm[i]) for i in range(len(units))}

    lines = [
        [[vocab[unit_to_token[u]] for u in w] for w in line if w]
        for line in unit_lines
        if line
    ]
    corpus = encode_corpus(lines, vocab, n_states=1)

    char_id = {c: i for i, c in enumerate(ALPHABET)}
    true_key = np.full((len(vocab) + 1, 2), NO_CHAR, dtype=np.int64)
    true_key[len(vocab)] = (SPACE_ID, NO_CHAR)
    for unit, tok in unit_to_token.items():
        true_key[tok, 0] = char_id[unit[0]]
        if len(unit) == 2:
            true_key[tok, 1] = char_id[unit[1]]

    plain_ids = expand_stream(true_key, corpus.token_stream)
    return corpus, true_key, plain_ids


def staged_expansion_init(
    corpus: EncodedCorpus,
    lm: CharNgramModel,
    order: int,
    iterations: int = 15_000,
    seed: int | None = None,
) -> np.ndarray:
    """Seed key for expansion searches: solve the plain-substitution
    problem first (fast, via NgramView) and lift its best key into
    expansion form (every token one letter, no seconds)."""
    view = NgramView(corpus, order)
    plain = anneal(
        view,
        lm.logp,
        n_states=1,
        n_tokens=corpus.n_tokens,
        iterations=iterations,
        restarts=2,
        seed=seed,
    )
    init = np.full((corpus.n_tokens, 2), NO_CHAR, dtype=np.int64)
    init[:, 0] = plain.best_key[0]
    init[corpus.space_token] = (SPACE_ID, NO_CHAR)
    return init


def make_anagram_cipher(
    plaintext: str, seed: int | None = None
) -> tuple[list[list[list[int]]], list[str], np.ndarray, list[tuple[tuple, str]]]:
    """Encrypt plaintext by (a) shuffling the letters inside every word
    and (b) substituting each letter with a glyph.  Word boundaries
    survive; letter order does not — the anagram hypothesis exactly.

    Returns (tokenized_lines, vocab, true_key, occurrences) where
    occurrences pairs each word occurrence's ciphertext type with the
    original plaintext word."""
    rng = np.random.default_rng(seed)
    words = [w for w in plaintext.split() if w]

    letters = sorted({ch for w in words for ch in w})
    perm = rng.permutation(len(letters))
    vocab = [f"g{j:02d}" for j in range(len(letters))]
    letter_to_token = {letters[i]: int(perm[i]) for i in range(len(letters))}

    lines: list[list[list[int]]] = []
    occurrences: list[tuple[tuple, str]] = []
    for i in range(0, len(words), 8):
        line = []
        for w in words[i : i + 8]:
            shuffled = [w[j] for j in rng.permutation(len(w))]
            toks = [letter_to_token[ch] for ch in shuffled]
            line.append(toks)
            occurrences.append((tuple(toks), w))
        if line:
            lines.append(line)

    true_key = np.zeros(len(vocab), dtype=np.int64)
    for ch, tok in letter_to_token.items():
        true_key[tok] = ord(ch) - 97
    return lines, vocab, true_key, occurrences


def _run_anagram_benchmark(
    lm_text: str,
    sample: str,
    iterations: int,
    restarts: int,
    seed: int | None,
    progress,
    should_stop,
) -> dict:
    lines, vocab, true_key, occurrences = make_anagram_cipher(sample, seed=seed)
    dictionary = WordDictionary(lm_text)
    word_types = WordTypes(lines)
    alpha_lm = CharNgramModel(order=4).fit(alphagram_text(lm_text))
    scorer = AnagramScorer(word_types, dictionary, alphagram_lm=alpha_lm)

    result = anneal_anagram(
        scorer,
        n_tokens=len(vocab),
        iterations=iterations,
        restarts=restarts,
        seed=seed,
        init_key=frequency_init_key(word_types, dictionary),
        progress=progress,
        should_stop=should_stop,
    )

    # Accuracy: share of word occurrences whose matched dictionary word
    # IS the original plaintext word.
    matched, match_words = scorer.match_info(result.best_key)
    type_index = {tuple(int(x) for x in t): i for i, t in enumerate(word_types.types)}
    hits = 0
    for toks, original in occurrences:
        ti = type_index[toks]
        if matched[ti] and match_words[ti] == original:
            hits += 1
    accuracy = hits / max(len(occurrences), 1)

    rng = np.random.default_rng(seed)
    rand_scores = [
        scorer.score_key(rng.integers(0, 26, size=len(vocab), dtype=np.int64))
        for _ in range(20)
    ]
    preview_words = []
    truth_words = []
    for toks, original in occurrences[:50]:
        ti = type_index[toks]
        preview_words.append(match_words[ti] if matched[ti] else "?" * len(toks))
        truth_words.append(original)

    return {
        "mode": "anagram",
        "accuracy": accuracy,
        "best_score": result.best_score,
        "true_key_score": scorer.score_key(true_key),
        "random_key_score_mean": float(np.mean(rand_scores)),
        "iterations_done": result.iterations_done,
        "restarts_done": result.restarts_done,
        "elapsed_sec": result.elapsed_sec,
        "history": result.history,
        "decoded_preview": " ".join(preview_words)[:300],
        "plaintext_preview": " ".join(truth_words)[:300],
        "cipher_letters": sum(len(t) for t, _ in occurrences),
    }


def make_null_cipher(
    plaintext: str,
    null_frac: float = 0.12,
    n_null_glyphs: int = 4,
    seed: int | None = None,
) -> tuple[EncodedCorpus, np.ndarray, np.ndarray]:
    """Encrypt plaintext with a monoalphabetic substitution AND sprinkle
    in meaningless null glyphs (the standard camouflage of quattrocento
    diplomatic ciphers, e.g. Tranchedino's ledger).

    Returns (encoded_corpus, true_expansion_key, plaintext_char_ids)."""
    rng = np.random.default_rng(seed)
    words = [w for w in plaintext.split() if w]
    letters = sorted({ch for w in words for ch in w})

    n_letters = len(letters)
    n_glyphs = n_letters + n_null_glyphs
    perm = rng.permutation(n_glyphs)
    vocab = [f"g{j:02d}" for j in range(n_glyphs)]
    letter_to_token = {letters[i]: int(perm[i]) for i in range(n_letters)}
    null_tokens = [int(perm[n_letters + i]) for i in range(n_null_glyphs)]

    lines: list[list[list[str]]] = []
    for i in range(0, len(words), 8):
        line = []
        for w in words[i : i + 8]:
            toks = []
            for ch in w:
                toks.append(vocab[letter_to_token[ch]])
                if rng.random() < null_frac:
                    toks.append(vocab[null_tokens[int(rng.integers(0, n_null_glyphs))]])
            line.append(toks)
        if line:
            lines.append(line)

    corpus = encode_corpus(lines, vocab, n_states=1)

    char_id = {c: i for i, c in enumerate(ALPHABET)}
    true_key = np.full((len(vocab) + 1, 2), NO_CHAR, dtype=np.int64)
    true_key[len(vocab)] = (SPACE_ID, NO_CHAR)
    for ch, tok in letter_to_token.items():
        true_key[tok, 0] = char_id[ch]
    # null tokens keep (NO_CHAR, NO_CHAR): they decode to nothing

    plain_ids = expand_stream(true_key, corpus.token_stream)
    return corpus, true_key, plain_ids


def _split_for_benchmark(text: str, cipher_chars: int) -> tuple[str, str]:
    cut = max(len(text) - cipher_chars - 1, len(text) // 2)
    lm_text = text[:cut]
    sample = text[cut : cut + cipher_chars]
    # Align to word boundaries.
    sample = sample[sample.find(" ") + 1 : sample.rfind(" ")]
    return lm_text, sample


def run_benchmark(
    text: str,
    order: int = 4,
    cipher_chars: int = 4000,
    iterations: int = 20_000,
    restarts: int = 2,
    seed: int | None = 0,
    mode: str = "substitution",
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train an LM on the head of `text`, encrypt a slice from the tail,
    and measure recovery.

    mode 'substitution': one glyph per letter, solved with plain keys.
    mode 'abbreviation': frequent bigrams become single glyphs, solved
    with expansion keys (token -> 1-2 letters).
    mode 'nulls': substitution plus meaningless null glyphs, solved with
    expansion keys allowed to decode tokens to nothing.
    mode 'anagram': letters shuffled inside every word, solved with
    word-level dictionary scoring."""
    lm_text, sample = _split_for_benchmark(text, cipher_chars)
    if mode == "anagram":
        return _run_anagram_benchmark(
            lm_text, sample, iterations, restarts, seed, progress, should_stop
        )
    lm = CharNgramModel(order=order).fit(lm_text)

    if mode in ("abbreviation", "nulls"):
        if mode == "nulls":
            corpus, true_key, plain_ids = make_null_cipher(sample, seed=seed)
            scorer = ExpansionScorer(corpus, lm, min_output_frac=0.5)
        else:
            corpus, true_key, plain_ids = make_abbreviation_cipher(sample, seed=seed)
            scorer = ExpansionScorer(corpus, lm)
        init_key = staged_expansion_init(
            corpus, lm, order, iterations=min(iterations, 15_000), seed=seed
        )
        result = anneal_expansion(
            scorer,
            n_tokens=corpus.n_tokens,
            iterations=iterations,
            restarts=restarts,
            seed=seed,
            init_key=init_key,
            allow_nulls=(mode == "nulls"),
            progress=progress,
            should_stop=should_stop,
        )
        # Per-token expansion accuracy (frequency-weighted): robust even
        # when solver and truth disagree about a token's length.
        truth_strs = expansion_strings(true_key, corpus.vocab)
        found_strs = expansion_strings(result.best_key, corpus.vocab)
        tok_ids = corpus.token_stream[corpus.token_stream != corpus.space_token]
        matches = np.array(
            [truth_strs[t] == found_strs[t] for t in corpus.vocab], dtype=bool
        )
        accuracy = float(matches[tok_ids].mean())
        decoded = expand_stream(result.best_key, corpus.token_stream)
        true_score = scorer.score(true_key)
        rand_scores = random_expansion_scores(
            scorer, corpus.n_tokens, n_samples=20, seed=seed
        )
    else:
        corpus, true_key, plain_ids = make_synthetic_cipher(sample, seed=seed)
        view = NgramView(corpus, order)
        result = anneal(
            view,
            lm.logp,
            n_states=1,
            n_tokens=corpus.n_tokens,
            iterations=iterations,
            restarts=restarts,
            seed=seed,
            progress=progress,
            should_stop=should_stop,
        )
        decoded = decode_stream(result.best_key, corpus)
        letter_mask = plain_ids != SPACE_ID
        accuracy = float((decoded[letter_mask] == plain_ids[letter_mask]).mean())
        true_score = view.score(true_key, lm.logp)
        rand_scores = random_key_scores(
            view, lm.logp, 1, corpus.n_tokens, n_samples=20, seed=seed
        )

    preview = "".join(ALPHABET[i] for i in decoded[:300])
    truth = "".join(ALPHABET[i] for i in plain_ids[:300])

    return {
        "mode": mode,
        "accuracy": accuracy,
        "best_score": result.best_score,
        "true_key_score": true_score,
        "random_key_score_mean": float(np.mean(rand_scores)),
        "iterations_done": result.iterations_done,
        "restarts_done": result.restarts_done,
        "elapsed_sec": result.elapsed_sec,
        "history": result.history,
        "decoded_preview": preview,
        "plaintext_preview": truth,
        "cipher_letters": int((plain_ids != SPACE_ID).sum()),
    }
