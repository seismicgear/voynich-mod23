"""
cipher.py — substitution-key hypotheses and vectorized decoding.

A corpus of tokenized lines is flattened into two parallel int arrays:

  token_stream  token id at each position; SPACE between words and lines
  state_stream  positional state of each token (for the positional
                hypothesis): 0 = token of the first word in a line,
                2 = token of the last word, 1 = body.  Spaces get 0.

A key is an int array of shape (n_states, n_tokens) mapping (state,
token) -> alphabet char id.  The simple (monoalphabetic) hypothesis uses
n_states == 1 and an all-zero state stream.  Decoding the whole corpus
is a single fancy-indexing gather, which is what makes MCMC over the key
space fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lm import A, ALPHABET, SPACE_ID

STATE_NAMES = {1: ["all"], 3: ["line-start", "body", "line-end"]}


@dataclass
class EncodedCorpus:
    vocab: list[str]              # token strings; SPACE is implicit, id = len(vocab)
    token_stream: np.ndarray      # int32, includes space token
    state_stream: np.ndarray      # int32
    n_states: int

    @property
    def space_token(self) -> int:
        return len(self.vocab)

    @property
    def n_tokens(self) -> int:
        return len(self.vocab) + 1


def encode_corpus(
    lines: list[list[list[str]]],
    vocab: list[str],
    n_states: int = 1,
) -> EncodedCorpus:
    """Flatten tokenized lines (line -> word -> token strings) into
    streams.  Tokens not in the vocab are skipped (should not happen when
    the vocab was built from the same lines)."""
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    space = len(vocab)

    tokens: list[int] = []
    states: list[int] = []

    def emit(tok_id: int, state: int) -> None:
        tokens.append(tok_id)
        states.append(state)

    for line in lines:
        if not line:
            continue
        if tokens:
            emit(space, 0)  # line separator
        last_word = len(line) - 1
        for w_idx, word in enumerate(line):
            if w_idx > 0:
                emit(space, 0)
            if n_states == 3:
                state = 0 if w_idx == 0 else (2 if w_idx == last_word else 1)
            else:
                state = 0
            for tok in word:
                tok_id = tok_to_id.get(tok)
                if tok_id is not None:
                    emit(tok_id, state)

    return EncodedCorpus(
        vocab=vocab,
        token_stream=np.array(tokens, dtype=np.int32),
        state_stream=np.array(states, dtype=np.int32),
        n_states=n_states,
    )


def random_key(n_states: int, n_tokens: int, rng: np.random.Generator) -> np.ndarray:
    """Random key: every (state, token) gets a random letter (not space);
    the space token is pinned to the space character in every state."""
    key = rng.integers(0, A - 1, size=(n_states, n_tokens), dtype=np.int64)
    key[:, n_tokens - 1] = SPACE_ID
    return key


def decode_stream(key: np.ndarray, corpus: EncodedCorpus) -> np.ndarray:
    """Char ids for the whole corpus under the key."""
    return key[corpus.state_stream, corpus.token_stream]


def decode_lines(
    key: np.ndarray,
    lines: list[list[list[str]]],
    vocab: list[str],
    n_states: int,
    max_lines: int | None = None,
) -> list[str]:
    """Human-readable decoding of (tokenized) lines under a key."""
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    out = []
    for line in lines[:max_lines]:
        last_word = len(line) - 1
        words = []
        for w_idx, word in enumerate(line):
            if n_states == 3:
                state = 0 if w_idx == 0 else (2 if w_idx == last_word else 1)
            else:
                state = 0
            chars = []
            for tok in word:
                tok_id = tok_to_id.get(tok)
                if tok_id is not None:
                    chars.append(ALPHABET[key[state, tok_id]])
            words.append("".join(chars))
        out.append(" ".join(words))
    return out


def key_table(key: np.ndarray, vocab: list[str]) -> list[dict]:
    """Key as a list of rows for display: token plus its letter per state."""
    names = STATE_NAMES.get(key.shape[0], [str(i) for i in range(key.shape[0])])
    rows = []
    for tok_id, tok in enumerate(vocab):
        row = {"token": tok}
        for s, name in enumerate(names):
            row[name] = ALPHABET[key[s, tok_id]]
        rows.append(row)
    return rows


class NgramView:
    """Compressed n-gram view of a (token_stream, state_stream) pair.

    Every window of `order` consecutive positions is reduced to a unique
    pattern with a count, so re-scoring a key costs O(unique patterns)
    instead of O(corpus length).  Scores are exact."""

    def __init__(self, corpus: EncodedCorpus, order: int):
        self.order = order
        n_tokens = corpus.n_tokens
        # Pad the front with spaces so the first characters are scored
        # with space context, matching CharNgramModel.ngram_indices.
        pad_t = np.full(order - 1, corpus.space_token, dtype=np.int64)
        pad_s = np.zeros(order - 1, dtype=np.int64)
        toks = np.concatenate([pad_t, corpus.token_stream.astype(np.int64)])
        stas = np.concatenate([pad_s, corpus.state_stream.astype(np.int64)])

        codes = stas * n_tokens + toks
        k = corpus.n_states * n_tokens
        n = len(codes) - order + 1
        packed = np.zeros(n, dtype=np.int64)
        for j in range(order):
            packed = packed * k + codes[j : j + n]
        uniq, counts = np.unique(packed, return_counts=True)

        self.tokens = np.empty((len(uniq), order), dtype=np.int64)
        self.states = np.empty((len(uniq), order), dtype=np.int64)
        rem = uniq.copy()
        for j in reversed(range(order)):
            c = rem % k
            rem //= k
            self.states[:, j] = c // n_tokens
            self.tokens[:, j] = c % n_tokens
        self.counts = counts.astype(np.float64)
        self.total = float(counts.sum())

    def score(self, key: np.ndarray, logp: np.ndarray) -> float:
        """Mean log2 probability per character under the key and the flat
        n-gram log-probability table."""
        chars = key[self.states, self.tokens]  # (P, order)
        idx = chars[:, 0]
        for j in range(1, self.order):
            idx = idx * A + chars[:, j]
        return float(self.counts @ logp[idx]) / self.total
