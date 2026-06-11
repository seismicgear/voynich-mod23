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


# ---- expansion (abbreviation-hypothesis) keys -----------------------------
#
# Medieval Latin manuscripts were written with heavy scribal abbreviation:
# single signs stood for letter groups ('9' for 'con-/-us', the macron for a
# following nasal, etc.).  If Voynichese tokens are such signs, a token must
# be allowed to decode to MORE than one letter.  An expansion key maps each
# token to one or two letters; this is also the leading family of
# explanations for Voynichese's anomalously low character entropy.

NO_CHAR = -1


def random_expansion_key(
    n_tokens: int, rng: np.random.Generator, p_second: float = 0.0
) -> np.ndarray:
    """Random expansion key of shape (n_tokens, 2).  Column 0 is a letter;
    column 1 is a letter with probability p_second, else NO_CHAR.  The
    space token (last index) is pinned to a bare space."""
    key = np.empty((n_tokens, 2), dtype=np.int64)
    key[:, 0] = rng.integers(0, A - 1, size=n_tokens)
    key[:, 1] = np.where(
        rng.random(n_tokens) < p_second,
        rng.integers(0, A - 1, size=n_tokens),
        NO_CHAR,
    )
    key[n_tokens - 1] = (SPACE_ID, NO_CHAR)
    return key


def expand_stream(key: np.ndarray, token_stream: np.ndarray) -> np.ndarray:
    """Decode a token stream under an expansion key: each position emits
    0, 1 or 2 chars (0 = the token is a NULL, decoding to nothing —
    the Tranchedino-style null-cipher hypothesis)."""
    toks = token_stream.astype(np.int64)
    first = key[toks, 0]
    second = key[toks, 1]
    v0 = first != NO_CHAR
    v1 = second != NO_CHAR
    lengths = v0.astype(np.int64) + v1
    starts = np.cumsum(lengths) - lengths
    out = np.empty(int(lengths.sum()), dtype=np.int64)
    out[starts[v0]] = first[v0]
    out[(starts + v0)[v1]] = second[v1]
    return out


class ExpansionScorer:
    """Scores expansion keys by decoding the full stream and scoring it
    under the language model.  Variable-length output rules out the
    NgramView compression, so this is O(corpus) per call — still a few
    milliseconds thanks to vectorization.

    The search objective is total log-probability divided by the (fixed)
    number of tokens.  Normalizing per OUTPUT char instead would reward
    degenerate keys that expand every token into high-probability filler
    ('rerere...'): padding raises the per-char average for free, whereas
    per-token it costs exactly the bits the extra letters consume."""

    def __init__(
        self,
        corpus: EncodedCorpus,
        lm,
        min_output_frac: float = 0.0,
        null_penalty: float = 3.0,
    ):
        self.token_stream = corpus.token_stream.astype(np.int64)
        self.n_positions = len(self.token_stream)
        # With nulls allowed, deleting text shrinks the (negative) total
        # and inflates the per-token objective.  Nulls therefore pay
        # rent: `null_penalty` bits per nulled occurrence (the cost of
        # signalling "skip this sign"), plus a hard output-length floor.
        # Genuine noise glyphs save far more than the rent; real text
        # does not.
        self.min_chars = int(min_output_frac * self.n_positions)
        self.null_penalty = null_penalty
        self.lm = lm

    def score(self, key: np.ndarray) -> float:
        """Search objective: bits per token (higher is better)."""
        out = expand_stream(key, self.token_stream)
        if len(out) < self.min_chars:
            return -1e9
        total = float(self.lm.logp[self.lm.ngram_indices(out)].sum())
        n_null = int((key[self.token_stream, 0] == NO_CHAR).sum())
        total -= self.null_penalty * n_null
        return total / self.n_positions

    def per_char(self, key: np.ndarray) -> float:
        """Reporting metric: mean bits per decoded character, comparable
        with the plain-substitution scores and corpus anchors."""
        return self.lm.score_ids(expand_stream(key, self.token_stream))


def expansion_strings(key: np.ndarray, vocab: list[str]) -> dict[str, str]:
    """token -> decoded letter string (0-2 letters; '' = null token)."""
    out = {}
    for tok_id, tok in enumerate(vocab):
        s = ""
        if key[tok_id, 0] != NO_CHAR:
            s += ALPHABET[key[tok_id, 0]]
        if key[tok_id, 1] != NO_CHAR:
            s += ALPHABET[key[tok_id, 1]]
        out[tok] = s
    return out


def decode_lines_expanded(
    key: np.ndarray,
    lines: list[list[list[str]]],
    vocab: list[str],
    max_lines: int | None = None,
) -> list[str]:
    """Human-readable decoding of tokenized lines under an expansion key."""
    strings = expansion_strings(key, vocab)
    out = []
    for line in lines[:max_lines]:
        words = []
        for word in line:
            words.append("".join(strings.get(tok, "") for tok in word))
        out.append(" ".join(words))
    return out


def expansion_key_table(key: np.ndarray, vocab: list[str]) -> list[dict]:
    strings = expansion_strings(key, vocab)
    return [
        {"token": tok, "all": strings[tok] or "∅"}  # ∅ marks a null token
        for tok in vocab
    ]


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
