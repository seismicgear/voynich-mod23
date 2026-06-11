import numpy as np

from voynich.cipher import (
    NgramView,
    decode_lines,
    decode_stream,
    encode_corpus,
    random_key,
)
from voynich.lm import ALPHABET, SPACE_ID, CharNgramModel


def _toy_corpus(n_states=1):
    lines = [
        [["d", "a", "i"], ["q", "o"]],
        [["ch", "e", "d"], ["d", "a"], ["o"]],
    ]
    vocab = ["d", "a", "i", "q", "o", "ch", "e"]
    return encode_corpus(lines, vocab, n_states=n_states), lines, vocab


def test_streams_have_spaces_between_words_and_lines():
    corpus, lines, vocab = _toy_corpus()
    n_words = sum(len(l) for l in lines)
    n_tokens = sum(len(w) for l in lines for w in l)
    spaces = (corpus.token_stream == corpus.space_token).sum()
    # word separators within lines + 1 line separator
    assert spaces == (n_words - len(lines)) + (len(lines) - 1)
    assert len(corpus.token_stream) == n_tokens + spaces


def test_positional_states_mark_first_and_last_words():
    corpus, _, _ = _toy_corpus(n_states=3)
    non_space = corpus.token_stream != corpus.space_token
    states = corpus.state_stream[non_space]
    # line 1: 'dai' (start) 'qo' (end); line 2: 'ched' (start) 'da' (body) 'o' (end)
    assert list(states) == [0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 2]


def test_ngram_view_score_matches_direct_scoring(english_text):
    corpus, _, _ = _toy_corpus(n_states=3)
    lm = CharNgramModel(order=3).fit(english_text)
    rng = np.random.default_rng(7)
    for _ in range(5):
        key = random_key(3, corpus.n_tokens, rng)
        direct = lm.score_ids(decode_stream(key, corpus).astype(np.int64))
        via_view = NgramView(corpus, 3).score(key, lm.logp)
        assert abs(direct - via_view) < 1e-12


def test_decode_lines_matches_stream():
    corpus, lines, vocab = _toy_corpus()
    rng = np.random.default_rng(3)
    key = random_key(1, corpus.n_tokens, rng)
    rendered = decode_lines(key, lines, vocab, n_states=1)
    stream_text = "".join(ALPHABET[i] for i in decode_stream(key, corpus))
    assert " ".join(rendered) == stream_text


def test_random_key_pins_space():
    corpus, _, _ = _toy_corpus()
    key = random_key(1, corpus.n_tokens, np.random.default_rng(0))
    assert key[0, corpus.space_token] == SPACE_ID
    assert (key[0, : corpus.space_token] != SPACE_ID).all()
