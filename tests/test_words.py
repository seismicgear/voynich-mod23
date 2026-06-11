import numpy as np
import pytest

from voynich.lm import CharNgramModel
from voynich.words import (
    AnagramScorer,
    WordDictionary,
    WordTypes,
    alphagram_text,
    evidence_locks,
    frequency_init_key,
    word_match_rate,
)


def _ids(word):
    return [ord(c) - 97 for c in word]


@pytest.fixture(scope="module")
def small_dict():
    return WordDictionary("the cat sat on the mat the cat ran far away")


def test_alphagram_text():
    assert alphagram_text("the cat") == "eht act"


def test_dictionary_signatures(small_dict):
    # 'the' and its anagram share a signature; most frequent word wins
    sig = bytes(sorted(_ids("the")))
    assert small_dict.sig_best[sig][1] == "the"
    assert small_dict.word_logp["the"] > small_dict.word_logp["far"]


def test_word_match_rate(small_dict):
    decoded = [("the", 5), ("cat", 2), ("zzz", 3), ("on", 9)]
    # 'on' is below min_len; 7 of 10 length>=3 tokens match
    assert word_match_rate(decoded, small_dict) == pytest.approx(0.7)


def _toy_problem(english_text):
    # Cipher: identity letters as tokens over a tiny corpus of words that
    # certainly occur in the KJV fixture
    lines = [[_ids("the"), _ids("and")], [_ids("god"), _ids("earth"), _ids("the")]]
    wt = WordTypes(lines)
    d = WordDictionary(english_text)
    lm = CharNgramModel(order=3).fit(alphagram_text(english_text))
    return wt, d, AnagramScorer(wt, d, alphagram_lm=lm)


def test_anagram_scorer_incremental_matches_stateless(english_text):
    wt, d, scorer = _toy_problem(english_text)
    rng = np.random.default_rng(0)
    key = rng.integers(0, 26, size=26, dtype=np.int64)
    scorer.reset(key)
    for _ in range(20):
        tok = int(rng.integers(0, 26))
        key[tok] = int(rng.integers(0, 26))
        scorer.update(key, [tok])
        assert scorer.objective() == pytest.approx(scorer.score_key(key))


def test_anagram_scorer_revert(english_text):
    wt, d, scorer = _toy_problem(english_text)
    key = np.arange(26, dtype=np.int64) % 26
    before = scorer.reset(key)
    old = int(key[7])
    key[7] = (old + 3) % 26
    undo = scorer.update(key, [7])
    scorer.revert(undo)
    key[7] = old
    assert scorer.objective() == pytest.approx(before)


def test_identity_key_matches_words(english_text):
    wt, d, scorer = _toy_problem(english_text)
    identity = np.arange(26, dtype=np.int64)
    matched, words = scorer.match_info(identity)
    assert matched.all()
    assert set(words) <= set(d.word_logp)


def test_frequency_init_key_aligns_ranks():
    wt = WordTypes([[[0, 1, 0], [0, 2]]])  # token 0 most frequent
    d = WordDictionary("aaa ab b e ee eee eeee")  # 'e' most frequent letter
    key = frequency_init_key(wt, d, n_tokens=3)
    assert key[0] == ord("e") - 97


def test_nomenclator_scorer_incremental_and_injective(english_text):
    from voynich.words import NomenclatorScorer

    lines = [[_ids("the"), _ids("god")], [_ids("earth"), _ids("the")]]
    wt = WordTypes(lines)
    d = WordDictionary(english_text)
    lm = CharNgramModel(order=3).fit(english_text)
    scorer = NomenclatorScorer(wt, d, lm, n_code_slots=3, n_code_words=50)

    key = np.arange(26, dtype=np.int64)
    base = scorer.reset(key)

    # Assign a code; objective must match the stateless evaluation.
    ti = int(scorer.eligible[0])
    undo = scorer.update_code(ti, 5)
    assert undo is not None
    stateless = scorer.score_key(key, scorer.codebook())
    assert scorer.objective() == pytest.approx(stateless)

    # Injectivity: the same code word cannot serve a second type.
    other = int(scorer.eligible[1])
    assert scorer.update_code(other, 5) is None

    # Reverting restores the baseline exactly.
    scorer.revert(undo)
    assert scorer.objective() == pytest.approx(base)

    # Token changes only affect spelled types.
    undo2 = scorer.update_code(ti, 7)
    before = scorer.objective()
    undo_tok = scorer.update_tokens(key, [int(wt.types[ti][0])])
    # key unchanged => spell scores unchanged => objective unchanged
    assert scorer.objective() == pytest.approx(before)
    scorer.revert(undo_tok)
    scorer.revert(undo2)


def test_nomenclator_code_cost_charged(english_text):
    from voynich.words import NomenclatorScorer

    lines = [[_ids("the")]] * 3
    wt = WordTypes(lines)
    d = WordDictionary(english_text)
    lm = CharNgramModel(order=3).fit(english_text)
    free = NomenclatorScorer(wt, d, lm, code_entry_cost_bits=0.0)
    costly = NomenclatorScorer(wt, d, lm, code_entry_cost_bits=30.0)
    key = np.arange(26, dtype=np.int64)
    free.reset(key)
    costly.reset(key)
    free.update_code(0, 0)
    costly.update_code(0, 0)
    assert free.objective() - costly.objective() == pytest.approx(
        30.0 / wt.n_word_tokens
    )


def test_evidence_locks_thresholds():
    # token 0 occurs in matched long words 30 times; token 1 unmatched
    lines = [[[0, 1, 2]] * 30 + [[1, 1, 2]] * 30]
    wt = WordTypes(lines)
    matched = np.array([True, False])
    locks = evidence_locks(wt, matched, n_tokens=3, min_support=20, min_ratio=0.6)
    assert bool(locks[0]) is True       # only in matched words
    assert bool(locks[1]) is False      # half its support is unmatched
