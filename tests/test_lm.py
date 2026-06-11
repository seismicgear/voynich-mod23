import numpy as np
import pytest

from voynich.lm import A, SPACE_ID, CharNgramModel, encode_text


def test_encode_text_filters_and_collapses():
    ids = encode_text("Ab  c!\nd ")
    assert list(ids) == [0, 1, SPACE_ID, 2, SPACE_ID, 3]


def test_probabilities_normalize(english_text):
    lm = CharNgramModel(order=3).fit(english_text)
    table = np.exp2(lm.logp).reshape(-1, A)
    np.testing.assert_allclose(table.sum(axis=1), 1.0, rtol=1e-9)


@pytest.mark.parametrize("order", [3, 4])
def test_real_text_beats_scrambled(english_text, order):
    lm = CharNgramModel(order=order).fit(english_text[:60000])
    held_out = english_text[60000:]
    rng = np.random.default_rng(0)
    scrambled = "".join(rng.permutation(list(held_out)))
    assert lm.score_text(held_out) > lm.score_text(scrambled) + 1.0


def test_score_is_mean_log2_per_char(english_text):
    lm = CharNgramModel(order=3).fit(english_text)
    ids = encode_text("the quick brown fox")
    per_char = lm.logp[lm.ngram_indices(ids)]
    assert lm.score_ids(ids) == pytest.approx(per_char.mean())
    assert len(per_char) == len(ids)
