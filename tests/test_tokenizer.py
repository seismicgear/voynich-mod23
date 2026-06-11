from voynich.tokenizer import EvaTokenizer, base_tokenize, learn_bpe_merges


def test_core_multigraphs_are_atomic():
    assert base_tokenize("chedy") == ["ch", "e", "d", "y"]
    assert base_tokenize("qokcthy") == ["q", "o", "k", "cth", "y"]
    assert base_tokenize("shol") == ["sh", "o", "l"]


def test_bpe_learns_frequent_pairs():
    words = ["qokedy"] * 50 + ["qokeedy"] * 30 + ["daiin"] * 40
    merges = learn_bpe_merges(words, num_merges=2)
    assert ("q", "o") in merges  # 'qo' appears in 80 words


def test_tokenizer_roundtrip_covers_word():
    tok = EvaTokenizer.train(["daiin", "qokeedy", "chedy"] * 10, num_merges=5)
    for word in ["daiin", "qokeedy", "chedy", "xyzzy"]:
        assert "".join(tok.tokenize(word)) == word


def test_vocab_sorted_by_frequency():
    tok = EvaTokenizer(merges=[])
    vocab = tok.build_vocab([["ddd", "dd"], ["da"]])
    assert vocab[0] == "d"
