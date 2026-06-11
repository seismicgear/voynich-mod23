from voynich.diagnostics import (
    self_citation_rates,
    shuffled_rates,
    within_edit_one,
)


def test_within_edit_one():
    assert within_edit_one("daiin", "daiin")        # identical
    assert within_edit_one("daiin", "dain")         # one deletion
    assert within_edit_one("daiin", "qaiin")        # one substitution
    assert within_edit_one("daiin", "daiins")       # one insertion
    assert not within_edit_one("daiin", "qokedy")
    assert not within_edit_one("daiin", "dai")      # two deletions
    assert not within_edit_one("abcd", "abdc")      # transposition = 2 edits


def test_self_citation_detects_locality():
    # Copy-and-mutate stream: each word echoes its predecessor
    base = ["qokeedy", "qokedy", "qokeedy", "qokeed", "qokeedy"] * 40
    clustered = base
    rates = self_citation_rates(clustered, window=10)
    assert rates["near_rate"] > 0.9

    # Distinct long words, no echoes: the doubled counter guarantees
    # consecutive words differ in at least two positions
    distinct = [f"{i:03d}{i:03d}w" for i in range(200)]
    assert self_citation_rates(distinct, window=10)["near_rate"] < 0.2


def test_short_words_excluded():
    words = ["an", "an", "an", "an"]  # below MIN_WORD_LEN
    rates = self_citation_rates(words, window=5)
    assert rates["near_rate"] == 0.0
    assert rates["exact_rate"] == 0.0


def test_model_showdown_detects_copying():
    from voynich.diagnostics import model_showdown

    # Heavy copy-and-mutate stream: each word repeats/mutates recent ones
    rng_words = []
    stems = ["qokeedy", "chedaiin", "sholkar", "otarchy"]
    for i in range(600):
        s = stems[i % 4]
        rng_words.append(s if i % 3 else s[:-1])
    report = model_showdown(rng_words, window=10)
    assert report["autocopy_bits_per_word"] >= report["static_bits_per_word"]
    assert set(report) >= {"rho", "eta", "autocopy_advantage_bits", "bic_penalty_bits"}


def test_model_showdown_nested_never_worse():
    from voynich.diagnostics import model_showdown

    # Arbitrary distinct words: rho should fit near 0 and the advantage
    # must be >= 0 (the autocopy model nests the static one).
    words = [f"{i:03d}{i:03d}w" for i in range(400)]
    report = model_showdown(words, window=10)
    assert report["autocopy_advantage_bits"] >= -1e-9


def test_shuffled_baseline_reduces_locality():
    # Locality lives in ORDER; shuffling must lower the near rate
    clustered = []
    for i in range(50):
        clustered += [f"stem{i:03d}a", f"stem{i:03d}b", f"stem{i:03d}c"]
    real = self_citation_rates(clustered, window=5)["near_rate"]
    null = shuffled_rates(clustered, window=5, seed=1)["near_rate"]
    assert real > null + 0.3
