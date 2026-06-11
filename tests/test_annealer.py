"""The load-bearing test: the annealer must crack a substitution cipher
with a known answer. If this fails, no claim about the Voynich text means
anything."""

from voynich.synthetic import run_benchmark


def test_solver_recovers_synthetic_cipher(english_text):
    report = run_benchmark(
        english_text,
        order=4,
        cipher_chars=3000,
        iterations=15000,
        restarts=2,
        seed=11,
    )
    assert report["accuracy"] >= 0.95
    # The solver should land at (or above) the true key's score.
    assert report["best_score"] >= report["true_key_score"] - 0.05
    # And far above chance.
    assert report["best_score"] > report["random_key_score_mean"] + 2.0


def test_solver_recovers_abbreviation_cipher(english_text):
    """The expansion-key solver must crack a scribal-abbreviation cipher
    (frequent bigrams written as single signs). Harder than plain
    substitution — segmentation errors are coupled — so the bar is lower
    but still far above anything chance produces."""
    report = run_benchmark(
        english_text,
        order=4,
        cipher_chars=3500,
        iterations=30000,
        restarts=2,
        seed=7,
        mode="abbreviation",
    )
    assert report["accuracy"] >= 0.75
    assert report["best_score"] > report["random_key_score_mean"] + 2.0


def test_solver_recovers_anagram_cipher(english_text):
    """The word-level solver must crack a cipher where letters are
    shuffled inside every word — order statistics destroyed, multisets
    intact. Frequency init + alphagram-LM gradient carry this."""
    report = run_benchmark(
        english_text,
        cipher_chars=3000,
        iterations=12000,
        restarts=2,
        seed=3,
        mode="anagram",
    )
    assert report["accuracy"] >= 0.6
    assert report["best_score"] > report["random_key_score_mean"] + 5.0


def test_stop_flag_aborts_early(english_text):
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    report = run_benchmark(
        english_text,
        order=3,
        cipher_chars=1500,
        iterations=50000,
        restarts=3,
        seed=0,
        should_stop=stop_after_two,
    )
    assert report["iterations_done"] < 50000 * 3
