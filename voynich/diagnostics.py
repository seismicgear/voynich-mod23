"""
diagnostics.py — self-citation statistics (the leading non-language theory).

Timm & Schinner's observation: Voynichese words resemble *recently
written* words far more often than chance or natural language allows —
as if the scribe generated each word by copying and lightly mutating an
earlier one.  This module measures that signature directly:

  * the rate at which a word has a near-duplicate (edit distance <= 1)
    among the previous `window` words,
  * the same rate after shuffling word order (the no-locality null),
  * the same pair of rates for real reference-language corpora.

The quantity that matters is the LOCALITY EXCESS: rate minus shuffled
rate.  Natural languages show a small excess (sentences echo nearby
words); a copy-and-mutate generator shows a large one.  No key search is
involved — this interrogates the text itself.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone

from . import corpus as corpus_mod

DEFAULT_WINDOW = 15


def within_edit_one(a: str, b: str) -> bool:
    """Levenshtein distance <= 1, specialized and fast."""
    la, lb = len(a), len(b)
    if a == b:
        return True
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return True
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


MIN_WORD_LEN = 4


def self_citation_rates(words: list[str], window: int = DEFAULT_WINDOW) -> dict:
    """Fraction of content words (length >= MIN_WORD_LEN) with an exact /
    near (edit<=1) duplicate of similar length among the preceding
    `window` words.  Short function words are excluded on both sides —
    'and'/'the' repetition would otherwise drown the signal in any
    language."""
    n = len(words)
    eligible = [len(w) >= MIN_WORD_LEN for w in words]
    near = 0
    exact = 0
    targets = 0
    for t in range(1, n):
        if not eligible[t]:
            continue
        targets += 1
        lo = max(0, t - window)
        w = words[t]
        found_near = False
        found_exact = False
        for s in range(t - 1, lo - 1, -1):
            if not eligible[s]:
                continue
            if words[s] == w:
                found_exact = True
                found_near = True
                break
            if not found_near and within_edit_one(w, words[s]):
                found_near = True
        near += found_near
        exact += found_exact
    denom = max(targets, 1)
    return {"near_rate": near / denom, "exact_rate": exact / denom, "n_words": n}


def shuffled_rates(
    words: list[str],
    window: int = DEFAULT_WINDOW,
    seed: int | None = 0,
    n_shuffles: int = 3,
) -> dict:
    """Mean rates over order-shuffled copies — same vocabulary and
    frequencies, no locality."""
    rng = random.Random(seed)
    near = exact = 0.0
    for _ in range(n_shuffles):
        shuffled = list(words)
        rng.shuffle(shuffled)
        r = self_citation_rates(shuffled, window)
        near += r["near_rate"]
        exact += r["exact_rate"]
    return {"near_rate": near / n_shuffles, "exact_rate": exact / n_shuffles}


def _corpus_row(name: str, words: list[str], window: int, seed) -> dict:
    real = self_citation_rates(words, window)
    null = shuffled_rates(words, window, seed=seed)
    return {
        "corpus": name,
        "n_words": real["n_words"],
        "near_rate": real["near_rate"],
        "exact_rate": real["exact_rate"],
        "shuffled_near_rate": null["near_rate"],
        "locality_excess": real["near_rate"] - null["near_rate"],
    }


# ---- model showdown: static lexicon vs self-citation generator ------------

def _edit1_neighbor_count(word: str, alphabet_size: int = 24) -> int:
    """Approximate number of distinct strings at edit distance exactly 1
    (substitutions + insertions + deletions, ignoring collisions)."""
    L = len(word)
    return L * (alphabet_size - 1) + (L + 1) * alphabet_size + max(L, 1)


def _copy_kernel_stats(words: list[str], window: int) -> tuple[list[float], list[float]]:
    """Per position t: a_t = (count of exact matches in window)/window and
    b_t = (sum over window of 1[edit==1]/N1(source))/window.  These are
    sufficient statistics for the copy mixture at any (rho, eta)."""
    a_stats: list[float] = []
    b_stats: list[float] = []
    for t in range(len(words)):
        lo = max(0, t - window)
        hist = words[lo:t]
        if not hist:
            a_stats.append(0.0)
            b_stats.append(0.0)
            continue
        a = 0.0
        b = 0.0
        for s in hist:
            if s == words[t]:
                a += 1.0
            elif within_edit_one(words[t], s):
                b += 1.0 / _edit1_neighbor_count(s)
        a_stats.append(a / window)
        b_stats.append(b / window)
    return a_stats, b_stats


def model_showdown(
    words: list[str], window: int = DEFAULT_WINDOW
) -> dict:
    """Fit two generative models of the word stream on its first half and
    compare held-out log-likelihood on the second half:

      STATIC    P(w) = unigram lexicon, char-model backoff for unseen words
      AUTOCOPY  (1-rho)*STATIC + rho * copy-and-mutate kernel over the
                previous `window` words (Timm & Schinner's mechanism,
                with mutation = single-glyph edit)

    AUTOCOPY nests STATIC (rho=0), so the question is how many bits per
    word the copy mechanism buys on text the fit never saw — for genuine
    language the answer should be near zero."""
    import math as _math

    from .lm import CharNgramModel

    cut = len(words) // 2
    train, evalw = words[:cut], words[cut:]

    # Background lexicon: unigram with a char-ngram model for unseen mass.
    counts: dict[str, int] = {}
    for w in train:
        counts[w] = counts.get(w, 0) + 1
    n_train = len(train)
    char_lm = CharNgramModel(order=3).fit(" ".join(train))
    beta = 0.05  # unseen-word mass

    _bg_cache: dict[str, float] = {}

    def p_bg(w: str) -> float:
        cached = _bg_cache.get(w)
        if cached is None:
            p_char = 2.0 ** char_lm.word_score(
                [ord(c) - 97 for c in w if "a" <= c <= "z"]
            )
            p_uni = counts.get(w, 0) / n_train
            cached = (1 - beta) * p_uni + beta * max(p_char, 1e-12)
            _bg_cache[w] = cached
        return cached

    def sequence_ll(seq: list[str], all_words: list[str], offset: int,
                    rho: float, eta: float,
                    a_stats: list[float], b_stats: list[float]) -> float:
        total = 0.0
        for i, w in enumerate(seq):
            t = offset + i
            copy = (1 - eta) * a_stats[t] + eta * b_stats[t]
            p = (1 - rho) * p_bg(w) + rho * copy
            total += _math.log2(max(p, 1e-300))
        return total

    a_stats, b_stats = _copy_kernel_stats(words, window)

    # Fit rho, eta on the training half (grid search; the kernel
    # statistics are precomputed so each combination is O(n)).
    best = (-float("inf"), 0.0, 0.5)
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        for eta in (0.1, 0.3, 0.5, 0.7, 0.9):
            ll = sequence_ll(train, words, 0, rho, eta, a_stats, b_stats)
            if ll > best[0]:
                best = (ll, rho, eta)
    _, rho_hat, eta_hat = best

    n_eval = len(evalw)
    ll_static = sequence_ll(evalw, words, cut, 0.0, 0.5, a_stats, b_stats)
    ll_copy = sequence_ll(evalw, words, cut, rho_hat, eta_hat, a_stats, b_stats)
    advantage = (ll_copy - ll_static) / max(n_eval, 1)
    # BIC penalty for the two extra parameters, in bits per word.
    bic_penalty = 2 * _math.log2(max(n_eval, 2)) / (2 * max(n_eval, 1))

    return {
        "rho": rho_hat,
        "eta": eta_hat,
        "static_bits_per_word": ll_static / max(n_eval, 1),
        "autocopy_bits_per_word": ll_copy / max(n_eval, 1),
        "autocopy_advantage_bits": advantage,
        "bic_penalty_bits": bic_penalty,
        "n_eval_words": n_eval,
    }


def run_diagnostics(config: dict | None = None) -> dict:
    """Compare the self-citation signature of Voynichese against
    reference languages.  Config keys: currier_language, section, window,
    seed, references (list of reference keys to compare against)."""
    cfg = {
        "currier_language": "A",
        "section": None,
        "window": DEFAULT_WINDOW,
        "seed": 0,
        "references": ["latin", "english"],
        **(config or {}),
    }
    t0 = time.time()
    window = int(cfg["window"])

    lang = cfg["currier_language"]
    lines = corpus_mod.load_lines(
        language=None if lang in (None, "all") else lang,
        section=cfg["section"],
    )
    voy_words = [w for line in lines for w in line.words]
    if len(voy_words) < 200:
        raise RuntimeError("not enough words for diagnostics under these filters")

    voy_name = (
        f"Voynichese (Currier {lang}"
        + (f", section {cfg['section']}" if cfg["section"] else "")
        + ")"
    )
    rows = [_corpus_row(voy_name, voy_words, window, cfg["seed"])]
    showdown_rows = [{"corpus": voy_name, **model_showdown(voy_words, window)}]
    for ref in cfg["references"]:
        ref_words = corpus_mod.load_reference(ref).split()[: len(voy_words)]
        rows.append(_corpus_row(f"{ref} (reference)", ref_words, window, cfg["seed"]))
        showdown_rows.append(
            {"corpus": f"{ref} (reference)", **model_showdown(ref_words, window)}
        )

    voy_excess = rows[0]["locality_excess"]
    ref_excess = max(r["locality_excess"] for r in rows[1:]) if len(rows) > 1 else 0.0
    base = (
        f"Voynichese content words (>= {MIN_WORD_LEN} glyphs) have a "
        f"near-duplicate (edit distance <= 1) among the previous {window} words "
        f"{rows[0]['near_rate'] * 100:.1f}% of the time — a locality excess of "
        f"{voy_excess * 100:.1f} points over its shuffled null, versus at most "
        f"{ref_excess * 100:.1f} points for the reference languages. "
    )
    if ref_excess > 0 and voy_excess > 1.5 * ref_excess:
        verdict = base + (
            f"That is {voy_excess / ref_excess:.1f}x the strongest reference — "
            "the fingerprint of copy-and-mutate generation (Timm & Schinner's "
            "self-citation model), and hard for any cipher of natural language "
            "to produce: enciphering preserves word order but not this graded "
            "similarity clustering."
        )
    else:
        verdict = base + (
            "Under these filters the excess is comparable to ordinary "
            "language; this configuration does not separate the self-citation "
            "model from a linguistic source. Try other sections, windows or "
            "Currier languages — the published effect is strongest within "
            "paragraphs."
        )

    voy_adv = showdown_rows[0]["autocopy_advantage_bits"]
    ref_adv = max(
        (r["autocopy_advantage_bits"] for r in showdown_rows[1:]), default=0.0
    )
    bic = showdown_rows[0]["bic_penalty_bits"]
    showdown_verdict = (
        f"Model showdown (held-out): adding a copy-and-mutate mechanism to a "
        f"static lexicon buys {voy_adv:.3f} bits/word on Voynichese "
        f"(rho={showdown_rows[0]['rho']:.2f}) versus {ref_adv:.3f} bits/word at "
        f"best on the reference languages (BIC cost of the extra parameters: "
        f"{bic:.4f} bits/word). "
        + (
            "The copying mechanism explains held-out Voynichese far better "
            "than it explains language — evidence for the generated-text "
            "hypothesis."
            if voy_adv > max(3 * ref_adv, 10 * bic) and voy_adv > 0.05
            else "Under this configuration the copying advantage is modest; "
            "it does not separate generation from language on its own."
        )
    )

    return {
        "kind": "diagnostics",
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            "config": cfg,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "rows": rows,
        "showdown": showdown_rows,
        "verdict": verdict,
        "showdown_verdict": showdown_verdict,
    }
