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

    rows = [
        _corpus_row(
            f"Voynichese (Currier {lang}"
            + (f", section {cfg['section']}" if cfg["section"] else "")
            + ")",
            voy_words,
            window,
            cfg["seed"],
        )
    ]
    for ref in cfg["references"]:
        ref_words = corpus_mod.load_reference(ref).split()[: len(voy_words)]
        rows.append(_corpus_row(f"{ref} (reference)", ref_words, window, cfg["seed"]))

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

    return {
        "kind": "diagnostics",
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            "config": cfg,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "rows": rows,
        "verdict": verdict,
    }
