"""
pipeline.py — end-to-end orchestration of a Voynich decipherment run.

A run:
  1. loads the Voynich lines (Currier language / section filters),
  2. tokenizes them (core EVA multigraphs + learned BPE merges),
  3. splits lines into train (even) / held-out test (odd),
  4. trains a character n-gram LM on the chosen reference corpus,
  5. anneals a substitution key on the training half,
  6. evaluates the best key on the held-out half against two anchors:
     the random-key floor and the reference corpus's self-score ceiling.

The verdict is reported as "gap closed": how far the held-out score
moved from the chance floor toward the real-language ceiling.  A genuine
decipherment would close most of that gap; structure-only artifacts
close a sliver of it.
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from . import corpus as corpus_mod
from .annealer import anneal, random_key_scores
from .cipher import NgramView, decode_lines, decode_stream, encode_corpus, key_table
from .lm import ALPHABET, CharNgramModel, encode_text
from .tokenizer import EvaTokenizer

RESULTS_DIR = pathlib.Path("results")

DEFAULT_CONFIG = {
    "currier_language": "A",     # 'A', 'B' or 'all'
    "section": None,             # section code or None
    "reference": "latin",        # latin | italian | english
    "hypothesis": "simple",      # simple | positional
    "order": 4,
    "bpe_merges": 30,
    "iterations": 60_000,
    "restarts": 3,
    "seed": None,
}


def reference_self_score(lm: CharNgramModel, text: str, holdout_frac: float = 0.1) -> float:
    """Score of the tail of the reference corpus under the LM — the
    ceiling a perfect decipherment of same-language text would approach."""
    tail = text[int(len(text) * (1 - holdout_frac)):]
    return lm.score_ids(encode_text(tail))


def solve_voynich(
    config: dict | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    t0 = time.time()

    # 1. Data
    lang = cfg["currier_language"]
    lines = corpus_mod.load_lines(
        language=None if lang in (None, "all") else lang,
        section=cfg["section"],
    )
    if len(lines) < 20:
        raise RuntimeError(
            f"Only {len(lines)} lines matched the filters — not enough to solve on."
        )
    ref_text = corpus_mod.load_reference(cfg["reference"])

    # 2. Tokenize
    all_words = [w for line in lines for w in line.words]
    tok = EvaTokenizer.train(all_words, num_merges=cfg["bpe_merges"])
    tokenized = [[tok.tokenize(w) for w in line.words] for line in lines]
    vocab = tok.build_vocab([line.words for line in lines])

    # 3. Split lines: train on even, validate on odd
    train_lines = tokenized[0::2]
    test_lines = tokenized[1::2]

    n_states = 3 if cfg["hypothesis"] == "positional" else 1
    train_corpus = encode_corpus(train_lines, vocab, n_states=n_states)
    test_corpus = encode_corpus(test_lines, vocab, n_states=n_states)

    # 4. Language model + anchors
    order = cfg["order"]
    lm = CharNgramModel(order=order).fit(ref_text)
    train_view = NgramView(train_corpus, order)
    test_view = NgramView(test_corpus, order)

    rand_scores = random_key_scores(
        test_view, lm.logp, n_states, test_corpus.n_tokens,
        n_samples=30, seed=cfg["seed"],
    )
    ceiling = reference_self_score(lm, ref_text)

    # 5. Anneal on the training half
    result = anneal(
        train_view,
        lm.logp,
        n_states=n_states,
        n_tokens=train_corpus.n_tokens,
        iterations=cfg["iterations"],
        restarts=cfg["restarts"],
        seed=cfg["seed"],
        progress=progress,
        should_stop=should_stop,
    )

    # 6. Held-out evaluation
    test_score = test_view.score(result.best_key, lm.logp)
    floor = float(np.mean(rand_scores))
    gap_closed = (test_score - floor) / (ceiling - floor) if ceiling > floor else 0.0

    decoded_sample = decode_lines(
        result.best_key, test_lines, vocab, n_states, max_lines=25
    )
    sample_folios = [
        f"{lines[1::2][i].folio}.{lines[1::2][i].line_number}"
        for i in range(min(25, len(test_lines)))
    ]

    decoded_ids = decode_stream(result.best_key, test_corpus)
    decoded_text = "".join(ALPHABET[i] for i in decoded_ids)

    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            "config": cfg,
            "n_lines": len(lines),
            "n_words": len(all_words),
            "vocab_size": len(vocab),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "scores": {
            "train_best": result.best_score,
            "test_heldout": test_score,
            "random_key_floor": floor,
            "random_key_std": float(np.std(rand_scores)),
            "reference_ceiling": ceiling,
            "gap_closed": gap_closed,
        },
        "verdict": _verdict(gap_closed),
        "history": result.history,
        "key": key_table(result.best_key, vocab),
        "decoded_sample": [
            {"ref": ref, "text": txt}
            for ref, txt in zip(sample_folios, decoded_sample)
        ],
        "decoded_text_head": decoded_text[:2000],
        "stopped_early": result.stopped_early,
    }
    return report


def _verdict(gap_closed: float) -> str:
    pct = gap_closed * 100
    if gap_closed >= 0.85:
        return (
            f"Gap closed: {pct:.1f}%. The held-out text scores close to real "
            "reference-language text. If this is reproducible across seeds and "
            "the decoded sample reads as language, treat it as a serious "
            "candidate — and expect to be wrong anyway; check the sample."
        )
    if gap_closed >= 0.5:
        return (
            f"Gap closed: {pct:.1f}%. Substantially better than chance, far "
            "from real language. Typical of a wrong-but-structured hypothesis: "
            "the optimizer exploits Voynichese's rigid word structure without "
            "producing language."
        )
    return (
        f"Gap closed: {pct:.1f}%. The mapping generalizes barely better than "
        "random keys. Under this hypothesis/reference pairing the text does "
        "not behave like a simple substitution of that language."
    )


def save_report(report: dict, results_dir: pathlib.Path | None = None) -> pathlib.Path:
    rdir = results_dir or RESULTS_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / f"solve_{report['meta']['timestamp']}.json"
    out.write_text(json.dumps(report, indent=2))
    return out
