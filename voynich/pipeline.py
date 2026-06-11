"""
pipeline.py — end-to-end orchestration of Voynich decipherment runs.

A solve run:
  1. loads the Voynich lines (Currier language / section filters),
  2. tokenizes them (core EVA multigraphs + learned BPE merges),
  3. splits lines into train (even) / held-out test (odd),
  4. trains a character n-gram LM on the chosen reference corpus,
  5. searches a key space on the training half:
       simple       one global token -> letter key
       positional   line-start / body / line-end keys
       abbreviation token -> 1-2 letters (scribal-abbreviation family),
  6. evaluates the best key on the held-out half against two anchors:
     the random-key floor and the reference corpus's self-score ceiling.

The verdict is reported as "gap closed": how far the held-out score
moved from the chance floor toward the real-language ceiling.  A genuine
decipherment would close most of that gap; structure-only artifacts
close a sliver of it.

A sweep runs the same configuration across every reference language and
ranks the results — the direct answer to "which language fits best?".
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from . import corpus as corpus_mod
from .annealer import (
    anneal,
    anneal_anagram,
    anneal_expansion,
    anneal_nomenclator,
    random_expansion_scores,
    random_key_scores,
)
from .words import (
    AnagramScorer,
    NomenclatorScorer,
    WordDictionary,
    WordTypes,
    alphagram_text,
    evidence_locks,
    frequency_init_key,
    word_match_rate,
)
from .cipher import (
    ExpansionScorer,
    NgramView,
    decode_lines,
    decode_lines_expanded,
    decode_stream,
    encode_corpus,
    expand_stream,
    expansion_key_table,
    key_table,
)
from .lm import ALPHABET, CharNgramModel, encode_text
from .synthetic import staged_expansion_init
from .tokenizer import EvaTokenizer

from .paths import default_results_dir

RESULTS_DIR = default_results_dir()

HYPOTHESES = ("simple", "positional", "abbreviation", "anagram", "nomenclator")
LOCKABLE_HYPOTHESES = ("simple", "anagram")

DEFAULT_CONFIG = {
    "currier_language": "A",     # 'A', 'B' or 'all'
    "section": None,             # section code or None
    "reference": "latin",        # any key in corpus.REFERENCE_SOURCES
    "hypothesis": "simple",      # see HYPOTHESES
    "abjad": False,              # score against consonant skeletons
    "reverse": "none",           # none | words | lines (mirror-writing theory)
    "control": False,            # shuffle tokens: the null-control calibration
    "allow_nulls": False,        # abbreviation only: tokens may decode to nothing
    "lock_rounds": 0,            # crib-locking iterations (simple/anagram)
    "order": 4,
    "bpe_merges": 30,
    "iterations": 60_000,
    "restarts": 3,
    "seed": None,
}

_VOWELS = set("aeiou")


def strip_vowels(text: str) -> str:
    """Consonant skeleton of a cleaned corpus — the abjad hypothesis
    (vowels unwritten, as in Hebrew or Arabic script conventions).  Words
    that were all vowels vanish; multiple spaces collapse."""
    out = "".join(c for c in text if c not in _VOWELS)
    return " ".join(out.split())


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
    if cfg["hypothesis"] not in HYPOTHESES:
        raise ValueError(f"unknown hypothesis: {cfg['hypothesis']}")
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
    if cfg.get("abjad"):
        ref_text = strip_vowels(ref_text)

    # 2. Tokenize
    all_words = [w for line in lines for w in line.words]
    tok = EvaTokenizer.train(all_words, num_merges=cfg["bpe_merges"])
    tokenized = [[tok.tokenize(w) for w in line.words] for line in lines]
    vocab = tok.build_vocab([line.words for line in lines])

    # 2b. Reading-order and control transforms.
    reverse = cfg.get("reverse", "none")
    if reverse == "words":
        # Mirror writing: glyph order reversed inside each word.
        tokenized = [[w[::-1] for w in line] for line in tokenized]
    elif reverse == "lines":
        # Full right-to-left reading: word order and glyph order reversed.
        tokenized = [[w[::-1] for w in line[::-1]] for line in tokenized]
    elif reverse != "none":
        raise ValueError("reverse must be none, words or lines")
    if cfg.get("control"):
        # Null control: scramble which token sits where, preserving word
        # lengths, line structure and the token multiset.  Any "signal" a
        # hypothesis also finds here is objective-gaming, not Voynichese.
        flat = [t for line in tokenized for w in line for t in w]
        perm = np.random.default_rng(cfg["seed"]).permutation(len(flat))
        flat = [flat[i] for i in perm]
        it = iter(flat)
        tokenized = [[[next(it) for _ in w] for w in line] for line in tokenized]

    # 3. Split lines: train on even, validate on odd
    train_lines = tokenized[0::2]
    test_lines = tokenized[1::2]

    hypothesis = cfg["hypothesis"]
    n_states = 3 if hypothesis == "positional" else 1
    train_corpus = encode_corpus(train_lines, vocab, n_states=n_states)
    test_corpus = encode_corpus(test_lines, vocab, n_states=n_states)

    tok_to_id = {t: i for i, t in enumerate(vocab)}

    def to_ids(token_lines):
        return [[[tok_to_id[t] for t in w] for w in line] for line in token_lines]

    dictionary = WordDictionary(ref_text)
    lock_rounds = int(cfg.get("lock_rounds", 0) or 0)
    if lock_rounds and hypothesis not in LOCKABLE_HYPOTHESES:
        raise ValueError(
            f"lock_rounds is only supported for {LOCKABLE_HYPOTHESES}"
        )
    if cfg.get("allow_nulls") and hypothesis != "abbreviation":
        raise ValueError("allow_nulls requires the abbreviation hypothesis")
    code_table: list[dict] = []
    locking_log: list[dict] = []

    # 4. Language model + anchors
    order = cfg["order"]
    lm = CharNgramModel(order=order).fit(ref_text)
    ceiling = reference_self_score(lm, ref_text)

    # 5/6. Search + held-out evaluation (per hypothesis family)
    if hypothesis == "anagram":
        train_types = WordTypes(to_ids(train_lines))
        test_types = WordTypes(to_ids(test_lines))
        alpha_lm = CharNgramModel(order=order).fit(alphagram_text(ref_text))
        train_scorer = AnagramScorer(train_types, dictionary, alphagram_lm=alpha_lm)
        test_scorer = AnagramScorer(test_types, dictionary, alphagram_lm=alpha_lm)
        n_tok = len(vocab)

        init = frequency_init_key(train_types, dictionary, n_tokens=n_tok)
        locked = None
        for rnd in range(lock_rounds + 1):
            result = anneal_anagram(
                train_scorer,
                n_tokens=n_tok,
                iterations=cfg["iterations"],
                restarts=cfg["restarts"],
                seed=cfg["seed"],
                init_key=init,
                locked_letters=locked,
                progress=progress,
                should_stop=should_stop,
            )
            best_key = result.best_key
            matched, _ = train_scorer.match_info(best_key, min_len=3)
            lock_mask = evidence_locks(train_types, matched, n_tok)
            new_locked = np.where(lock_mask, best_key, -1)
            locking_log.append(
                {
                    "round": rnd,
                    "locked_tokens": int(lock_mask.sum()),
                    "train_score": result.best_score,
                }
            )
            done = (
                rnd == lock_rounds
                or (result.stopped_early)
                or (locked is not None and np.array_equal(new_locked, locked))
            )
            if done:
                break
            locked = new_locked
            init = best_key.copy()

        train_score = result.best_score
        test_score = test_scorer.score_key(best_key)
        rng = np.random.default_rng(cfg["seed"])
        rand_scores = [
            test_scorer.score_key(
                rng.integers(0, 26, size=n_tok, dtype=np.int64)
            )
            for _ in range(30)
        ]
        # Word-level ceiling: what the reference corpus scores against its
        # own dictionary (same units, bits per word token).
        ceiling = dictionary.expected_word_score(ref_text)

        # Decoded sample: matched words read as words, the rest as raw
        # decoded letters in written order.
        test_matched, test_words = test_scorer.match_info(best_key, min_len=1)
        type_index = {
            tuple(int(x) for x in t): i for i, t in enumerate(test_types.types)
        }

        def render_word(word_ids: list[int]) -> str:
            ti = type_index.get(tuple(word_ids))
            if ti is not None and test_matched[ti]:
                return test_words[ti]
            return "".join(ALPHABET[best_key[t]] for t in word_ids) + "?"

        test_ids = to_ids(test_lines)
        decoded_sample = [
            " ".join(render_word(w) for w in line) for line in test_ids[:25]
        ]
        decoded_text = "\n".join(
            " ".join(render_word(w) for w in line) for line in test_ids
        )
        def _sig_match_rate(min_len: int) -> float:
            mask = np.array([len(t) >= min_len for t in test_types.types])
            denom = test_types.freqs[mask].sum()
            return float(
                test_types.freqs[test_matched & mask].sum() / max(denom, 1)
            )

        match_rate = _sig_match_rate(3)
        match_rate_long = _sig_match_rate(5)
        key_rows = [
            {"token": tok, "all": ALPHABET[best_key[i]]}
            for i, tok in enumerate(vocab)
        ]
        decoded_ids = None
    elif hypothesis == "nomenclator":
        train_types = WordTypes(to_ids(train_lines))
        test_types = WordTypes(to_ids(test_lines))
        scorer = NomenclatorScorer(train_types, dictionary, lm)
        init = staged_expansion_init(
            train_corpus, lm, order,
            iterations=min(cfg["iterations"], 15_000), seed=cfg["seed"],
        )[: len(vocab), 0]
        result = anneal_nomenclator(
            scorer,
            n_tokens=len(vocab),
            iterations=cfg["iterations"],
            restarts=cfg["restarts"],
            seed=cfg["seed"],
            init_key=init.copy(),
            progress=progress,
            should_stop=should_stop,
        )
        best_key = result.best_key
        codebook = scorer.codebook()
        train_score = result.best_score
        test_score = scorer.score_key(best_key, codebook, test_types)
        rng = np.random.default_rng(cfg["seed"])
        rand_scores = [
            scorer.score_key(
                rng.integers(0, 26, size=len(vocab), dtype=np.int64),
                {},
                test_types,
            )
            for _ in range(30)
        ]
        ceiling = dictionary.expected_word_score(ref_text)

        def render_word(word_ids: list[int]) -> str:
            ci = codebook.get(tuple(word_ids), -1)
            if ci >= 0:
                return f"[{scorer.code_words[ci][0]}]"  # codebook word
            return "".join(ALPHABET[best_key[t]] for t in word_ids)

        test_ids = to_ids(test_lines)
        decoded_sample = [
            " ".join(render_word(w) for w in line) for line in test_ids[:25]
        ]
        decoded_text = "\n".join(
            " ".join(render_word(w) for w in line) for line in test_ids
        )
        # Match metrics over SPELLED words only — codebook words are real
        # words by construction and would flatter the rate.
        spelled_words = [
            ("".join(ALPHABET[best_key[t]] for t in word_ids), 1)
            for line in test_ids
            for word_ids in line
            if tuple(word_ids) not in codebook
        ]
        match_rate = word_match_rate(spelled_words, dictionary)
        match_rate_long = word_match_rate(spelled_words, dictionary, min_len=5)
        key_rows = [
            {"token": tok, "all": ALPHABET[best_key[i]]}
            for i, tok in enumerate(vocab)
        ]
        # Expose the codebook itself — the most interesting artifact.
        id_to_tok = {i: t for i, t in enumerate(vocab)}
        code_table = sorted(
            (
                {
                    "voynich_word": "".join(id_to_tok[t] for t in tup),
                    "plaintext_word": scorer.code_words[ci][0],
                }
                for tup, ci in codebook.items()
            ),
            key=lambda r: r["voynich_word"],
        )
        decoded_ids = None
    elif hypothesis == "abbreviation":
        allow_nulls = bool(cfg.get("allow_nulls", False))
        frac = 0.5 if allow_nulls else 0.0
        train_scorer = ExpansionScorer(train_corpus, lm, min_output_frac=frac)
        test_scorer = ExpansionScorer(test_corpus, lm, min_output_frac=frac)
        init_key = staged_expansion_init(
            train_corpus, lm, order,
            iterations=min(cfg["iterations"], 15_000), seed=cfg["seed"],
        )
        result = anneal_expansion(
            train_scorer,
            n_tokens=train_corpus.n_tokens,
            iterations=cfg["iterations"],
            restarts=cfg["restarts"],
            seed=cfg["seed"],
            init_key=init_key,
            allow_nulls=allow_nulls,
            progress=progress,
            should_stop=should_stop,
        )
        # Anchors and held-out evaluation in per-char units, comparable
        # across hypotheses (the per-token objective is search-internal).
        test_score = test_scorer.per_char(result.best_key)
        train_score = train_scorer.per_char(result.best_key)
        rng = np.random.default_rng(cfg["seed"])
        from .cipher import random_expansion_key

        rand_scores = [
            test_scorer.per_char(random_expansion_key(test_corpus.n_tokens, rng))
            for _ in range(30)
        ]
        decoded_sample = decode_lines_expanded(
            result.best_key, test_lines, vocab, max_lines=25
        )
        key_rows = expansion_key_table(result.best_key, vocab)
        decoded_ids = expand_stream(result.best_key, test_corpus.token_stream)
    else:
        train_view = NgramView(train_corpus, order)
        test_view = NgramView(test_corpus, order)
        rand_scores = random_key_scores(
            test_view, lm.logp, n_states, test_corpus.n_tokens,
            n_samples=30, seed=cfg["seed"],
        )
        train_types = WordTypes(to_ids(train_lines)) if lock_rounds else None
        locked = None
        for rnd in range(lock_rounds + 1):
            result = anneal(
                train_view,
                lm.logp,
                n_states=n_states,
                n_tokens=train_corpus.n_tokens,
                iterations=cfg["iterations"],
                restarts=cfg["restarts"],
                seed=cfg["seed"],
                locked_letters=locked,
                progress=progress,
                should_stop=should_stop,
            )
            if not lock_rounds:
                break
            # Crib locking: freeze tokens corroborated by exact dictionary
            # matches and re-anneal the rest.
            key_row = result.best_key[0]
            matched = np.array(
                [
                    len(t) >= 3
                    and "".join(ALPHABET[key_row[x]] for x in t)
                    in dictionary.word_logp
                    for t in train_types.types
                ]
            )
            # The space token never occurs inside words, so lock_mask
            # leaves it free; moves already pin it anyway.
            lock_mask = evidence_locks(
                train_types, matched, train_corpus.n_tokens
            )
            new_locked = np.where(lock_mask, key_row, -1)
            locking_log.append(
                {
                    "round": rnd,
                    "locked_tokens": int(lock_mask.sum()),
                    "train_score": result.best_score,
                }
            )
            if (
                rnd == lock_rounds
                or result.stopped_early
                or (locked is not None and np.array_equal(new_locked, locked))
            ):
                break
            locked = new_locked
        test_score = test_view.score(result.best_key, lm.logp)
        train_score = result.best_score
        decoded_sample = decode_lines(
            result.best_key, test_lines, vocab, n_states, max_lines=25
        )
        key_rows = key_table(result.best_key, vocab)
        decoded_ids = decode_stream(result.best_key, test_corpus)

    # Dictionary word-match rate on held-out lines: the "is it language
    # yet?" diagnostic, shared by every hypothesis.  The anagram and
    # nomenclator branches compute their own rates above.
    if hypothesis not in ("anagram", "nomenclator"):
        if hypothesis == "abbreviation":
            all_decoded = decode_lines_expanded(result.best_key, test_lines, vocab)
        else:
            all_decoded = decode_lines(result.best_key, test_lines, vocab, n_states)
        decoded_words = [(w, 1) for line in all_decoded for w in line.split()]
        match_rate = word_match_rate(decoded_words, dictionary)
        match_rate_long = word_match_rate(decoded_words, dictionary, min_len=5)

    floor = float(np.mean(rand_scores))
    gap_closed = (test_score - floor) / (ceiling - floor) if ceiling > floor else 0.0

    odd_lines = lines[1::2]
    sample_folios = [
        f"{odd_lines[i].folio}.{odd_lines[i].line_number}"
        for i in range(min(25, len(test_lines)))
    ]
    if decoded_ids is not None:
        decoded_text = "".join(ALPHABET[i] for i in decoded_ids)

    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            "config": cfg,
            "reference_label": corpus_mod.REFERENCE_SOURCES[cfg["reference"]].label,
            "n_lines": len(lines),
            "n_words": len(all_words),
            "vocab_size": len(vocab),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "scores": {
            "train_best": train_score,
            "test_heldout": test_score,
            "random_key_floor": floor,
            "random_key_std": float(np.std(rand_scores)),
            "reference_ceiling": ceiling,
            "gap_closed": gap_closed,
            "word_match_rate": match_rate,
            "word_match_rate_long": match_rate_long,
        },
        "locking": locking_log,
        "code_table": code_table,
        "verdict": _verdict(gap_closed, match_rate, match_rate_long),
        "history": result.history,
        "key": key_rows,
        "decoded_sample": [
            {"ref": ref, "text": txt}
            for ref, txt in zip(sample_folios, decoded_sample)
        ],
        "decoded_text_head": decoded_text[:2000],
        "stopped_early": result.stopped_early,
    }
    return report


def _verdict(
    gap_closed: float,
    match_rate: float | None = None,
    match_rate_long: float | None = None,
) -> str:
    pct = gap_closed * 100
    matches = (
        f" Dictionary matches on held-out words: {match_rate * 100:.1f}% "
        f"(len>=3), {match_rate_long * 100:.1f}% (len>=5) — real text scores "
        "near 100% on both; junk keys hit short words by chance and long "
        "words almost never."
        if match_rate is not None and match_rate_long is not None
        else ""
    )
    if gap_closed > 1.0:
        return (
            f"Gap closed: {pct:.1f}% — ABOVE the real-language ceiling. With "
            "word-multiset (anagram) scoring this means the optimizer stuffed "
            "the text with short, frequent dictionary words; it is gaming the "
            "objective, not reading the manuscript. Trust the long-word match "
            "rate instead." + matches
        )
    if gap_closed >= 0.85:
        return (
            f"Gap closed: {pct:.1f}%. The held-out text scores close to real "
            "reference-language text. If this is reproducible across seeds and "
            "the decoded sample reads as language, treat it as a serious "
            "candidate — and expect to be wrong anyway; check the sample."
            + matches
        )
    if gap_closed >= 0.5:
        return (
            f"Gap closed: {pct:.1f}%. Substantially better than chance, far "
            "from real language. Typical of a wrong-but-structured hypothesis: "
            "the optimizer exploits Voynichese's rigid word structure without "
            "producing language." + matches
        )
    return (
        f"Gap closed: {pct:.1f}%. The mapping generalizes barely better than "
        "random keys. Under this hypothesis/reference pairing the text does "
        "not behave like a simple substitution of that language." + matches
    )


def sweep_references(
    config: dict | None = None,
    references: list[str] | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Run the same solve configuration against every reference language
    and rank the outcomes by gap closed.  This is the experiment that
    answers 'which candidate language fits Voynichese best — and does any
    of them actually fit?'."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    refs = references or list(corpus_mod.REFERENCE_SOURCES)
    for r in refs:
        if r not in corpus_mod.REFERENCE_SOURCES:
            raise ValueError(f"unknown reference: {r}")

    t0 = time.time()
    rows = []
    for i, ref in enumerate(refs):
        if should_stop is not None and should_stop():
            break

        def sub_progress(p: dict, _i=i, _ref=ref) -> None:
            if progress is not None:
                progress({**p, "language": _ref, "lang_index": _i, "n_langs": len(refs)})

        run_cfg = {**cfg, "reference": ref}
        report = solve_voynich(run_cfg, progress=sub_progress, should_stop=should_stop)
        s = report["scores"]
        rows.append(
            {
                "reference": ref,
                "label": report["meta"]["reference_label"],
                "family": corpus_mod.REFERENCE_SOURCES[ref].family,
                "gap_closed": s["gap_closed"],
                "word_match_rate": s["word_match_rate"],
                "test_heldout": s["test_heldout"],
                "random_key_floor": s["random_key_floor"],
                "reference_ceiling": s["reference_ceiling"],
                "sample": report["decoded_sample"][1]["text"][:80]
                if len(report["decoded_sample"]) > 1
                else "",
            }
        )

    rows.sort(key=lambda r: r["gap_closed"], reverse=True)
    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S"),
            "config": cfg,
            "references": refs,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "table": rows,
        "note": (
            "Ranked by gap closed on held-out lines. Verdicts in the 50-85% "
            "band with unreadable samples mean the optimizer matched the "
            "language's statistics, not its words. A real hit would pair a "
            ">85% gap with a sample a reader of that language can parse."
        ),
    }


def save_report(report: dict, results_dir: pathlib.Path | None = None) -> pathlib.Path:
    rdir = results_dir or RESULTS_DIR
    rdir.mkdir(parents=True, exist_ok=True)
    kind = report.get("kind") or ("sweep" if "table" in report else "solve")
    out = rdir / f"{kind}_{report['meta']['timestamp']}.json"
    out.write_text(json.dumps(report, indent=2))
    return out
