"""
synthetic.py — solver validation on ciphers with known answers.

"Can this thing solve anything?" is answerable: encrypt a held-out slice
of a real-language corpus with a random substitution into an artificial
glyph alphabet, hand the solver only the ciphertext and a language model
trained on *different* text, and measure how much of the plaintext it
recovers.  This is the honest benchmark behind any claim the machinery
makes about the Voynich text itself.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .annealer import anneal, random_key_scores
from .cipher import EncodedCorpus, NgramView, decode_stream, encode_corpus
from .lm import ALPHABET, SPACE_ID, CharNgramModel, encode_text


def make_synthetic_cipher(
    plaintext: str, seed: int | None = None
) -> tuple[EncodedCorpus, np.ndarray, np.ndarray]:
    """Encrypt plaintext (a-z + spaces) with a random monoalphabetic
    substitution into fake glyph tokens.

    Returns (encoded_corpus, true_key, plaintext_char_ids) where
    plaintext_char_ids aligns 1:1 with the corpus token stream."""
    rng = np.random.default_rng(seed)
    ids = encode_text(plaintext)
    letters = sorted({int(i) for i in ids if i != SPACE_ID})

    # One fake glyph per plaintext letter, in shuffled order so glyph
    # index carries no information.
    perm = rng.permutation(len(letters))
    vocab = [f"g{j:02d}" for j in range(len(letters))]
    letter_to_token = {letters[i]: int(perm[i]) for i in range(len(letters))}

    # Build "lines" of ~8 words to mirror the Voynich line structure.
    words = plaintext.split()
    lines: list[list[list[str]]] = []
    for i in range(0, len(words), 8):
        chunk = words[i : i + 8]
        line = []
        for w in chunk:
            toks = [
                vocab[letter_to_token[c]]
                for c in (int(x) for x in encode_text(w))
            ]
            if toks:
                line.append(toks)
        if line:
            lines.append(line)

    corpus = encode_corpus(lines, vocab, n_states=1)

    true_key = np.full((1, len(vocab) + 1), SPACE_ID, dtype=np.int64)
    for letter, tok in letter_to_token.items():
        true_key[0, tok] = letter

    plain_ids = decode_stream(true_key, corpus)
    return corpus, true_key, plain_ids


def run_benchmark(
    text: str,
    order: int = 4,
    cipher_chars: int = 4000,
    iterations: int = 20_000,
    restarts: int = 2,
    seed: int | None = 0,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train an LM on the head of `text`, encrypt a slice from the tail,
    and measure recovery.  Returns a report dict."""
    cut = max(len(text) - cipher_chars - 1, len(text) // 2)
    lm_text = text[:cut]
    sample = text[cut : cut + cipher_chars]
    # Align to word boundaries.
    sample = sample[sample.find(" ") + 1 : sample.rfind(" ")]

    lm = CharNgramModel(order=order).fit(lm_text)
    corpus, true_key, plain_ids = make_synthetic_cipher(sample, seed=seed)
    view = NgramView(corpus, order)

    result = anneal(
        view,
        lm.logp,
        n_states=1,
        n_tokens=corpus.n_tokens,
        iterations=iterations,
        restarts=restarts,
        seed=seed,
        progress=progress,
        should_stop=should_stop,
    )

    decoded = decode_stream(result.best_key, corpus)
    letter_mask = plain_ids != SPACE_ID
    accuracy = float((decoded[letter_mask] == plain_ids[letter_mask]).mean())

    rand_scores = random_key_scores(
        view, lm.logp, 1, corpus.n_tokens, n_samples=20, seed=seed
    )
    preview = "".join(ALPHABET[i] for i in decoded[:300])
    truth = "".join(ALPHABET[i] for i in plain_ids[:300])

    return {
        "accuracy": accuracy,
        "best_score": result.best_score,
        "true_key_score": view.score(true_key, lm.logp),
        "random_key_score_mean": float(np.mean(rand_scores)),
        "iterations_done": result.iterations_done,
        "restarts_done": result.restarts_done,
        "elapsed_sec": result.elapsed_sec,
        "history": result.history,
        "decoded_preview": preview,
        "plaintext_preview": truth,
        "cipher_letters": int(letter_mask.sum()),
    }
