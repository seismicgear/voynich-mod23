"""
annealer.py — simulated annealing over substitution keys.

Search state: an int key of shape (n_states, n_tokens) mapping each
glyph token (per positional state) to a plaintext letter.  Proposals
either reassign one token to a new letter or swap the letters of two
tokens.  Reassignment matters because good keys are many-to-one (the EVA
token inventory is larger than the target alphabet); swap-only search
cannot change which letters are over- or under-represented.

The objective is the mean log2 probability per character of the decoded
text under a reference-language n-gram model, evaluated exactly via the
compressed NgramView.  Acceptance follows the Metropolis criterion with
a geometric cooling schedule, and the whole search is repeated from
independent random starts (restarts), keeping the global best.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cipher import NgramView, random_key
from .lm import A


@dataclass
class AnnealResult:
    best_key: np.ndarray
    best_score: float
    history: list[tuple[int, float]] = field(default_factory=list)
    iterations_done: int = 0
    restarts_done: int = 0
    stopped_early: bool = False
    elapsed_sec: float = 0.0


def anneal(
    view: NgramView,
    logp: np.ndarray,
    n_states: int,
    n_tokens: int,
    iterations: int = 60_000,
    restarts: int = 3,
    t_start: float = 0.08,
    t_end: float = 0.0008,
    p_swap: float = 0.5,
    seed: int | None = None,
    locked_letters: np.ndarray | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_every: int = 500,
) -> AnnealResult:
    """Run simulated annealing; returns the best key found across restarts.

    `progress` (if given) is called every `progress_every` iterations with
    a dict: restart, iteration, total_iterations, temperature, score,
    best_score.  `should_stop` is polled at the same cadence; returning
    True aborts cleanly with the best result so far.

    `locked_letters` (length n_tokens, -1 = free) pins tokens to fixed
    letters across every state and restart — the crib-locking mechanism.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    global_best_key: np.ndarray | None = None
    global_best_score = -math.inf
    history: list[tuple[int, float]] = []
    iters_done = 0
    restarts_done = 0
    stopped = False

    # Space is pinned; movable tokens are everything else minus locks.
    movable_idx = np.arange(n_tokens - 1)
    if locked_letters is not None:
        movable_idx = movable_idx[locked_letters[: n_tokens - 1] < 0]
    if len(movable_idx) < 2:
        raise ValueError("fewer than two unlocked tokens — nothing to search")
    cooling = (t_end / t_start) ** (1.0 / max(iterations - 1, 1))

    for restart in range(restarts):
        key = random_key(n_states, n_tokens, rng)
        if locked_letters is not None:
            pinned = locked_letters >= 0
            key[:, pinned] = locked_letters[pinned]
        score = view.score(key, logp)
        best_key = key.copy()
        best_score = score
        temp = t_start

        for it in range(iterations):
            state = int(rng.integers(0, n_states))
            if rng.random() < p_swap:
                a, b = rng.choice(movable_idx, size=2, replace=False)
                key[state, a], key[state, b] = key[state, b], key[state, a]
                undo = ("swap", state, int(a), int(b))
            else:
                tok = int(movable_idx[rng.integers(0, len(movable_idx))])
                old = int(key[state, tok])
                # New letter uniform over the 25 letters != old; space
                # (id 26) is unreachable since draws cap at 25.
                new = int(rng.integers(0, A - 2))
                if new >= old:
                    new += 1
                key[state, tok] = new
                undo = ("set", state, tok, old)

            new_score = view.score(key, logp)
            delta = new_score - score
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                score = new_score
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
            else:
                if undo[0] == "swap":
                    _, s, a_, b_ = undo
                    key[s, a_], key[s, b_] = key[s, b_], key[s, a_]
                else:
                    _, s, tok_, old_ = undo
                    key[s, tok_] = old_

            temp *= cooling
            iters_done += 1

            if (it + 1) % progress_every == 0 or it == iterations - 1:
                history.append((restart * iterations + it + 1, best_score))
                if progress is not None:
                    progress(
                        {
                            "restart": restart,
                            "restarts": restarts,
                            "iteration": it + 1,
                            "total_iterations": iterations,
                            "temperature": temp,
                            "score": score,
                            "best_score": max(best_score, global_best_score),
                        }
                    )
                if should_stop is not None and should_stop():
                    stopped = True
                    break

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_key = best_key
        restarts_done = restart + 1
        if stopped:
            break

    return AnnealResult(
        best_key=global_best_key,
        best_score=global_best_score,
        history=history,
        iterations_done=iters_done,
        restarts_done=restarts_done,
        stopped_early=stopped,
        elapsed_sec=time.time() - t0,
    )


def anneal_expansion(
    scorer,
    n_tokens: int,
    iterations: int = 40_000,
    restarts: int = 3,
    t_start: float = 0.08,
    t_end: float = 0.0008,
    seed: int | None = None,
    init_key: np.ndarray | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_every: int = 500,
) -> AnnealResult:
    """Simulated annealing over expansion keys (token -> 1 or 2 letters).

    Same schedule and bookkeeping as `anneal`, with a move set that can
    grow, shrink and permute expansions:
      - set the first letter of a token
      - set or replace a token's second letter
      - clear a token's second letter (back to a single-letter decode)
      - swap the full expansions of two tokens

    `init_key` (if given) seeds the first restart — staging from the best
    plain-substitution key is far better than a cold start, because the
    coupled segmentation errors (which token absorbs which letter) are
    hard to fix once the chain has cooled.  Later restarts stay random.
    """
    from .cipher import NO_CHAR, random_expansion_key

    rng = np.random.default_rng(seed)
    t0 = time.time()

    global_best_key: np.ndarray | None = None
    global_best_score = -math.inf
    history: list[tuple[int, float]] = []
    iters_done = 0
    restarts_done = 0
    stopped = False

    movable = n_tokens - 1  # space (last index) is pinned
    cooling = (t_end / t_start) ** (1.0 / max(iterations - 1, 1))

    for restart in range(restarts):
        if restart == 0 and init_key is not None:
            key = init_key.copy()
        else:
            key = random_expansion_key(n_tokens, rng)
        score = scorer.score(key)
        best_key = key.copy()
        best_score = score
        temp = t_start

        for it in range(iterations):
            r = rng.random()
            if r < 0.35:
                tok = int(rng.integers(0, movable))
                old = int(key[tok, 0])
                new = int(rng.integers(0, A - 2))
                if new >= old:
                    new += 1
                key[tok, 0] = new
                undo = (tok, 0, old)
            elif r < 0.65:
                tok = int(rng.integers(0, movable))
                old = int(key[tok, 1])
                key[tok, 1] = int(rng.integers(0, A - 1))
                undo = (tok, 1, old)
            elif r < 0.80:
                tok = int(rng.integers(0, movable))
                old = int(key[tok, 1])
                key[tok, 1] = NO_CHAR
                undo = (tok, 1, old)
            else:
                a, b = rng.choice(movable, size=2, replace=False)
                tmp = key[a].copy()
                key[a] = key[b]
                key[b] = tmp
                undo = ("swap", int(a), int(b))

            new_score = scorer.score(key)
            delta = new_score - score
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                score = new_score
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
            else:
                if undo[0] == "swap":
                    _, a_, b_ = undo
                    tmp = key[a_].copy()
                    key[a_] = key[b_]
                    key[b_] = tmp
                else:
                    tok_, col, old_ = undo
                    key[tok_, col] = old_

            temp *= cooling
            iters_done += 1

            if (it + 1) % progress_every == 0 or it == iterations - 1:
                history.append((restart * iterations + it + 1, best_score))
                if progress is not None:
                    progress(
                        {
                            "restart": restart,
                            "restarts": restarts,
                            "iteration": it + 1,
                            "total_iterations": iterations,
                            "temperature": temp,
                            "score": score,
                            "best_score": max(best_score, global_best_score),
                        }
                    )
                if should_stop is not None and should_stop():
                    stopped = True
                    break

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_key = best_key
        restarts_done = restart + 1
        if stopped:
            break

    if global_best_key is not None and not stopped:
        if progress is not None:
            progress(
                {
                    "restart": restarts_done - 1,
                    "restarts": restarts,
                    "iteration": iterations,
                    "total_iterations": iterations,
                    "temperature": t_end,
                    "score": global_best_score,
                    "best_score": global_best_score,
                    "phase": "polishing",
                }
            )
        global_best_key, global_best_score = polish_expansion(
            scorer, global_best_key, should_stop=should_stop
        )
        history.append((restarts_done * iterations, global_best_score))

    return AnnealResult(
        best_key=global_best_key,
        best_score=global_best_score,
        history=history,
        iterations_done=iters_done,
        restarts_done=restarts_done,
        stopped_early=stopped,
        elapsed_sec=time.time() - t0,
    )


def anneal_anagram(
    scorer,
    n_tokens: int,
    iterations: int = 20_000,
    restarts: int = 3,
    t_start: float = 0.15,
    t_end: float = 0.002,
    p_swap: float = 0.5,
    seed: int | None = None,
    init_key: np.ndarray | None = None,
    locked_letters: np.ndarray | None = None,
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_every: int = 500,
) -> AnnealResult:
    """Simulated annealing for the anagram hypothesis: a flat key (one
    letter per token, no space entry) scored at the WORD level by an
    incremental AnagramScorer.  Supports crib locking via
    `locked_letters` (length n_tokens, -1 = free)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()

    global_best_key: np.ndarray | None = None
    global_best_score = -math.inf
    history: list[tuple[int, float]] = []
    iters_done = 0
    restarts_done = 0
    stopped = False

    movable_idx = np.arange(n_tokens)
    if locked_letters is not None:
        movable_idx = movable_idx[locked_letters < 0]
    if len(movable_idx) < 2:
        raise ValueError("fewer than two unlocked tokens — nothing to search")
    cooling = (t_end / t_start) ** (1.0 / max(iterations - 1, 1))

    for restart in range(restarts):
        if restart == 0 and init_key is not None:
            key = init_key.copy()
        else:
            key = rng.integers(0, A - 1, size=n_tokens, dtype=np.int64)
        if locked_letters is not None:
            pinned = locked_letters >= 0
            key[pinned] = locked_letters[pinned]

        score = scorer.reset(key)
        best_key = key.copy()
        best_score = score
        temp = t_start

        for it in range(iterations):
            if rng.random() < p_swap:
                a, b = (int(x) for x in rng.choice(movable_idx, size=2, replace=False))
                key[a], key[b] = key[b], key[a]
                changed = [a, b]
                undo_key = ("swap", a, b)
            else:
                tok = int(movable_idx[rng.integers(0, len(movable_idx))])
                old = int(key[tok])
                new = int(rng.integers(0, A - 2))
                if new >= old:
                    new += 1
                key[tok] = new
                changed = [tok]
                undo_key = ("set", tok, old)

            undo_score = scorer.update(key, changed)
            new_score = scorer.objective()
            delta = new_score - score
            if delta >= 0 or rng.random() < math.exp(delta / temp):
                score = new_score
                if score > best_score:
                    best_score = score
                    best_key = key.copy()
            else:
                scorer.revert(undo_score)
                if undo_key[0] == "swap":
                    _, a, b = undo_key
                    key[a], key[b] = key[b], key[a]
                else:
                    _, tok, old = undo_key
                    key[tok] = old

            temp *= cooling
            iters_done += 1

            if (it + 1) % progress_every == 0 or it == iterations - 1:
                history.append((restart * iterations + it + 1, best_score))
                if progress is not None:
                    progress(
                        {
                            "restart": restart,
                            "restarts": restarts,
                            "iteration": it + 1,
                            "total_iterations": iterations,
                            "temperature": temp,
                            "score": score,
                            "best_score": max(best_score, global_best_score),
                        }
                    )
                if should_stop is not None and should_stop():
                    stopped = True
                    break

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_key = best_key
        restarts_done = restart + 1
        if stopped:
            break

    if global_best_key is not None and not stopped:
        global_best_key, global_best_score = polish_anagram(
            scorer, global_best_key, movable_idx, should_stop=should_stop
        )
        history.append((restarts_done * iterations, global_best_score))

    return AnnealResult(
        best_key=global_best_key,
        best_score=global_best_score,
        history=history,
        iterations_done=iters_done,
        restarts_done=restarts_done,
        stopped_early=stopped,
        elapsed_sec=time.time() - t0,
    )


def polish_anagram(
    scorer,
    key: np.ndarray,
    movable_idx: np.ndarray,
    max_passes: int = 4,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy coordinate descent for anagram keys: every movable token
    tries all 26 letters, keeping improvements, until a full pass changes
    nothing.  Uses the scorer's incremental updates."""
    key = key.copy()
    best = scorer.reset(key)

    for _ in range(max_passes):
        improved = False
        for tok in movable_idx:
            if should_stop is not None and should_stop():
                return key, best
            tok = int(tok)
            # Invariant: scorer always reflects `key`; rejected candidates
            # are rolled back immediately, accepted ones become the new
            # baseline for the remaining candidates.
            best_c = int(key[tok])
            for c in range(A - 1):
                if c == best_c:
                    continue
                key[tok] = c
                undo = scorer.update(key, [tok])
                s = scorer.objective()
                if s > best:
                    best, best_c, improved = s, c, True
                else:
                    scorer.revert(undo)
                    key[tok] = best_c
        if not improved:
            break
    return key, best


def polish_expansion(
    scorer,
    key: np.ndarray,
    max_passes: int = 4,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, float]:
    """Greedy coordinate descent on an expansion key: for each token try
    every first letter, then every second-letter option (including none),
    keeping improvements.  Deterministic finisher after annealing."""
    from .cipher import NO_CHAR

    key = key.copy()
    best = scorer.score(key)
    n_tokens = key.shape[0]
    second_options = [NO_CHAR] + list(range(A - 1))

    for _ in range(max_passes):
        improved = False
        for tok in range(n_tokens - 1):  # space (last index) stays pinned
            if should_stop is not None and should_stop():
                return key, best
            for col, options in ((0, range(A - 1)), (1, second_options)):
                orig = int(key[tok, col])
                best_c = orig
                for c in options:
                    if c == orig:
                        continue
                    key[tok, col] = c
                    s = scorer.score(key)
                    if s > best:
                        best, best_c, improved = s, c, True
                key[tok, col] = best_c
        if not improved:
            break
    return key, best


def random_expansion_scores(
    scorer,
    n_tokens: int,
    n_samples: int = 30,
    seed: int | None = None,
) -> list[float]:
    """Chance floor for expansion-key searches (plain single-letter random
    keys, so floors are comparable across hypotheses)."""
    from .cipher import random_expansion_key

    rng = np.random.default_rng(seed)
    return [
        scorer.score(random_expansion_key(n_tokens, rng))
        for _ in range(n_samples)
    ]


def random_key_scores(
    view: NgramView,
    logp: np.ndarray,
    n_states: int,
    n_tokens: int,
    n_samples: int = 30,
    seed: int | None = None,
) -> list[float]:
    """Scores of random keys — the chance floor a real solution must beat."""
    rng = np.random.default_rng(seed)
    return [
        view.score(random_key(n_states, n_tokens, rng), logp)
        for _ in range(n_samples)
    ]
