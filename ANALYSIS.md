# If none of these languages work — what would actually solve it?

This document is the thinking behind the workbench's design, and the road
map it implies. It assumes the results you can reproduce with one click:
the solver provably cracks real substitution and abbreviation ciphers in
seconds, and no reference language — Romance, Germanic, Slavic, Greek,
Uralic, or Basque — turns Voynichese into language under substitution-family
keys. That negative is not a software limitation. Here is why, and what
follows from it.

## 1. Why the language sweep was always going to fail

The decisive obstacle is **conditional character entropy**. Voynichese's
second-order entropy is ~2.1–2.3 bits/char; every European language in this
workbench sits near 3–4. A monoalphabetic substitution — *any*
monoalphabetic substitution, in any alphabet — preserves entropy exactly.
So no relabeling of EVA glyphs can make Voynichese statistically resemble
plain Latin, Czech, Greek, or Basque. The sweep quantifies this per
language (as "gap closed"), and the per-language ranking is still
informative — it tells you *which* languages' letter statistics Voynichese
can best be costumed as — but the costume never becomes the language: the
decoded samples stay unreadable in all 16 cases.

This is also why the sweep's 69–83% gap-closure numbers must not be
over-read. Voynichese's word structure is rigid enough that an optimizer can
match a large share of any language's n-gram statistics — and the *ranking*
confirms the entropy story: vowel-rich, low-entropy orthographies
(Portuguese 82.6%, transliterated Russian 82.5%, Catalan 82.0%) costume
best, while English (68.9%) and Basque (69.8%) fit worst. Give the
optimizer a richer key family and it climbs further without getting truer:
the abbreviation hypothesis pushes Portuguese to 86.2% with a sample
(`aara do dias sado e do a das`) that no Portuguese reader can parse. The
benchmark mode exists precisely to calibrate this: on a *real* cipher the
same machinery closes essentially the whole gap and the output reads as
text.

## 2. The hypothesis ladder

If the text encodes language at all, the encoding must *change* entropy.
Ranked by (historical plausibility × testability), with implementation
status in this workbench:

| # | Hypothesis | Entropy direction | Status |
|---|-----------|-------------------|--------|
| 1 | **Verbose cipher** — glyph *groups* encode single letters | lowers ciphertext entropy ✓ (matches Voynichese) | testable now: raise `bpe_merges` so tokens become whole glyph groups, then sweep |
| 2 | **Scribal abbreviation** — single signs encode letter *groups* (15th-c. Latin practice, Cappelli's lexicon) | raises decoded information per token | implemented: the `abbreviation` hypothesis (token → 1–2 letters, staged init + polish) |
| 3 | **Abjad** — vowels unwritten (Hebrew/Arabic convention; basis of Hauer & Kondrak's 2016 Hebrew result) | removes the most predictable letters | implemented: the `abjad` flag scores against consonant-skeleton corpora |
| 4 | **Positional/state-dependent alphabets** | mild | implemented (`positional`); extendable to word-position states |
| 5 | **Nulls + nomenclator** — some glyphs decode to nothing, some tokens are whole-word codes | lowers | designed: allow zero-length expansions (needs a minimum-output guard so the optimizer can't delete the text) plus a code-token dictionary |
| 6 | **Per-word anagramming** — letters written in scrambled/canonical order | n-grams uninformative; needs word-level scoring | implemented: the `anagram` hypothesis — words scored as letter multisets against a reference dictionary, with an *alphagram LM* (n-grams over sorted-letter words) supplying the smooth gradient that pure dictionary matching lacks; validated at 97% recovery on synthetic anagram ciphers |
| 7 | **Unknown language / invented script** | — | partially testable: typological comparison (entropy, word grammar) against diverse families — Uralic and Basque are in the sweep for exactly this reason |

Two further mechanisms automate the grind:

* **Crib locking** (`lock_rounds`): after each solve, tokens corroborated by
  enough dictionary-matched words (length ≥ 3, ≥ 60% of their occurrences
  matched, ≥ 20 occurrences of support) are frozen and the rest re-anneal.
  This is the codebreaker's bootstrap, automated — with thresholds, because
  locking junk early poisons everything downstream.
* **Brute force is not an option, and that's a theorem, not a shortcut.**
  With ~55 tokens and 26 letters the key space exceeds 26^55 ≈ 10^77.
  "Try every combination" is what the annealer *approximates*: restarts
  explore basins, Metropolis acceptance escapes local traps, locking
  preserves what's already won.

### A warning from the anagram experiments: objectives can be gamed

Anagram (multiset) scoring is intrinsically looser than ordered scoring —
many letter bags collide with *some* dictionary word. On the real
manuscript the anagram optimizer promptly scored **above the
real-language ceiling** (gap "closed" >100%) by mapping Voynich's rigid
short words onto frequent short dictionary words. The workbench now flags
this verdict explicitly and reports **long-word match rates** (≥5 letters)
separately, where multiset collisions are rare. The next methodological
upgrade on this front is a **null control**: run the identical search on a
token-shuffled pseudo-Voynich and report the score *difference* — only
signal that survives the control counts.

Combinations matter more than single rungs: the historically attested
candidate for a 15th-century herbal is **abbreviated Latin** (#2 + #3 +
nulls), and the workbench can now run #2 and #3 jointly
(`--hypothesis abbreviation --abjad`).

## 3. The non-language branch — and the experiment that would settle it

The current scientific frontrunner (Timm & Schinner 2020; consistent with
Bowern & Lindemann's survey) is that the text is **generated, not
encoded**: a scribe producing each new word by copying and mutating earlier
words ("self-citation"). This explains, with very few parameters, the
observations that cipher hypotheses struggle with:

* quasi-reduplication (`qokeedy qokeedy qokedy` sequences far above chance),
* the rigid in-word glyph grammar,
* line-position effects (line as a functional unit),
* the Currier A/B continuum (drift in copying habits, two scribal cohorts),
* low entropy *with* Zipf-like word frequencies.

**The decisive experiment** — the one this workbench should grow toward —
is model comparison on held-out folios:

> Fit (a) the best language+cipher model the hypothesis ladder allows and
> (b) a self-citation generator, each on half the manuscript. Score both on
> the held-out half (per-token log-likelihood; penalize parameters, e.g.
> BIC). If the generator — with ~10 parameters — beats every cipher family
> with hundreds, the "meaningful text" hypothesis loses on evidence, not
> fashion.

The infrastructure for (a) is this repository. The missing piece for (b)
is a parameterized autocopy model (copy-distance distribution + per-glyph
mutation rates) and a shared likelihood harness — the natural next
milestone.

## 4. What "solved" would have to look like

A claimed solution should be accepted only if **all four** hold:

1. **Statistical**: >85% gap closed on *held-out* folios (not the training
   half), reproducible across seeds and across transcribers (the loader
   supports H/C/F/... transcriptions for exactly this check).
2. **Linguistic**: the decoded held-out sample parses for a fluent reader
   of the claimed language — not "with some imagination", but parses.
3. **Topical**: decoded vocabulary tracks the manuscript's sections (plant
   terms in Herbal, star terms in Astronomical). The loader exposes the
   section codes; a topical-coherence metric is a small, well-defined
   addition.
4. **Procedural**: the key/procedure is simple enough that a 15th-century
   scribe could write *and read* it. (Many published "solutions" fail only
   here — they are decoding procedures so lossy that any text can be
   extracted from any source.)

Every published claim so far fails at least two of these. A tool that makes
all four checkable at a button press is more useful to the field than
another claimed decipherment.

## 5. Honest accounting of this workbench's limits

* Reference corpora are proxies: Bible translations and later literature
  stand in for 15th-century vernaculars where no clean medieval corpus is
  downloadable (periods are labelled in the GUI). Genre and era shift
  letter statistics by a few percent of gap — enough to reorder mid-table
  sweep entries, not enough to turn a 75% into a 95%.
* Greek and Cyrillic are transliterated one-letter-per-letter (θ→q, ч→q
  conventions documented in `corpus.py`); merged Cyrillic vowels (ю→u, я→a)
  slightly flatten Russian statistics.
* The abbreviation search recovers 75–93% of synthetic abbreviation
  ciphers, not 100% — segmentation errors are coupled, and the residual is
  honest difficulty, not a bug. Budgets matter: prefer ≥50k iterations ×3
  restarts for real runs.
* Scores are character-level. Word-level scoring (dictionary models) is the
  single highest-value addition on the cipher branch (ladder #6).
