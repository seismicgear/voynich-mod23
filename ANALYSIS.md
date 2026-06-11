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
| 5 | **Nulls** — some glyphs decode to nothing (standard camouflage in Tranchedino's quattrocento cipher ledger) | lowers | implemented: `allow_nulls` on the abbreviation hypothesis. Nulls pay rent (3 bits per nulled occurrence) plus an output floor — without the rent the optimizer "solves" the manuscript by deleting it. Validated: 100% recovery on synthetic null ciphers, nulls correctly identified. The nomenclator half (code-tokens for whole words) remains designed |
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

A first instrument on this branch now exists: the **diagnostics mode**
measures the self-citation signature directly (how often a content word
has a near-duplicate among the previous N words, against an
order-shuffled null and against reference languages). Early honest
readings: with content words ≥4 glyphs and a 15-word window, Currier A
shows a locality excess of ~8 points — double medieval Latin (~4) but
*below* the King James Bible (~13), whose verse formulas are their own
kind of self-citation. The instrument reports this without spin; sharper
separation likely needs Timm's full graded-similarity measure and
paragraph-level windows, which is exactly what the autocopy-likelihood
milestone would deliver.

Two further controls round out the methodology:

* **Reading-order modes** (`reverse: words|lines`) test the
  mirror-writing / right-to-left family of theories in one flag.
* **The shuffled-text control** (`control: true`) reruns any
  configuration on token-scrambled pseudo-Voynichese with identical word
  lengths and line structure. Any "signal" that survives the scramble is
  a property of the objective, not the manuscript — this is the
  calibration the anagram episode (§2) showed we needed.

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

## 5. Specific theories and how this workbench tests them

### The Byblos-script / Italo-Celtic theory

One circulating theory holds that the manuscript is written in a
descendant of the **Byblos syllabary** (a Bronze Age pseudo-hieroglyphic
script from Lebanon) and encodes an **archaic Italo-Celtic language**
(possibly Venetic, with Gaulish/Greek/Germanic loans). Honest assessment
before testing:

* The Byblos syllabary is *itself undeciphered* (its corpus is roughly a
  dozen inscriptions). Reading one undeciphered script through another
  multiplies unknowns rather than dividing them — no statistical test can
  validate the chain end to end.
* There is no Venetic or Gaulish corpus large enough to train a language
  model on (a few hundred short inscriptions each).

What *can* be tested, and now is:

* **The syllabary half**: a CV-syllabary reading means each glyph token
  decodes to a consonant+vowel pair — which is exactly the
  `abbreviation` hypothesis (token → 1–2 letters). A Byblos-style reading
  of Voynichese into any reference language is a run of
  `--hypothesis abbreviation` against it; the machinery recovers
  synthetic syllabary-style ciphers at 75–93%.
* **The Italo-Celtic half**: the nearest living relatives with usable
  corpora are now in the registry — **Welsh** and **Irish** on the Celtic
  side, Latin/Italian on the Italic side. If Voynichese had Italo-Celtic
  morphology underneath a substitution-family encoding, the Celtic and
  Italic references would rank far above unrelated controls (Somali,
  Chinese, Basque) in the sweep. That separation is the testable claim;
  run it and read the table.

### The trade-network argument (it was found near Rome…)

…and Rome's trade world reached from Iceland to China, so the registry
now covers it: **Arabic** (Mediterranean lingua franca), **Hebrew**
(Hauer & Kondrak's published best candidate), **Persian** (Silk Road),
**Turkish** (the Ardıç family's Old Turkic theory), **Amharic** (an
Ethiopian embassy reached Rome for the Council of Florence in 1441 —
inside the carbon window), **Somali** (Indian Ocean coast), **Hindi**
(Indian Ocean trade), **Chinese** (as toneless pinyin syllables — Stolfi's
classic observation that Voynichese word grammar resembles Mandarin
syllable structure operates exactly at this level), plus **Welsh, Irish,
Icelandic, Albanian** for the European periphery. Arabic, Hebrew, Persian,
Amharic and Hindi are transliterated one-sign-per-sign (tables and
Unicode-name romanization in `corpus.py`); Arabic-script and Hebrew
references pair naturally with the `abjad` flag since those scripts
already omit most vowels.

## 6. Polymaths and invented scripts: the historical company the manuscript keeps

The question "did anyone of that era make things like this?" has a firm
answer: yes, and the parallels are instructive.

* **Giovanni Fontana (c. 1395–1455)** — Paduan-trained physician-engineer
  in Venetian service. His *Bellicorum instrumentorum liber* (1420s, now
  in Munich) and *Secretum de thesauro* are **book-length manuscripts
  written in an invented cipher script of letterless signs**, with
  copious technical drawings — the same decade, the same Veneto, and the
  same "entire enciphered book" format as the Voynich. Crucially,
  Fontana's cipher is a **simple substitution** and was read easily once
  examined. The Voynich resists exactly the attack that cracked Fontana —
  quantitatively, by every run in this workbench — which is the cleanest
  single argument that it is not a plain substitution cipher.
* **Hildegard of Bingen (1098–1179)** — her *Litterae ignotae* (invented
  alphabet) and *Lingua ignota* (constructed vocabulary of ~1,000 words,
  mostly for plants, medicine and the divine) prove the medieval
  "invented script + invented lexicon for a herbal-medical corpus" genre
  existed centuries before the Voynich.
* **Leon Battista Alberti (1404–1472)** — *De cifris* (1467), the cipher
  disk, and the first European description of **polyalphabetic**
  encipherment; the `positional` hypothesis tests the simplest
  state-dependent family on Voynichese.
* **Francesco Tranchedino (Milan, c. 1450s)** — his surviving notebook of
  diplomatic ciphers documents the *actual* key families of the
  quattrocento chancery: substitutions extended with **nulls, homophones
  and nomenclator code-words**. That is precisely rung #5 of the
  hypothesis ladder, and the strongest historical argument for
  implementing it next.
* **Ramon Llull (1232–1316)** and the Kabbalist **Abraham Abulafia
  (1240–c. 1291)** — combinatorial letter-permutation arts. If the
  manuscript is a *generated* artifact (the self-citation branch), this
  is the intellectual tradition a generator would have come from; Gordon
  Rugg's later grille argument is the same idea mechanized.
* **Johannes Trithemius (1462–1516)** and **Athanasius Kircher
  (1602–1680)** bracket the story: Trithemius founded systematic
  steganography just after the manuscript's creation; Kircher — the era's
  most famous (and famously overconfident) decipherer — is the very
  person the manuscript was sent to in 1665, and he never published a
  reading.

The pattern across all of them: every *genuine* enciphered book of the
period that survives (Fontana's above all) fell to frequency analysis
the moment someone competent tried, because real working ciphers of that
century were substitution-family. The Voynich's resistance to the entire
substitution family — now demonstrated across 28 languages and four key
families in this workbench — is the strongest internal evidence that it
is either something rarer (verbose/nomenclator, which Tranchedino's
notebook shows existed) or not a cipher at all.

## 7. Honest accounting of this workbench's limits

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
