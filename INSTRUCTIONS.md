# Instructions (v3 — Decipherment Workbench)

## 1. Install

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

## 2. Get the data

```bash
python -m voynich setup
```

Downloads into `data/` (~35 MB total): the word-level Voynich transcription
(folio/section/Currier/line metadata) and 16 cleaned reference corpora
spanning Italic, Romance, Germanic, Slavic, Hellenic, Uralic and Basque —
see the table in [README.md](README.md). Greek and Cyrillic sources are
transliterated one letter per letter during cleaning.

You can also do this from the GUI's **Data** tab.

## 3. Use the GUI (recommended)

```bash
python -m voynich gui          # http://127.0.0.1:5000/
```

* **Validate solver** first: run a benchmark to see the annealer recover a
  known substitution cipher (expect ~100% letter recovery in seconds).
* **Solve**: pick Currier language, reference, hypothesis and budget, then
  start. The run streams its annealing curve and finishes with a report.
* Click any entry under **Runs** to revisit it; reports are also saved as
  JSON under `results/`.

## 4. Or use the CLI

```bash
# A decipherment attempt: Currier A vs. medieval Latin, simple substitution
python -m voynich solve --language A --reference latin --hypothesis simple

# The positional (line-start/body/line-end) hypothesis on Currier B
python -m voynich solve --language B --hypothesis positional --iterations 100000

# Scribal abbreviation (token -> 1-2 letters), vowels unwritten
python -m voynich solve --reference latin --hypothesis abbreviation --abjad

# Anagram hypothesis (letters unordered inside words) with crib locking:
# tokens supported by dictionary-matched words freeze between rounds
python -m voynich solve --reference latin --hypothesis anagram --lock-rounds 3

# Rank ALL 16 reference languages on one configuration
python -m voynich sweep --language A --hypothesis simple --iterations 40000

# Known-answer validation (substitution, abbreviation, anagram ciphers)
python -m voynich benchmark --reference latin
python -m voynich benchmark --reference latin --mode anagram
```

## 5. Reading a report

Every solve reports four numbers (all in mean bits/char under the reference
language model; higher = more language-like):

| Number | Meaning |
|--------|---------|
| `random-key floor` | what a random key scores — chance level |
| `held-out score` | the best key applied to lines the optimizer never saw |
| `language ceiling` | what real reference-language text scores |
| `gap closed` | where the held-out score sits between floor (0%) and ceiling (100%) |

Interpretation, bluntly:

* **Gap closed near 100%** *and* the decoded sample reads as language →
  a serious candidate. Re-run with other seeds and sections before believing it.
* **50–85%** → the optimizer is exploiting Voynichese's rigid word structure;
  the sample will look like `tee te e rua are te...`. This is the expected
  outcome for substitution hypotheses and is a meaningful *negative* result.
* **< 50%** → the hypothesis/reference pairing does worse than structure
  alone usually allows; the text does not behave like a substitution of that
  language at all.

A high train score with a much lower held-out score means the key memorized
the training lines — the split exists precisely to expose that.

Every report also carries a **dictionary word-match rate** for the decoded
held-out text, split by word length. Short words (3–4 letters) match cheaply;
**long words (≥5) are the hard currency** — real decipherments produce them,
gamed objectives don't. If a run reports gap closed **above 100%** (possible
under anagram scoring), the optimizer beat the objective, not the manuscript;
the verdict text explains this whenever it happens.

## 6. Tests

```bash
python -m pytest tests/
```

Runs offline. `tests/test_annealer.py` is the load-bearing one: the solver
must recover ≥95% of a synthetic cipher.

## Legacy

* `experiment/` — v1 mod-23 statistical experiment (unchanged, still tested).
* `legacy/v2/` — the previous positional-decoder scripts this workbench replaced.
