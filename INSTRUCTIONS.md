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

Downloads into `data/` (~10 MB total):

| File | Contents |
|------|----------|
| `interlinear_full_words.txt` | word-level Voynich transcription with folio/section/Currier/line metadata |
| `reference_latin.txt` | De Imitatione Christi (~1420s) + De Bello Gallico, cleaned |
| `reference_italian.txt` | La Divina Commedia, cleaned |
| `reference_english.txt` | King James Bible, cleaned |

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

# Known-answer validation
python -m voynich benchmark --reference latin
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

## 6. Tests

```bash
python -m pytest tests/
```

Runs offline. `tests/test_annealer.py` is the load-bearing one: the solver
must recover ≥95% of a synthetic cipher.

## Legacy

* `experiment/` — v1 mod-23 statistical experiment (unchanged, still tested).
* `legacy/v2/` — the previous positional-decoder scripts this workbench replaced.
