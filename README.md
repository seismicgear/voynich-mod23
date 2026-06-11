# Voynich Decipherment Workbench

A computational workbench for testing decipherment hypotheses against the
Voynich Manuscript (Beinecke MS 408): MCMC simulated annealing over
substitution-key spaces, scored by reference-language n-gram models, with a
browser GUI, held-out validation, and a built-in solver benchmark on ciphers
with known answers.

```bash
pip install -r requirements.txt
python -m voynich setup     # download the transcription + reference corpora
python -m voynich gui       # open http://127.0.0.1:5000/
```

---

## What this can and cannot claim

The Voynich Manuscript is undeciphered. **No software can promise to solve
it**, and this one doesn't. What it does is make decipherment hypotheses
*testable and falsifiable*:

1. **A hypothesis is a key family** — a mapping from EVA glyph tokens to
   letters of a target language, either one global key (*simple
   substitution*) or three position-dependent keys (*line-start / body /
   line-end*).
2. **Candidates are scored honestly** — by the mean log-probability of the
   decoded text under a smoothed character n-gram model trained on a real
   corpus. The Latin reference is built on Thomas à Kempis' *De Imitatione
   Christi* (~1420s), contemporary with the manuscript's 1404–1438 carbon
   dating.
3. **The search is validated** — the benchmark mode encrypts held-out
   reference text with a random substitution and checks the annealer recovers
   it. It does, routinely at **100% letter accuracy in seconds**. So when the
   same machinery fails on Voynichese, that is evidence about the text, not
   about the solver.
4. **Results are anchored** — every run reports its held-out score between a
   *random-key floor* (chance) and a *reference-language ceiling* (what real
   text scores), as "gap closed". Training and validation use disjoint
   alternating lines, so the optimizer cannot simply memorize.

The reproducible finding — consistent with the published literature — is that
no simple or positional substitution into Latin, Italian, or English turns
Voynichese into language, even though its rigid word structure lets an
optimizer close much of the statistical gap. Negative results are results.

---

## The GUI

`python -m voynich gui` serves a single-page workbench:

* **Solve** — configure a run (Currier language A/B, manuscript section,
  reference language, hypothesis, n-gram order, BPE merges, iterations,
  restarts, seed) and watch it live: annealing curve, temperature, best
  score, progress.
* **Validate solver** — run the known-answer benchmark and see decoded
  output vs. ground truth.
* **Data** — download / check the datasets.
* **Runs** — every run (running, done, stopped, failed) with full reports:
  score anchors, verdict, the decoded held-out sample by folio/line, and the
  complete best key. Runs can be stopped mid-flight; reports are saved to
  `results/`.

## The CLI

```bash
python -m voynich solve --language A --reference latin --hypothesis simple \
                        --iterations 60000 --restarts 3
python -m voynich benchmark --reference latin --cipher-chars 4000
python -m voynich setup --force
```

---

## How it works

```
interlinear transcription (word-level, Takahashi line, folio/section/Currier metadata)
        │ parse + filter                       reference corpus (Latin/Italian/English)
        ▼                                                   │ clean
EVA lines ──► tokenizer (core multigraphs + BPE) ──► token/state streams
        │                                                   ▼
        │ split: even lines = train, odd = held out   char n-gram LM (order 3–4,
        ▼                                              interpolated smoothing)
NgramView (compressed exact scoring)  ◄────────────────────┘
        │
        ▼
simulated annealing over keys  (swap + reassign moves, Metropolis acceptance,
        │                       geometric cooling, multiple restarts)
        ▼
report: held-out score vs floor/ceiling · gap closed · decoded sample · key
```

Implementation notes:

* **Scoring is exact and fast.** Every window of *n* consecutive
  (token, state) positions is compressed to a unique pattern with a count
  (`NgramView`), so re-scoring a mutated key is one vectorized gather over
  ~tens of thousands of patterns instead of a pass over the whole corpus —
  thousands of MCMC iterations per second.
* **Reassignment moves matter.** Good keys are many-to-one (the EVA token
  inventory is larger than any target alphabet); swap-only search cannot
  change letter multiplicities, so the proposal mixes swaps and single-token
  reassignments.
* **One transcriber.** The interlinear corpus interleaves up to 18
  transcribers per line; the loader pins to Takahashi ("H"), the most
  complete, so lines aren't silently duplicated (a bug in earlier versions
  of this repo).

## Baseline results

Full-budget runs (60,000 iterations × 3 restarts, n-gram order 4, seed 42)
on the real manuscript. Scores are mean bits/char under the reference LM;
*gap closed* locates the held-out score between the random-key floor (0%)
and the real-language ceiling (100%).

| Currier | Reference | Hypothesis  | Held-out | Floor   | Ceiling | Gap closed | Decoded sample (held-out) |
|---------|-----------|-------------|----------|---------|---------|------------|---------------------------|
| A       | Latin     | simple      | −4.07    | −10.08  | −2.30   | 77.2%      | `ttis et este atet a et t eta` |
| A       | Latin     | positional  | −4.07    | −10.18  | −2.30   | 77.5%      | `eere et este atet a et t tus` |
| B       | Latin     | simple      | −3.97    | −10.04  | −2.30   | 78.5%      | `et ste tet et tet set e tet e tt` |
| A       | Italian   | simple      | −3.86    | −10.89  | −2.24   | 81.2%      | `iino mi mail aima e mi i mio` |
| A       | English   | simple      | −4.93    | −11.44  | −2.02   | 69.1%      | `ssed an ansa esat i at s are` |

Read the samples: every configuration closes 70–80% of the statistical gap —
and none of them is language. The optimizer finds letter assignments that
mimic the reference language's n-gram statistics (note the convincing
Italian-ish `mi`/`mio` function words) because Voynichese's word structure is
rigid enough to support that, but the output never resolves into meaning. For
calibration, the same solver at the same budget recovers 100% of a genuine
substitution cipher. That contrast *is* the result.

## Layout

```
voynich/            the workbench package
  corpus.py         data download, cleaning, interlinear parsing
  tokenizer.py      EVA multigraphs + BPE
  lm.py             character n-gram language model
  cipher.py         key encoding, vectorized decoding, NgramView
  annealer.py       simulated annealing + restarts
  synthetic.py      known-answer cipher benchmark
  pipeline.py       end-to-end solve runs + reports
  webapp/           Flask GUI (no external JS/CSS dependencies)
tests/              pytest suite (offline; includes the solver benchmark)
experiment/         v1 statistical experiment (mod-23 hypothesis, kept as-is)
legacy/             superseded v1/v2 scripts and notes
```

## Tests

```bash
python -m pytest tests/
```

The suite runs offline (bundled fixtures) and includes the load-bearing test:
the annealer must recover ≥95% of a synthetic substitution cipher's letters.

---

## Data sources

* Voynich transcription: word-level interlinear corpus from
  [chirila/Voynich-public](https://github.com/chirila/Voynich-public)
  (Bowern lab), Takahashi transcription line.
* Latin: Thomas à Kempis, *De Imitatione Christi*; Caesar, *De Bello
  Gallico* (CLTK Latin Library mirror).
* Italian: Dante, *La Divina Commedia* (Project Gutenberg #1012).
* English: King James Bible (Project Gutenberg #10).
