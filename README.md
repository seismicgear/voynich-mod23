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
   letters of a target language: one global key (*simple substitution*),
   three position-dependent keys (*positional*), or expansions of one token
   to 1–2 letters (*abbreviation* — the scribal-abbreviation theory, the
   leading family of explanations for Voynichese's anomalously low entropy).
   Any of them can also be scored against consonant skeletons (*abjad* —
   the vowels-unwritten hypothesis behind the best published result).
2. **Candidates are scored honestly** — by the log-probability of the
   decoded text under a smoothed character n-gram model trained on a real
   corpus. The Latin reference is built on Thomas à Kempis' *De Imitatione
   Christi* (~1420s), contemporary with the manuscript's 1404–1438 carbon
   dating; Froissart's *Chroniques* (Middle French) and Leizarraga's Basque
   New Testament anchor other families to the right era where possible.
3. **The search is validated** — the benchmark mode encrypts held-out
   reference text with a random substitution (or a synthetic abbreviation
   cipher) and checks the annealer recovers it: **100% letter accuracy in
   seconds** for substitution, 75–93% for the harder abbreviation case. So
   when the same machinery fails on Voynichese, that is evidence about the
   text, not about the solver.
4. **Results are anchored** — every run reports its held-out score between a
   *random-key floor* (chance) and a *reference-language ceiling* (what real
   text scores), as "gap closed". Training and validation use disjoint
   alternating lines, so the optimizer cannot simply memorize.

The reproducible finding — consistent with the published literature — is
that no substitution-family decoding into any of the 16 reference languages
turns Voynichese into language, even though its rigid word structure lets an
optimizer close much of the statistical gap. Negative results are results;
[ANALYSIS.md](ANALYSIS.md) lays out what they imply and what would actually
settle the question.

## Reference languages

Sixteen corpora across the language families plausibly present in
15th-century Europe, each labelled with its period in the GUI:

| Family | References |
|--------|------------|
| Italic | Latin — *De Imitatione Christi* (~1420s) + *De Bello Gallico* |
| Romance | Italian — Dante (c. 1320) · Middle French — Froissart, *Chroniques* (c. 1370–1400) · Spanish — *Don Quijote* (1605) · Portuguese — *Os Lusíadas* (1572) · Catalan (19th c. literary) |
| Germanic | English — KJV (1611) · German — Luther Bible tradition · Dutch — Bible |
| Slavic | Czech — *Bible kralická* (1613) · Polish — *Pan Tadeusz* (1834) · Russian — Synodal Bible (1876) |
| Hellenic | Greek — Koine New Testament (the Byzantine standard) |
| Uralic | Hungarian — Jókai (1872) · Finnish — *Kalevala* (1849) |
| Isolate | Basque — Leizarraga New Testament (1571) |

Greek and Cyrillic are transliterated **one letter per letter** (θ→q, ч→q;
tables in `corpus.py`) — multi-letter romanizations like θ→"th" would
smuggle fake digraph statistics into the language models.

**Sweep mode** runs one configuration against all 16 references and ranks
them by gap closed — the one-click answer to "which language fits best, and
does any of them actually fit?".

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
python -m voynich solve --reference latin --hypothesis abbreviation --abjad
python -m voynich sweep --language A --hypothesis simple        # all 16 languages, ranked
python -m voynich benchmark --reference latin --mode abbreviation
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

## Baseline results: the 16-language sweep

`python -m voynich sweep --language A --seed 42` (simple substitution,
40,000 iterations × 2 restarts, order 4). *Gap closed* locates the held-out
score between the random-key floor (0%) and the real-language ceiling
(100%); the decoded samples are held-out lines the optimizer never saw.

| #  | Reference   | Gap closed | Decoded sample (held-out)      |
|----|-------------|-----------:|--------------------------------|
| 1  | Portuguese  | 82.6%      | `oora do dios aodo e do o dos` |
| 2  | Russian     | 82.5%      | `aami na naai iana i na a nam` |
| 3  | Catalan     | 82.0%      | `aara la llas sala a la a las` |
| 4  | Spanish     | 81.5%      | `eelo la lees aela a la e los` |
| 5  | Italian     | 80.8%      | `iino si sain aise e si i min` |
| 6  | Finnish     | 80.0%      | `aana sa saan aasa a sa a san` |
| 7  | Czech       | 79.5%      | `eeli se stes sese z se e sel` |
| 8  | Dutch       | 79.1%      | `rren en eere enen u en n erd` |
| 9  | Greek       | 75.3%      | `nnos en eina inen h en n ena` |
| 10 | Middle French | 74.6%    | `iins si suis sisi a si i ses` |
| 11 | Polish      | 74.0%      | `aada na niaz lana z na a nal` |
| 12 | Latin       | 73.9%      | `eeum te taet seti a te e tem` |
| 13 | Hungarian   | 73.4%      | `eere te ties tete s te e tet` |
| 14 | German      | 72.9%      | `ssar an ansa anan s an n ast` |
| 15 | Basque      | 69.8%      | `aari da duan cada o da a den` |
| 16 | English     | 68.9%      | `ssed an ansa esat i an s are` |

Three readings of this table:

* **No language works.** Nothing crosses 85%, and every sample is
  repetitive non-language. For calibration, the same solver at a smaller
  budget recovers 100% of a genuine substitution cipher.
* **The ranking itself is informative.** Vowel-rich, low-entropy
  orthographies (Portuguese, Russian transliteration, Catalan, Spanish)
  costume best — exactly what entropy arithmetic predicts, since
  Voynichese's conditional entropy is far below any European language's.
* **Richer hypotheses close more gap without becoming language.** Under
  the *abbreviation* hypothesis (token → 1–2 letters, 50k × 3), Portuguese
  reaches **86.2%** — past the naive threshold — while its sample
  (`aara do dias sado e do a das`) remains unreadable. That is why the
  verdict demands a parseable sample *and* reproducibility, not a score.
  See [ANALYSIS.md](ANALYSIS.md) for the full argument and the road map
  beyond substitution ciphers.

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
