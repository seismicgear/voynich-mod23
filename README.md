# Voynich Decipherment Workbench

A computational workbench for testing decipherment hypotheses against the
Voynich Manuscript (Beinecke MS 408): MCMC simulated annealing over
substitution-key spaces, scored by reference-language n-gram models, with a
browser GUI, held-out validation, and a built-in solver benchmark on ciphers
with known answers.

## Install

**Desktop (Linux / Windows / macOS)** — download the build for your OS from
the repository's GitHub Releases (produced by the `build-installers`
workflow: a `.tar.gz` binary for Linux, a `.dmg` for macOS, and both a
portable `.zip` and a `VoynichWorkbench-Setup.exe` installer for Windows).
Run it; the workbench opens in your browser. Data downloads on first use
into your user data directory.

**With pip** (any OS with Python 3.10+):

```bash
pip install .
voynich-workbench           # launches the GUI in your browser
```

**From source**:

```bash
pip install -r requirements.txt
python -m voynich setup     # download the transcription + reference corpora
python -m voynich gui       # open http://127.0.0.1:5000/
```

Maintainers: tag a release (`git tag v3.2.0 && git push --tags`) or run the
`build-installers` workflow manually — it builds, tests and smoke-runs the
executable on all three OSes and attaches the artifacts.

---

## What this can and cannot claim

The Voynich Manuscript is undeciphered. **No software can promise to solve
it**, and this one doesn't. What it does is make decipherment hypotheses
*testable and falsifiable*:

1. **A hypothesis is a key family** — a mapping from EVA glyph tokens to
   letters of a target language: one global key (*simple substitution*),
   three position-dependent keys (*positional*), expansions of one token
   to 1–2 letters (*abbreviation* — the scribal-abbreviation theory, the
   leading family of explanations for Voynichese's anomalously low entropy),
   or order-free letter bags scored against a dictionary (*anagram* — for
   the possibility that scribes scrambled letter order within words).
   Any of them can also be scored against consonant skeletons (*abjad* —
   the vowels-unwritten hypothesis behind the best published result).
   *Crib locking* (`lock_rounds`) automates the codebreaker's bootstrap:
   tokens corroborated by enough dictionary-matched words are frozen and
   the rest re-anneal, round after round.
2. **Candidates are scored honestly** — by the log-probability of the
   decoded text under a smoothed character n-gram model trained on a real
   corpus. The Latin reference is built on Thomas à Kempis' *De Imitatione
   Christi* (~1420s), contemporary with the manuscript's 1404–1438 carbon
   dating; Froissart's *Chroniques* (Middle French) and Leizarraga's Basque
   New Testament anchor other families to the right era where possible.
3. **The search is validated** — the benchmark mode encrypts held-out
   reference text with a random substitution, a synthetic abbreviation
   cipher, or a shuffled-letters anagram cipher, and checks the annealer
   recovers it: **100% letter accuracy in seconds** for substitution,
   75–93% for abbreviation, ~97% word recovery for anagrams. So when the
   same machinery fails on Voynichese, that is evidence about the text,
   not about the solver.
4. **Results are anchored** — every run reports its held-out score between a
   *random-key floor* (chance) and a *reference-language ceiling* (what real
   text scores), as "gap closed", plus the **dictionary word-match rate** of
   the decoded held-out text (short and long words separately — long-word
   matches are the hard currency; short ones come cheap). Training and
   validation use disjoint alternating lines, so the optimizer cannot
   simply memorize — and when an optimizer games a loose objective past the
   ceiling (the anagram mode can), the verdict says so in plain words.

The reproducible finding — consistent with the published literature — is
that no substitution-family decoding into any of the 16 reference languages
turns Voynichese into language, even though its rigid word structure lets an
optimizer close much of the statistical gap. Negative results are results;
[ANALYSIS.md](ANALYSIS.md) lays out what they imply and what would actually
settle the question.

## Reference languages

Twenty-eight corpora spanning Rome's medieval trade world — Europe, the
Middle East, Africa and Asia — each labelled with its period in the GUI:

| Family | References |
|--------|------------|
| Italic | Latin — *De Imitatione Christi* (~1420s) + *De Bello Gallico* |
| Romance | Italian — Dante (c. 1320) · Middle French — Froissart (c. 1370–1400) · Spanish — *Don Quijote* (1605) · Portuguese — *Os Lusíadas* (1572) · Catalan (19th c. literary) |
| Celtic | Welsh (19th c. literary) · Irish — *Niamh* (Munster Irish, 1907) |
| Germanic | English — KJV (1611) · German — Luther tradition · Dutch — Bible · Icelandic (closest living to Old Norse) |
| Slavic | Czech — *Bible kralická* (1613) · Polish — *Pan Tadeusz* (1834) · Russian — Synodal (1876) |
| Hellenic | Greek — Koine New Testament (the Byzantine standard) |
| Albanian | Albanian — Bible (Venetian Albania) |
| Uralic | Hungarian — Jókai (1872) · Finnish — *Kalevala* (1849) |
| Turkic | Turkish — Bible (Ottoman sphere; the Ardıç Old-Turkic theory) |
| Iranian | Persian — Bible (Silk Road lingua franca) |
| Indo-Aryan | Hindi — Bible (Indian Ocean trade) |
| Semitic | Arabic — Bible (Mediterranean lingua franca) · Hebrew — Bible (Hauer & Kondrak's top candidate) · Amharic — Bible (Ethiopian embassy to Rome, 1441) |
| Cushitic | Somali — Bible (Mogadishu / Indian Ocean coast) |
| Sinitic | Chinese — Bible as toneless pinyin syllables (Stolfi's comparison) |
| Isolate | Basque — Leizarraga New Testament (1571) |

Non-Latin scripts are transliterated **one sign per sign** (Greek θ→q,
Cyrillic ч→q, Arabic ش→w, Hebrew ש→w; Ethiopic and Devanagari via their
Unicode-name romanizations; Chinese via toneless pinyin) — multi-letter
romanizations like θ→"th" would smuggle fake digraph statistics into the
language models. Arabic and Hebrew pair naturally with the `abjad` flag,
since those scripts already write few vowels.

On the **Byblos-script / Italo-Celtic theory**: the Byblos syllabary is
itself undeciphered and Venetic/Gaulish corpora are a few hundred
inscriptions, so the theory cannot be tested end-to-end by anyone. Its
two testable halves are covered: a CV-syllabary reading *is* the
`abbreviation` hypothesis (token → consonant+vowel), and the Celtic and
Italic references give the language side its nearest trainable relatives.
If Italo-Celtic morphology lay under Voynichese, Welsh/Irish/Latin would
separate from unrelated controls in the sweep. See
[ANALYSIS.md](ANALYSIS.md) for the full assessment and for the
quattrocento context (Giovanni Fontana's enciphered notebooks, Alberti,
Tranchedino's cipher ledger, Hildegard's invented script).

**Sweep mode** runs one configuration against all 28 references and ranks
them by gap closed — the one-click answer to "which language fits best, and
does any of them actually fit?".

---

## Modes for niche theories

Beyond the four key families and the abjad flag, every solve accepts:

* **Reading order** (`--reverse words|lines`) — mirror writing and full
  right-to-left reading.
* **Null tokens** (`--allow-nulls`, abbreviation only) — glyphs that decode
  to nothing, the camouflage Tranchedino's 1450s cipher ledger documents.
  Nulls pay rent (a per-occurrence bit penalty), or the optimizer would
  "solve" the manuscript by deleting it; validated at 100% recovery on
  synthetic null ciphers, nulls correctly identified.
* **Shuffled-text control** (`--control`) — rerun any configuration on
  token-scrambled pseudo-Voynichese with identical word lengths and line
  structure. Signal that survives the scramble is objective-gaming, not
  manuscript.
* **Nomenclator hypothesis** (`--hypothesis nomenclator`) — substitution
  plus a bounded, injective codebook assigning whole plaintext words to
  frequent Voynich word types, the structure of real quattrocento
  diplomatic ciphers. Codebook entries pay a description-length cost; the
  recovered codebook is reported in full. Validated at 73% on synthetic
  nomenclator ciphers (residual misses are code glyphs whose words can
  pose as cheap one-letter decodes — a documented objective limitation).
* **Self-citation diagnostics** (`python -m voynich diagnostics`, or the
  Diagnostics tab) — measures the copy-and-mutate fingerprint of the
  leading non-language theory (Timm & Schinner): how often a content word
  has a near-duplicate among the previous N words, versus shuffled nulls
  and real languages. Includes the **model showdown**: a static-lexicon
  model versus an autocopy generator, fitted on half the word stream and
  compared on held-out likelihood. With the conservative uniform-mutation
  kernel, the copy mechanism currently buys ~0.6 bits/word on formulaic
  KJV English and ~0 on Voynichese — reported as the non-separation it is;
  a plausibility-weighted mutation kernel is the next upgrade.

## The GUI

`python -m voynich gui` serves a single-page workbench (styled after the
manuscript itself — parchment, iron-gall ink, rubrication — and built by
**Montgomery Kuykendall**):

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
python -m voynich solve --reference latin --hypothesis anagram --lock-rounds 3
python -m voynich sweep --language A --hypothesis simple        # all 16 languages, ranked
python -m voynich benchmark --reference latin --mode anagram
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

The twelve trade-world and Celtic references, same configuration:

| #  | Reference  | Gap closed | Dict matches ≥3 | Decoded sample (held-out) |
|----|------------|-----------:|----------------:|---------------------------|
| 1  | Hindi      | 83.1%      | 38.8%           | `aana na naaa aana a na a nae` |
| 2  | Persian    | 82.7%      | 51.1%           | `iian an aria mian u ai i aid` |
| 3  | Somali     | 82.4%      | 55.1%           | `aaba ha haan aaha u ha a han` |
| 4  | **Irish**  | 81.1%      | 32.9%           | `nnso ar anna anan a an n ann` |
| 5  | **Welsh**  | 80.2%      | 37.9%           | `ddim ei eddo oiei i ei i ddi` |
| 6  | Arabic     | 79.5%      | 49.9%           | `aaha la liam mala m lh a lhm` |
| 7  | Hebrew     | 78.5%      | 44.6%           | `hhlh lu lihm lhlu h lh h lim` |
| 8  | Albanian   | 78.5%      | 20.2%           | `aari si shat tesa i si e sia` |
| 9  | Icelandic  | 74.8%      | 15.1%           | `aara ta trad datu a ta a tad` |
| 10 | Amharic    | 72.8%      | 4.9%            | `eehe ne neee eene o ne e nee` |
| 11 | Turkish    | 69.3%      | 40.1%           | `aana da duan nada o da a dan` |
| 12 | Chinese    | 66.5%      | 19.3%           | `aang ni nuan nana a ni a nan` |

The Celtic result is the **Italo-Celtic theory's test**: if Voynichese hid
Italo-Celtic morphology, Irish, Welsh and Latin would separate decisively
from unrelated controls. They don't — Irish (81.1%) and Welsh (80.2%) sit
mid-pack, *below Somali* (82.4%), and Latin sits at 73.9%. The ranking
everywhere tracks orthographic vowel-richness, not genetic affinity, and
no sample reads as language. Hebrew — the best published candidate — lands
at 78.5% with the same repetitive non-language output as everything else.

Under the **anagram hypothesis** (letters unordered within words, dictionary
scoring, 2 crib-locking rounds, 20k × 2), the optimizer *beats the
real-language ceiling* on every tested language — and that is a finding about
the objective, not the manuscript:

| Reference  | "Gap closed" | Dict matches ≥3 | Dict matches ≥5 | Sample |
|------------|-------------:|----------------:|----------------:|--------|
| Latin      | 108.4%       | 62.0%           | 28.2%           | `eeri? et test sete? a et e est` |
| Portuguese | 107.8%       | 62.7%           | 22.3%           | `aano? os sdad? aoso? a os o sao` |
| Italian    | 105.5%       | 54.9%           | 4.5%            | `eeil? la elli iele? e le e nel` |
| English    | 89.6%        | 60.6%           | 3.0%            | `soon to tuos? noto? i to o not` |

Letter-bag matching is loose enough that Voynichese's short rigid words can
be mapped onto frequent short dictionary words wholesale. The workbench
flags any above-ceiling verdict as objective-gaming and reports long-word
matches separately — real decipherments produce long-word matches; gamed
objectives mostly don't. (For calibration: on a *genuine* synthetic anagram
cipher the same solver recovers ~97% of words, long ones included.)

The **nomenclator** hypothesis (substitution + word codebook, 30k × 2,
seed 42, Currier A vs Latin) is the strongest costume yet: **95.3% gap
closed**, with a 25-entry codebook assigning frequent Voynich types to
Latin function words (`aiin→me`, `ar→esse`, `chol→in`, `char→omnia`) —
and long-word dictionary matches of **1.0%** with unreadable samples
(`ttis et este ates [et] et [te] eti`). The progression across
hypotheses — simple 74%, abbreviation 80%, anagram >100%, nomenclator
95% — is itself the lesson: every added degree of freedom buys statistics,
none buys language, and the long-word match rate exposes the difference
every time.

Three readings of the sweep table:

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

---

*Built by Montgomery Kuykendall.*
