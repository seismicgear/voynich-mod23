# EXPERIMENT.md

This document defines the statistical experiment used to evaluate the modular-23 cipher hypothesis for the Voynich Manuscript.

---

## 1. Hypothesis

Voynichese encodes linguistic structure via a monoalphabetic cipher based on modular inversion (mod 23), producing output that exhibits:

- Smaller gzip compression size than randomized baselines
- Higher trigram similarity to known 15th-century Latin corpora
- Consistent behavior across different folio sections (Currier A/B)

---

## 2. Method

### 2.1 Decoder Construction

1. Map each EVA glyph to a unique number (1–23).
2. Apply modular inverse (mod 23) to each number.
3. Map the result to a letter in a 23-character Latin alphabet (A–Z, excluding J, U, W).
4. Join the resulting characters to form Latin-style output strings.

### 2.2 Corpus

- **Input**: `voynich_eva_takahashi.txt` (one word per line, labeled by folio and Currier classification if possible)
- **Reference**: `latin_reference.txt` (15th-century medical or botanical Latin text, lowercased)

---

## 3. Metrics

### 3.1 Gzip Compression Size

Byte length of the decoded output after gzip compression.
Smaller compressed size implies greater regularity/structure.

### 3.2 Trigram Cosine Similarity

- Extract character trigrams from the decoded Voynich output
- Compare against trigrams from the Latin reference using cosine similarity

Higher similarity indicates linguistic overlap with Latin morphology.

---

## 4. Monte Carlo Control

1. Shuffle the Latin alphabet 10,000 times
2. Re-decode the EVA corpus using the same glyph-to-number map but random Latin assignments
3. Record gzip size and trigram similarity for each iteration
4. Compute p-values by comparing observed metrics to the null distribution

---

## 5. Evaluation Criteria

| Test                     | Threshold                          |
|--------------------------|-------------------------------------|
| Gzip size (observed)     | Below 1st percentile of null dist   |
| Trigram similarity       | Above 99th percentile of null dist  |
| Currier A vs B gzip size | Within 1 SD of each other           |
| Out-of-sample bigram fit | Top 1% predictive log-likelihood    |

---

## 6. Reproducibility

To replicate this experiment:

1. Clone the repo
2. Place the EVA and Latin corpora in `/data/`
3. Run `run_experiment.py`
4. Compare the observed metrics and null distribution results

All logic for decoding, compression, similarity, and shuffling is contained in `decoder.py` and `metrics.py`.

---

## 7. Notes

- This framework does not assert final translation of Voynichese
- The goal is to falsify or validate the **existence of modular structure**
- Statistical outperformance does not imply semantic meaning

