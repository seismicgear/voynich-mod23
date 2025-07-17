# Modular-23 Decoder for Voynich Manuscript

This repository contains an experimental framework to test the hypothesis that **Voynichese encodes modular structure** through modular arithmetic (mod 23), mapped to a 23-letter Latin alphabet. The goal is to determine whether this decoding method yields statistically meaningful structure or performs no better than random baselines.

---

## Repo Contents

| File/Dir             | Purpose |
|----------------------|---------|
| `decoder.py`         | Maps EVA glyphs → numbers → modular inverses (mod 23) → Latin letters. Core decoding logic.
| `metrics.py`         | Computes Shannon entropy, n-grams, cosine similarity, and Monte Carlo null distributions.
| `run_experiment.py`  | Loads corpus, applies decoder, prints observed metrics, and runs 10,000× shuffle test.
| `data/`              | Contains `voynich_eva_takahashi.txt` and a Latin corpus for trigram comparison.
| `results/`           | (Optional) Save your entropy curves, p-values, and decoded strings here.
| `notebooks/`         | For interactive experiments or visualizations (optional).
| `src/`               | Alternate location for core modules if separating cleanly.

---

## How It Works

### Step 1: **Decode**
Maps glyphs to numbers (e.g. `'q' → 1`, `'o' → 2`)  
→ Invert under mod 23 (e.g. `1⁻¹ mod 23 = 1`, `2⁻¹ mod 23 = 12`)  
→ Map to classical Latin letters (e.g. `1 → A`, `12 → L`)

### Step 2: **Measure**
- Shannon entropy of the output string
- Trigram cosine similarity vs. a 15th-century Latin corpus

### Step 3: **Monte Carlo Test**
- Randomly shuffle the Latin letter assignments 10,000×
- Compute entropy and trigram similarity for each run
- Compare your mapping to the null distribution

---

## Evaluation Metrics

| Metric                | Interpretation |
|------------------------|----------------|
| **Shannon entropy**    | Lower = more structured; compare against shuffled controls  
| **Trigram similarity** | Higher = closer to natural language (Latin)  
| **Monte Carlo p-values** | How unlikely your results are under randomness

---

## Success Criteria

- `p_entropy < 0.01` and `p_trigram < 0.01`  
- Robust performance across Currier A and B  
- Predictive consistency on held-out folios

---

## Setup

```bash
pip install pandas numpy scipy nltk
python -m nltk.downloader punkt
