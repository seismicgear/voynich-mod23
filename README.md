# Modular-23 Decoder for Voynich Manuscript

This repository contains an experimental framework to test the hypothesis that
a modular‑23 inverse cipher applied to Voynich EVA glyphs yields text with
natural‑language‑like structure (compression, n‑grams) rather than behaving
like a random labeling.

---

## Quickstart

```bash
pip install -r requirements.txt
python run_experiment.py --eva data/voynich_eva_takahashi.txt --latin data/latin_reference.txt --n-iter 10000
```

## Repo Contents

| File/Dir             | Purpose |
|----------------------|---------|
| `decoder.py`         | Maps EVA glyphs → numbers → modular inverses (mod 23) → Latin letters. Core decoding logic.
| `metrics.py`         | Computes gzip compression size, n-grams, cosine similarity, and Monte Carlo null distributions.
| `run_experiment.py`  | Loads corpus, applies decoder, prints observed metrics, and runs 10,000× shuffle test.
| `data/`              | Contains `voynich_eva_takahashi.txt` and a Latin corpus for trigram comparison.
| `results/`           | (Optional) compression stats, p-values, and decoded strings here.
| `notebooks/`         | For interactive experiments or visualizations (optional).
| `src/`               | Alternate location for core modules if separating cleanly.

---

## How It Works

### Step 1: **Decode**
Maps glyphs to numbers (e.g. `'q' → 1`, `'o' → 2`)  
→ Invert under mod 23 (e.g. `1⁻¹ mod 23 = 1`, `2⁻¹ mod 23 = 12`)  
→ Map to classical Latin letters (e.g. `1 → A`, `12 → L`)

### Step 2: **Measure**
- gzip compression size of the decoded string
- Trigram cosine similarity vs. a 15th-century Latin corpus

### Step 3: **Monte Carlo Test**
- Randomly shuffle the Latin letter assignments 10,000×
- Compute gzip size and trigram similarity for each run
- Compare your mapping to the null distribution

---

## Evaluation Metrics

| Metric                   | Interpretation |
|--------------------------|----------------|
| **Gzip compression size** | Smaller = more structured; compare against shuffled controls
| **Trigram similarity**    | Higher = closer to natural language (Latin)
| **Monte Carlo p-values**  | How unlikely your results are under randomness

---

## Success Criteria

- `p_gzip < 0.01` and `p_trigram < 0.01`
- Robust performance across Currier A and B  
- Predictive consistency on held-out folios

---

## Setup

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for a full walkthrough.
