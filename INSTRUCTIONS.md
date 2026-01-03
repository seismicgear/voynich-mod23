# INSTRUCTIONS.md (Version 2.0)

This guide explains how to run the Version 2.0 Positional Decoder pipeline.

---

## 1. Setup Environment

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. The Pipeline

The V2 experiment consists of three sequential scripts. You must run them in order.

### Step A: Download Data (`setup_v2.py`)
Fetches the Voynich interlinear transcription and NLTK reference corpora (Brown/English, UDHR/Italian).

```bash
python setup_v2.py
```
*   **Output:** Creates `data/interlinear_full_words.txt`, `data/english_brown.txt`, etc.

### Step B: Learn Vocabulary (`tokenize_eva.py`)
Analyzes the Voynich text to learn "words" (tokens) using a simplified BPE algorithm. This handles composite glyphs like `ch`, `sh`, `qo` automatically.

```bash
python tokenize_eva.py
```
*   **Output:** Creates `data/vocab_a.txt` (List of most common tokens).

### Step C: Run the Solver (`solver.py`)
Runs the Simulated Annealing optimization to find the best mapping.

```bash
python solver.py
```
*   **Input:** Uses the data and vocabulary from previous steps.
*   **Process:** Trains on even lines, validates on odd lines.
*   **Duration:** ~1-5 minutes depending on iterations.

---

## 3. Interpreting Results

The solver outputs two critical metrics at the end:

```
Training Score (Best): 0.71234
TEST SET Score:        0.64501
```

1.  **Training Score:** How much the decoded text looks like the target language (Max = 1.0).
2.  **TEST SET Score:** The *real* validation.

| Test Score | Interpretation |
|------------|----------------|
| **> 0.65** | **Strong Match.** The mapping works on unseen text. High probability of linguistic structure. |
| **0.50 - 0.65** | **Moderate.** Some patterns match, but noise is high. |
| **< 0.50** | **Failure / Overfitting.** The solver just memorized the training data or the target language is wrong. |

---

## Legacy (Version 1.0)
For the old Mod-23 experiment, see `run_experiment.py`. Note that it is now deprecated.
