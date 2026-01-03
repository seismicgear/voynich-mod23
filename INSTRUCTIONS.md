# INSTRUCTIONS.md (Version 2.0)

This guide walks you through setting up and running the Version 2.0 experiment (Positional Decoder + BPE + Solver).

---

## Step 1: Clone & Install

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourname/voynich-positional-decoder.git
    cd voynich-positional-decoder
    ```

2.  Install dependencies (requires Python 3.8+):
    ```bash
    pip install -r requirements.txt
    ```

---

## Step 2: Setup Data

Run the setup script to download the Voynich "Interlinear" file and the NLTK reference corpora (English Brown, Italian UDHR).

```bash
python3 setup_v2.py
```

*   **Creates:** `data/` directory.
*   **Downloads:** `data/interlinear_full_words.txt`, `data/english_brown.txt`, `data/italian_sample.txt`.

---

## Step 3: Learn Vocabulary (Tokenization)

Run the BPE tokenizer to learn the manuscript's vocabulary. This script merges common pairs (like `c`+`h` -> `ch`) to find the best atomic units.

```bash
python3 tokenize_eva.py
```

*   **Input:** `data/interlinear_full_words.txt` (Language A)
*   **Output:** `data/vocab_a.txt` (List of learned tokens)

---

## Step 4: Run the Solver

Run the Simulated Annealing solver to find the optimal positional mapping.

```bash
python3 solver.py
```

*   **Input:** `data/interlinear_full_words.txt`, `data/english_brown.txt`, `data/vocab_a.txt`
*   **Process:**
    1.  Splits data into Train (Even Lines) and Test (Odd Lines).
    2.  Optimizes a mapping to make the Train set look like English.
    3.  Runs for ~100,000 iterations (takes 1-5 minutes).
*   **Output:**
    *   Prints progress (`Iter 5000: Best Score = 0.45...`).
    *   **Final Result:** `Training Score` vs `TEST SET Score`.
    *   Sample decoded text.

### Interpreting the Output

Look at the **TEST SET Score**:

*   **> 0.65:** Strong Match. The mapping generalizes.
*   **0.50 - 0.65:** Moderate Match.
*   **< 0.50:** Weak Match / Failure.

---

## Customization

To test against Italian instead of English, edit `solver.py` (bottom of file):

```python
solve(
    voynich_path="data/interlinear_full_words.txt",
    reference_path="data/italian_sample.txt",  # <--- Change this
    vocab_path="data/vocab_a.txt"
)
```
