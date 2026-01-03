# EXPERIMENT.md (Version 2.0)

## Hypothesis: Positional Encoding & Learned Vocabulary

### Core Premise
Version 2.0 moves beyond simple monoalphabetic substitution (like the V1 Mod-23 hypothesis) to test a more complex encoding scheme. We hypothesize that:

1.  **Positional Dependency:** The meaning or mapping of a Voynich glyph changes based on its position in the line. Specifically, "Line Start" (first word), "Line Body" (middle), and "Line End" (last word) may use different substitution tables or encounter different entropy constraints.
2.  **Atomic Tokens (BPE):** The standard "EVA" transcription is over-segmented. "Characters" like `c`, `h`, `o` are likely parts of larger composite glyphs (e.g., `cho`, `sh`). We use **Byte Pair Encoding (BPE)** to statistically discover these atomic units rather than guessing them.
3.  **Linguistic Affinity:** If the manuscript encodes a natural language (e.g., medieval Italian or English), there exists a mapping that produces trigram statistics (sequences of 3 letters) highly correlated with that language.

---

## Experimental Design

The experiment uses an **Optimization & Validation** approach rather than simple hypothesis testing.

### 1. Data Preparation
*   **Source:** `interlinear_full_words.txt` (Chirila et al.), filtered for **Currier Language A**.
*   **Reconstruction:** We group words by `(Folio, Line)` to reconstruct full lines, preserving the positional structure.
*   **Tokenization:** We run BPE on the raw EVA text to generate a vocabulary of ~50-100 "tokens" (e.g., `qo`, `daiin`, `ol`, etc.).

### 2. The Solver (Simulated Annealing)
We treat decoding as an optimization problem.
*   **Objective Function:** Maximize the **Cosine Similarity** between the trigram frequency vector of the *decoded text* and the *reference language* (e.g., Brown Corpus for English).
*   **State Space:** The solver maintains three separate mappings:
    *   `Map_Start`: Used for the first token of a line.
    *   `Map_Body`: Used for all middle tokens.
    *   `Map_End`: Used for the last token.
*   **Algorithm:** Metropolis-Hastings (Simulated Annealing).
    *   Propose a swap in one of the mappings.
    *   Accept if score improves.
    *   Accept with probability $e^{\Delta / T}$ if score worsens (to escape local optima).

### 3. Validation: Interleaved Train/Test Split
To ensure we aren't just "hallucinating" a language by overfitting (finding a mapping that forces random noise to look like English), we use a rigorous split:

*   **Training Set:** All **EVEN** numbered lines. The solver *only* sees these lines during optimization.
*   **Test Set:** All **ODD** numbered lines. These are held out completely.

### Interpretation
After the solver converges on the Training Set, we apply the final mappings to the **Test Set**.

*   **High Test Score (> 0.65):** Strong evidence. The rules learned on the even lines successfully predict the structure of the odd lines. This suggests a genuine linguistic system.
*   **Moderate Test Score (0.50 - 0.65):** Partial evidence. Some structure exists, but the mapping might be imperfect or the text may be a "stochastic language" (like a constructed language or chant).
*   **Low Test Score (< 0.50):** Failure. The solver "memorized" the training data but the pattern didn't hold up on unseen data. This supports the null hypothesis (hoax or meaningless gibberish).

---

## Metrics

*   **Trigram Cosine Similarity:** The primary objective. Measures the angle between the frequency vectors of the decoded text and the reference text. Range: [0, 1].
*   **Gzip Compression:** Used as a secondary check for structural regularity.
