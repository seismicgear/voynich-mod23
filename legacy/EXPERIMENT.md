# EXPERIMENT.md (Version 2.0)

## Hypothesis: Positional Decoding

### H₀ (Null Hypothesis)
There is no structural difference between the mapping of glyphs at the beginning, middle, or end of a line. A single global mapping should suffice to decode the text if it is a simple substitution cipher.

### H₁ (Alternative Hypothesis)
The Voynich Manuscript uses a **state-dependent polyalphabetic cipher**. The mapping function $f(g)$ depends on the line position state $S \in \{Start, Body, End\}$.
*   **Start:** First token of a line.
*   **Body:** Intermediate tokens.
*   **End:** Last token of a line.

This structure implies that the "same" glyph may map to different Latin letters depending on where it appears in the line (e.g., `chedy` at the start might be 'H', but `chedy` at the end might be 'S').

---

## Experimental Design

### 1. Data Preparation
*   **Source:** `interlinear_full_words.txt` (Currier Language A).
*   **Tokenization:** Learned BPE vocabulary (e.g., `ch`, `sh`, `qo`) rather than raw characters.
*   **Splitting:** We use an **Interleaved Split**:
    *   **Training Set:** All EVEN lines (0, 2, 4...). Used to optimize the mappings.
    *   **Testing Set:** All ODD lines (1, 3, 5...). Used to validate the result.

### 2. Optimization (Simulated Annealing)
We search for three simultaneous mappings ($M_{start}, M_{body}, M_{end}$) that maximize the **Trigram Cosine Similarity** between the decoded training text and a target language reference (e.g., English or Italian).

**Algorithm:**
1.  Initialize random mappings for Start, Body, and End.
2.  **Loop** (N iterations):
    *   Propose a change: Swap two letters in one of the mappings.
    *   Decode the *Training Set* with the new mappings.
    *   Calculate Score: Cosine Similarity of trigrams vs. Reference Corpus.
    *   **Accept** if Score improves OR with probability $e^{\Delta / T}$ (Metropolis criterion).
    *   Decrease Temperature $T$.

### 3. Validation & Interpretation
After training, we apply the *best found mappings* to the **Testing Set** (which the solver never saw).

*   **Training Score:** How well we fit the even lines.
*   **Testing Score:** How well that fit generalizes to the odd lines.

**Interpretation:**
*   **High Test Score (> 0.65):** Strong evidence of a linguistic signal. The structure found in even lines predicts the structure in odd lines.
*   **Low Test Score (< 0.50) with High Train Score:** Overfitting. The solver "memorized" the training set but found no real linguistic pattern.
*   **Low Scores:** The hypothesis or the target language is incorrect.
