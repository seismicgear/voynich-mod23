# EXPERIMENT.md

## Hypothesis

### H₀ (Null Hypothesis): No Special Structure
1. **Structure:** After modular‑23 inversion, the decoded string has compressibility (gzip size) indistinguishable from a randomly shuffled sequence of the same characters.
2. **Linguistic Affinity:** The decoded string's trigram profile is no more similar to historical Latin than any random monoalphabetic relabeling of the same text.
3. **Distribution:** The text's Entropy and Index of Coincidence are indistinguishable from random noise or do not resemble natural language baselines.

### H₁ (Alternative Hypothesis): Meaningful Signal
The modular‑23 mapping yields:
1. **Structure:** Significantly smaller gzip size (higher compressibility) than shuffled controls, indicating preserved sequential patterns (morphemes, words).
2. **Linguistic Affinity:** Higher trigram cosine similarity to historical Latin than random monoalphabetic relabelings, indicating that the specific Mod-23 mapping captures Latin-like n-gram frequencies better than chance.
3. **Properties:** Entropy and Index of Coincidence comparable to samples of actual Latin text (not just random strings).

---

## Experimental Design

The experiment applies the modular-23 inverse decoding to the EVA text and compares it against three distinct baselines/null models.

### 1. Structure Test (Gzip)
*   **Metric:** Gzip compression size (bytes).
*   **Null Model:** `Shuffle Text`.
    *   **Randomize:** The sequence of characters in the decoded text is randomly permuted.
    *   **Fix:** The set of characters (frequency distribution) remains identical.
*   **Interpretation:** If the observed text compresses significantly better (smaller size) than the shuffled versions, the text contains non-random sequential structure (e.g., repeated words, prefixes, suffixes).

### 2. Mapping Specificity Test (Trigrams & Gzip)
We employ two null models to test the mapping's validity:

**A. Alphabet Shuffle (Linguistic Affinity)**
*   **Metric:** Cosine similarity of trigram frequency vectors vs. a Latin reference corpus.
*   **Null Model:** Random monoalphabetic substitution on the *decoded* text.
*   **Interpretation:** Does the decoded text have Latin-like trigrams regardless of the specific letter labels? This tests if the underlying structure is compatible with Latin.

**B. Glyph Mapping Shuffle (Mapping Validation)**
*   **Metric:** Gzip, Trigrams, Entropy, IoC.
*   **Null Model:** `Random Glyph Mapping`.
    *   **Randomize:** The assignment of numeric values (1..23) to EVA glyphs is randomly shuffled.
    *   **Fix:** The EVA text and the Mod-23 logic.
*   **Interpretation:** Is the specific `DEFAULT_GLYPH_TO_NUM` mapping special? If the observed metrics are significantly better than those from random glyph-to-number assignments, it supports the hypothesis that the mapping was not found by chance (or that it was carefully tuned).

### 3. Natural Language Profile Test (Entropy, IoC, Trigrams)
*   **Metrics:** Shannon Entropy, Index of Coincidence (IoC), Trigram Cosine Similarity.
*   **Baseline:** `Latin Windows`.
    *   **Method:** We sample random contiguous windows from the Latin reference corpus of the same length as the decoded text.
    *   **Comparison:** We compare the observed metrics against the distribution of these Latin windows.
*   **Interpretation:** This tells us if the decoded text "looks like" a piece of Latin text of the same length, rather than just "better than random noise."

### Controls
*   **Input Data:** Takahashi EVA transcription (cleaned).
*   **Reference Data:** 15th-century Latin corpus.
*   **Fixed Parameters:**
    *   EVA tokenization (greedy longest-match).
    *   `glyph_to_num` mapping (presumed input values).
    *   Mod-23 inversion logic.
*   **Train/Test Split (Optional):** To avoid overfitting the mapping, the experiment can be run on a held-out fraction of the text (`--test-fraction`).

### Statistics
For each metric, we report:
*   **Observed Value:** The metric calculated on the actual decoded text.
*   **Null/Baseline Distribution:** Mean and Standard Deviation of the null model (N iterations).
*   **Z-score:** Number of standard deviations the observation is from the null mean.
*   **p-value:** Probability of observing a result at least as extreme as the actual result under the null hypothesis.
