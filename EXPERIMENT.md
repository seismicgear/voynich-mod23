# EXPERIMENT.md

## Hypothesis

### H₀ (Null Hypothesis): No Special Structure
1. **Structure:** After modular‑23 inversion, the decoded string has compressibility (gzip size) indistinguishable from a randomly shuffled sequence of the same characters.
2. **Linguistic Affinity:** The decoded string's trigram profile is no more similar to historical Latin than any random monoalphabetic relabeling of the same text.

### H₁ (Alternative Hypothesis): Meaningful Signal
The modular‑23 mapping yields:
1. **Structure:** Significantly smaller gzip size (higher compressibility) than shuffled controls, indicating preserved sequential patterns (morphemes, words).
2. **Linguistic Affinity:** Higher trigram cosine similarity to historical Latin than random monoalphabetic relabelings, indicating that the specific Mod-23 mapping captures Latin-like n-gram frequencies better than chance.
3. **Properties:** Comparable Entropy and Index of Coincidence to natural language texts.

---

## Experimental Design

The experiment applies the modular-23 inverse decoding to the EVA text and compares it against two distinct null models.

### 1. Structure Test (Gzip)
*   **Metric:** Gzip compression size (bytes).
*   **Null Model:** `Shuffle Text`.
    *   **Randomize:** The sequence of characters in the decoded text is randomly permuted.
    *   **Fix:** The set of characters (frequency distribution) remains identical.
*   **Interpretation:** If the observed text compresses significantly better (smaller size) than the shuffled versions, the text contains non-random sequential structure (e.g., repeated words, prefixes, suffixes).

### 2. Latin-likeness Test (Trigrams)
*   **Metric:** Cosine similarity of trigram frequency vectors vs. a Latin reference corpus.
*   **Null Model:** `Shuffle Alphabet`.
    *   **Randomize:** The mapping between Voynich glyphs and Latin letters is randomly permuted (monoalphabetic substitution).
    *   **Fix:** The underlying structure of the Voynich text (repetition patterns) is preserved; only the "labels" change.
*   **Interpretation:** If the observed mapping yields higher similarity to Latin than random mappings, it suggests the specific Mod-23 choice aligns Voynich patterns with Latin patterns better than arbitrary assignment.

### Controls
*   **Input Data:** Takahashi EVA transcription (cleaned).
*   **Reference Data:** 15th-century Latin corpus.
*   **Fixed Parameters:**
    *   EVA tokenization (greedy longest-match).
    *   `glyph_to_num` mapping (presumed input values).
    *   Mod-23 inversion logic.

### Statistics
For each metric, we report:
*   **Observed Value:** The metric calculated on the actual decoded text.
*   **Null Distribution:** Mean and Standard Deviation of the null model (N iterations).
*   **Z-score:** Number of standard deviations the observation is from the null mean.
*   **p-value:** Probability of observing a result at least as extreme as the actual result under the null hypothesis.
