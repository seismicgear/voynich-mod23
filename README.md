# Voynich Manuscript Decoder v2.0

This repository contains an advanced computational framework for decoding the Voynich Manuscript. It implements a **Positional Decoder** optimized via **Simulated Annealing (MCMC)** to test the hypothesis that the text is a state-dependent polyalphabetic cipher.

---

## Hypothesis: State-Dependent Polyalphabetic Cipher

We hypothesize that the "EVA" transcription of the Voynich Manuscript is not a simple monoalphabetic substitution, but a polyalphabetic system where the mapping changes based on the glyph's position within a line.

Specifically, we model the line as having three distinct states:
1.  **Start:** The first token of the line.
2.  **Body:** The middle tokens.
3.  **End:** The last token of the line.

Our goal is to learn three separate mapping functions ($f_{start}, f_{body}, f_{end}$) that maximize the linguistic similarity of the decoded text to a target language (e.g., 15th-century Latin, Italian, or English).

---

## Methodology

### Pipeline Diagram

```mermaid
graph TD
    %% Global Styles
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef artifact fill:#e0f2f1,stroke:#00695c,stroke-width:2px;

    subgraph Inputs ["1. Input Data & Resources"]
        RawVoynich[("Voynich Transcriptions<br>(EVA)")]:::data
        RefCorpus[("Reference Corpus<br>(English/Italian/Latin)")]:::data
    end

    subgraph Preprocessing ["2. Preprocessing & Initialization"]
        RawVoynich -->|"Parse & Filter (Lang A)"| DataLoader(data_loader.py):::process
        DataLoader -->|Reconstruct Lines| Lines[Line Objects]:::artifact
        Lines -->|BPE Tokenization| Tokenizer(tokenize_eva.py):::process
        Tokenizer -->|Generate| Vocab[("Vocabulary<br>(vocab_a.txt)")]:::artifact

        RefCorpus -->|N-gram Analysis| RefTrigrams[("Reference Trigrams<br>(Vector V_ref)")]:::artifact

        Lines -->|Interleaved Split| Split{Split Data}:::decision
        Split -->|Even Lines| TrainSet[("Training Set")]:::data
        Split -->|Odd Lines| TestSet[("Test Set")]:::data
    end

    subgraph Optimization ["3. Optimization (Simulated Annealing)"]
        direction TB
        Init["Init Random Maps<br>M_start, M_body, M_end"]:::process
        TrainSet --> LoopStart((Start Loop))
        Init --> LoopStart

        LoopStart --> Perturb["Perturbation:<br>Swap 2 tokens in random Map M"]:::process
        Perturb --> DecodeTrain[Decode Training Set]:::process
        DecodeTrain --> Measure["Measure:<br>Trigram Cosine Similarity (S_new)"]:::process
        Measure --> CalcDelta["Calculate ΔE = S_new - S_curr"]:::process

        RefTrigrams -.-> Measure

        CalcDelta --> CheckAccept{"Accept?<br>(Metropolis Criterion)"}:::decision
        
        %% Path 1: Accept
        CheckAccept -- "Yes (or P > rand)" --> Update[Update Current Map]:::process
        Update --> CheckBest{"Is S_new > S_best?"}:::decision
        
        CheckBest -- Yes --> SaveBest[Save Best Maps]:::process
        SaveBest --> CoolDown
        CheckBest -- No --> CoolDown
        
        %% Path 2: Reject
        CheckAccept -- No --> Revert[Revert Swap]:::process
        Revert --> CoolDown["Cool Down:<br>Decrease Temperature T"]:::process
        
        CoolDown --> LoopEnd{Iter < Max?}:::decision
        LoopEnd -- Yes --> LoopStart
    end

    subgraph Validation ["4. Validation & Output"]
        LoopEnd -- No --> FinalEval[Final Evaluation]:::process
        SaveBest -.->|Best Maps| FinalEval
        TestSet --> FinalEval
        FinalEval -->|Decode Test Set| DecodedText[("Decoded Text")]:::artifact
        FinalEval -->|Compute Final Metrics| Metrics[("Final Score<br>(Generalization)")]:::artifact
    end
```

### 1. Vocabulary Learning (BPE)
Instead of treating every EVA glyph as atomic, we use a simplified Byte Pair Encoding (BPE) algorithm to learn a vocabulary of common tokens (e.g., merging `c` and `h` into `ch`).

### 2. Metric: Trigram Cosine Similarity
We evaluate the quality of a candidate decryption by comparing its **trigram frequency vector** to that of a reference corpus (e.g., the Brown Corpus or UDHR). A higher cosine similarity indicates a more natural-looking text structure.

### 3. Optimization: Simulated Annealing
Because the search space for three simultaneous 26-letter permutations is incredibly vast ($26!^3$), we use Markov Chain Monte Carlo (MCMC) methods—specifically Simulated Annealing—to find an optimal configuration.

### 4. Validation: Interleaved Split
To prevent overfitting (where the solver just memorizes the specific letters that make *this* text look like English), we split the manuscript lines into **Training (Even Lines)** and **Testing (Odd Lines)** sets.
*   **Train:** The solver sees these lines and optimizes the mapping to maximize their score.
*   **Test:** The final mapping is applied to these held-out lines. A high score here indicates the pattern is generalizable.

---

## Usage

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for a step-by-step guide on setting up and running the experiment.

---

## Legacy Experiments (v1.0)

**Modular-23 Hypothesis**

Our initial experiment tested a simpler hypothesis: that the text was encoded via a fixed modular-23 inverse mapping ($x^{-1} \pmod{23}$).
*   **Code:** `legacy/run_experiment_v1.py`
*   **Notebook:** `legacy/reproduce_results_v1.ipynb`
*   **Findings:** While statistically distinguishable from random noise, the Mod-23 mapping failed to produce readable text or reach natural language metric thresholds. This code is retained for historical comparison.
