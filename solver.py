"""
solver.py
Optimizes the PositionalDecoder using MCMC (Simulated Annealing).
"""
import math
import random
import copy
from pathlib import Path
import collections
import numpy as np

# Re-use your existing metrics code
import experiment.metrics as metrics
from data_loader import load_voynich_lines
from decode_v2 import PositionalDecoder

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ITERATIONS = 100000  # Increased for deeper search
START_TEMP = 2.0
END_TEMP = 0.001

def generate_initial_map(vocab, target_alphabet):
    """Randomly assign vocab tokens to target letters."""
    mapping = {}
    shuffled_alpha = list(target_alphabet)
    # Ensure every vocab token has a mapping
    for i, token in enumerate(vocab):
        # We cycle through the alphabet if vocab > 26
        letter = shuffled_alpha[i % len(shuffled_alpha)]
        mapping[token] = letter
        # Shuffle occasionally to prevent deterministic assignment patterns
        if i % len(shuffled_alpha) == 0:
            random.shuffle(shuffled_alpha)
    return mapping

def solve(voynich_path, reference_path, vocab_path):
    print(f"--- Solving {voynich_path} against {reference_path} ---")

    # 1. Load Resources
    # Note: load_voynich_lines defaults to "A" if we don't specify, which is what we want.
    lines = load_voynich_lines(voynich_path)
    vocab = Path(vocab_path).read_text().splitlines()
    ref_text = Path(reference_path).read_text()
    ref_trigrams = metrics.ngram_counter(ref_text, 3)

    # 2. Split Data (Interleaved)
    train_lines = lines[0::2]
    test_lines = lines[1::2]
    print(f"Data Loaded: {len(train_lines)} train lines, {len(test_lines)} test lines.")

    # 3. Initialize Mappings
    current_maps = {
        "start": generate_initial_map(vocab, ALPHABET),
        "body":  generate_initial_map(vocab, ALPHABET),
        "end":   generate_initial_map(vocab, ALPHABET)
    }

    decoder = PositionalDecoder(current_maps["start"], current_maps["body"], current_maps["end"])

    # Initial Evaluation
    decoded_text = decoder.decode_text(train_lines)
    current_score = metrics.cosine_similarity(metrics.ngram_counter(decoded_text, 3), ref_trigrams)

    best_score = current_score
    best_maps = copy.deepcopy(current_maps)

    print(f"Initial Score: {current_score:.5f}")

    # 4. Annealing Loop
    print("Optimizing...")
    for i in range(ITERATIONS):
        # Temp Schedule
        frac = i / ITERATIONS
        temp = START_TEMP * (END_TEMP / START_TEMP) ** frac

        # Mutation: Pick a map and swap two tokens
        target_key = random.choice(["start", "body", "end"])
        target_map = current_maps[target_key]

        t1, t2 = random.sample(vocab, 2)
        old_val1, old_val2 = target_map[t1], target_map[t2]

        # Apply Swap
        target_map[t1], target_map[t2] = old_val2, old_val1

        # Evaluate
        new_text = decoder.decode_text(train_lines)
        new_score = metrics.cosine_similarity(metrics.ngram_counter(new_text, 3), ref_trigrams)

        # Acceptance Probability
        delta = new_score - current_score

        if delta > 0 or math.exp(delta / temp) > random.random():
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_maps = copy.deepcopy(current_maps)
                if i % 5000 == 0:
                    print(f"  Iter {i}: Best Score = {best_score:.5f}")
        else:
            # Revert
            target_map[t1], target_map[t2] = old_val1, old_val2

    # 5. Final Validation on HELD-OUT Test Set
    print("\n--- Final Validation ---")
    final_decoder = PositionalDecoder(best_maps["start"], best_maps["body"], best_maps["end"])

    test_text = final_decoder.decode_text(test_lines)
    test_score = metrics.cosine_similarity(metrics.ngram_counter(test_text, 3), ref_trigrams)

    print(f"Training Score (Best): {best_score:.5f}")
    print(f"TEST SET Score:        {test_score:.5f}")

    # Heuristic interpretation
    if test_score > 0.65:
        print("CONCLUSION: STRONG MATCH. The mapping generalizes well.")
    elif test_score > 0.50:
        print("CONCLUSION: MODERATE. Some structure was found, but may be overfitting.")
    else:
        print("CONCLUSION: WEAK. The text does not match the target language structure.")

    # Save Results? (Optional)
    # You can inspect 'test_text' here to see if it's readable!
    print(f"\nSample Decoded Text (First 200 chars):\n{test_text[:200]}")

if __name__ == "__main__":
    solve(
        voynich_path="data/interlinear_full_words.txt",
        reference_path="data/english_brown.txt", # Try changing to italian_sample.txt next!
        vocab_path="data/vocab_a.txt"
    )
