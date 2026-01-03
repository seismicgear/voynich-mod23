"""
cli.py

Orchestrates the experiment:
1. Loads data
2. Decodes Voynich text
3. Calculates observed metrics
4. Runs Monte Carlo simulations (Null Models)
5. Saves results to JSON
6. (Optional) Plots histograms
"""

import argparse
import json
import pathlib
import random
import re
import time
from datetime import datetime, timezone

import experiment.decode as dec
import experiment.metrics as met
import experiment.null_models as nulls
import experiment.stats as st

def load_eva(path: str) -> list[str]:
    """Return a list of EVA tokens (lowercase a‑z only)."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"EVA file not found: {path}")
    raw = p.read_text().split()
    return [w for w in raw if re.fullmatch(r"[a-z]+", w)]

def load_latin(path: str) -> str:
    """Return the Latin reference corpus as an uppercase, letters-only string."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Latin file not found: {path}")
    text = p.read_text().upper()
    return re.sub(r"[^A-Z]", "", text)

def split_eva_words(eva_words, test_fraction=0.5, seed=None):
    """
    Split EVA words into train/test sets to avoid overfitting the mapping.
    """
    rng = random.Random(seed)
    eva_words_copy = list(eva_words)
    rng.shuffle(eva_words_copy)
    split_idx = int(len(eva_words_copy) * (1 - test_fraction))
    return eva_words_copy[:split_idx], eva_words_copy[split_idx:]

def run_experiment(
    eva_path: str,
    latin_path: str,
    n_iter: int,
    seed: int | None,
    plot: bool,
    no_raw: bool,
    test_fraction: float = 0.0,
    n_latin_windows: int = 500
):
    # 1. Setup
    if seed is not None:
        random.seed(seed)

    # We also need a distinct RNG for splitting/sampling to keep things cleaner
    rng = random.Random(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    print(f"[{timestamp}] Starting experiment...")
    print(f"  Seed: {seed}")
    print(f"  Iterations: {n_iter}")

    # 2. Load Data
    try:
        eva_words = load_eva(eva_path)
        latin_text = load_latin(latin_path)

        print(f"  Loaded {len(eva_words)} EVA words.")
        print(f"  Loaded Latin corpus ({len(latin_text)} chars).")

        # Step 0: Sanity Checks (Enforce "realish" data)
        if len(eva_words) < 1000 or len(latin_text) < 10000:
            raise RuntimeError(
                f"Data looks tiny (EVA={len(eva_words)}, Latin={len(latin_text)}). "
                "You are probably using the stub files; drop in real corpora as described in data/README.md."
            )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Step 5: Train/Test Split
    if test_fraction > 0.0:
        train_words, test_words = split_eva_words(eva_words, test_fraction, seed)
        print(f"  Splitting data: {len(train_words)} train, {len(test_words)} test (fraction={test_fraction})")
        # We only experiment on the test set
        eva_words = test_words
        if not eva_words:
            print("  Error: Test set is empty after split!")
            return
    else:
        print("  Using full dataset (no train/test split).")

    # 3. Decode
    decoder = dec.Mod23Decoder()
    decoded_words = [decoder.decode_word(w) for w in eva_words]
    decoded_text = "".join(decoded_words)

    if not decoded_text:
        print("  Error: Decoded text is empty!")
        return

    # 4. Measure Observed Metrics
    print("  Calculating observed metrics...")

    # Gzip Size
    obs_gzip = met.gzip_size(decoded_text)

    # Trigram Similarity
    obs_tri_counts = met.ngram_counter(decoded_text, 3)
    ref_tri_counts = met.ngram_counter(latin_text, 3)
    obs_similarity = met.cosine_similarity(obs_tri_counts, ref_tri_counts)

    # Entropy & IoC
    obs_entropy = met.shannon_entropy(decoded_text)
    obs_ioc = met.index_of_coincidence(decoded_text)

    # Step 1: Explicit Latin Baselines
    latin_gzip = met.gzip_size(latin_text)
    latin_entropy = met.shannon_entropy(latin_text)
    latin_ioc = met.index_of_coincidence(latin_text)
    latin_tri = met.ngram_counter(latin_text, 3) # Stored for completeness/logging

    # 5. Latin Window Comparison (Step 2 & 3)
    print(f"  Sampling {n_latin_windows} Latin windows for baseline comparison...")
    latin_windows = nulls.sample_latin_windows(latin_text, len(decoded_text), n_latin_windows, rng)

    latin_window_sims = []
    latin_window_entropies = []
    latin_window_iocs = []

    for w in latin_windows:
        w_tri = met.ngram_counter(w, 3)
        latin_window_sims.append(met.cosine_similarity(w_tri, ref_tri_counts))
        latin_window_entropies.append(met.shannon_entropy(w))
        latin_window_iocs.append(met.index_of_coincidence(w))

    latin_sim_stats = st.calculate_stats(obs_similarity, latin_window_sims, smaller_is_better=False)
    latin_entropy_stats = st.calculate_stats(obs_entropy, latin_window_entropies, smaller_is_better=False) # Or True? Usually we want similar, so maybe distance?
    # For Entropy/IoC, we usually want to know if it's "comparable".
    # If we treat it as a "greater" test, we see if it's higher than average Latin.
    # The user suggested: "p_value_greater" in the JSON example.
    latin_ioc_stats = st.calculate_stats(obs_ioc, latin_window_iocs, smaller_is_better=False)


    # 6. Monte Carlo Simulations (Null Models)
    print(f"  Running {n_iter} Monte Carlo simulations...")

    null_gzip_samples = [] # Structure Null (Shuffle Text)

    # Mapping Nulls:
    # 1. Alphabet Shuffle (Random substitution on decoded text)
    #    - Measures: "Is the structure Latin-like regardless of specific labels?" (for Trigram)
    # 2. Glyph Mapping Shuffle (Random EVA->Num mapping)
    #    - Measures: "Is this specific mapping special compared to other valid mappings?"

    alphabet_shuffle_sim_samples = []

    # New Mapping-Level Null (Step 4)
    # We compute Gzip, Trigram, Entropy, IoC for random glyph mappings.
    glyph_map_gzip_samples = []
    glyph_map_sim_samples = []
    glyph_map_entropy_samples = []
    glyph_map_ioc_samples = []

    # Pre-calculate alphabet for shuffling (for the alphabet shuffle null)
    alphabet = decoder.ALPHABET

    start_time = time.time()

    for i in range(n_iter):
        if n_iter >= 10 and (i + 1) % (n_iter // 10) == 0:
            print(f"    ... {i + 1}/{n_iter} iterations")

        # Null Model 1: Text Shuffle (Structure)
        # Test: Does the text have more structure (lower gzip) than a random shuffle?
        shuffled_text = nulls.shuffle_text(decoded_text)
        null_gzip_samples.append(met.gzip_size(shuffled_text))

        # Null Model 2: Alphabet Shuffle (Legacy Trigram Null)
        # Random substitution on the ALREADY decoded text.
        # This keeps the "shape" of the words but changes letters.
        permuted_text = nulls.shuffle_alphabet_mapping(decoded_text, alphabet)
        perm_tri_counts = met.ngram_counter(permuted_text, 3)
        alphabet_shuffle_sim_samples.append(met.cosine_similarity(perm_tri_counts, ref_tri_counts))

        # Null Model 3: Glyph Mapping Shuffle (True Mapping Null)
        # Generate a random EVA->Num mapping, decode, and measure.
        random_map = nulls.random_glyph_mapping(decoder.DEFAULT_GLYPH_TO_NUM)
        random_decoder = dec.Mod23Decoder(glyph_to_num=random_map)

        rand_decoded_words = [random_decoder.decode_word(w) for w in eva_words]
        rand_decoded_text = "".join(rand_decoded_words)

        glyph_map_gzip_samples.append(met.gzip_size(rand_decoded_text))
        glyph_map_entropy_samples.append(met.shannon_entropy(rand_decoded_text))
        glyph_map_ioc_samples.append(met.index_of_coincidence(rand_decoded_text))

        rand_tri_counts = met.ngram_counter(rand_decoded_text, 3)
        glyph_map_sim_samples.append(met.cosine_similarity(rand_tri_counts, ref_tri_counts))

    elapsed = time.time() - start_time
    print(f"  Simulation finished in {elapsed:.2f}s.")

    # 7. Statistics
    # Structure test (Text Shuffle)
    gzip_stats = st.calculate_stats(obs_gzip, null_gzip_samples, smaller_is_better=True)

    # Mapping test (Alphabet Shuffle - Legacy/Basic)
    alphabet_shuffle_stats = st.calculate_stats(obs_similarity, alphabet_shuffle_sim_samples, smaller_is_better=False)

    # Mapping test (Glyph Mapping Shuffle - Advanced)
    glyph_map_gzip_stats = st.calculate_stats(obs_gzip, glyph_map_gzip_samples, smaller_is_better=True)
    glyph_map_sim_stats = st.calculate_stats(obs_similarity, glyph_map_sim_samples, smaller_is_better=False)
    glyph_map_entropy_stats = st.calculate_stats(obs_entropy, glyph_map_entropy_samples, smaller_is_better=False) # or True, distance?
    glyph_map_ioc_stats = st.calculate_stats(obs_ioc, glyph_map_ioc_samples, smaller_is_better=False)

    # 8. Construct Result JSON
    results = {
        "meta": {
            "timestamp": timestamp,
            "eva_path": eva_path,
            "latin_path": latin_path,
            "n_iter": n_iter,
            "seed": seed,
            "modulus": decoder.MOD,
            "decoder": "Mod23Decoder_v1",
            "test_fraction": test_fraction,
            "n_latin_windows": n_latin_windows
        },
        "reference": {
            "latin": {
                "length": len(latin_text),
                "gzip": latin_gzip,
                "entropy": latin_entropy,
                "index_of_coincidence": latin_ioc
            }
        },
        "metrics": {
            "gzip": {
                "observed": obs_gzip,
                "null_text_shuffle": {
                    "mean": gzip_stats["mean"],
                    "std": gzip_stats["std"],
                    "z_score": gzip_stats["z_score"],
                    "p_value_smaller": gzip_stats["p_value"]
                },
                "null_glyph_mapping": {
                     "mean": glyph_map_gzip_stats["mean"],
                     "std": glyph_map_gzip_stats["std"],
                     "z_score": glyph_map_gzip_stats["z_score"],
                     "p_value_smaller": glyph_map_gzip_stats["p_value"]
                }
            },
            "trigram_cosine": {
                "observed": obs_similarity,
                "null_alphabet_shuffle": {
                    "mean": alphabet_shuffle_stats["mean"],
                    "std": alphabet_shuffle_stats["std"],
                    "z_score": alphabet_shuffle_stats["z_score"],
                    "p_value_greater": alphabet_shuffle_stats["p_value"]
                },
                "null_glyph_mapping": {
                    "mean": glyph_map_sim_stats["mean"],
                    "std": glyph_map_sim_stats["std"],
                    "z_score": glyph_map_sim_stats["z_score"],
                    "p_value_greater": glyph_map_sim_stats["p_value"]
                },
                "latin_windows": {
                    "mean": latin_sim_stats["mean"],
                    "std": latin_sim_stats["std"],
                    "p_value_greater": latin_sim_stats["p_value"]
                }
            },
            "entropy": {
                "observed": obs_entropy,
                "null_glyph_mapping": {
                    "mean": glyph_map_entropy_stats["mean"],
                    "std": glyph_map_entropy_stats["std"],
                    "p_value_greater": glyph_map_entropy_stats["p_value"]
                },
                "latin_windows": {
                    "mean": latin_entropy_stats["mean"],
                    "std": latin_entropy_stats["std"],
                    "p_value_greater": latin_entropy_stats["p_value"]
                }
            },
            "index_of_coincidence": {
                "observed": obs_ioc,
                "null_glyph_mapping": {
                    "mean": glyph_map_ioc_stats["mean"],
                    "std": glyph_map_ioc_stats["std"],
                    "p_value_greater": glyph_map_ioc_stats["p_value"]
                },
                "latin_windows": {
                    "mean": latin_ioc_stats["mean"],
                    "std": latin_ioc_stats["std"],
                    "p_value_greater": latin_ioc_stats["p_value"]
                }
            }
        }
    }

    if not no_raw:
        results["metrics"]["gzip"]["null_text_samples"] = null_gzip_samples
        results["metrics"]["gzip"]["null_glyph_samples"] = glyph_map_gzip_samples
        results["metrics"]["trigram_cosine"]["null_alphabet_samples"] = alphabet_shuffle_sim_samples
        results["metrics"]["trigram_cosine"]["null_glyph_samples"] = glyph_map_sim_samples

    # 9. Save Results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    out_file = results_dir / f"run_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {out_file}")

    # 10. Plotting (Optional)
    if plot:
        try:
            import matplotlib.pyplot as plt

            # Gzip Histogram (Text Shuffle)
            plt.figure(figsize=(10, 6))
            plt.hist(null_gzip_samples, bins=50, color='gray', alpha=0.7, label='Null (Shuffled Text)')
            plt.axvline(obs_gzip, color='red', linestyle='dashed', linewidth=2, label=f'Observed ({obs_gzip})')
            plt.title(f"Gzip Compression (Text Shuffle)\np={gzip_stats['p_value']:.5f}")
            plt.xlabel("Compressed Size (bytes)")
            plt.ylabel("Frequency")
            plt.legend()
            plot_path_gzip = results_dir / f"run_{timestamp}_gzip_text_hist.png"
            plt.savefig(plot_path_gzip)
            plt.close()

            # Similarity Histogram (Mapping Shuffle)
            plt.figure(figsize=(10, 6))
            plt.hist(mapping_sim_samples, bins=50, color='gray', alpha=0.7, label='Null (Random Mapping)')
            plt.axvline(obs_similarity, color='blue', linestyle='dashed', linewidth=2, label=f'Observed ({obs_similarity:.4f})')
            # Add Latin Windows distribution
            plt.hist(latin_window_sims, bins=50, color='green', alpha=0.5, label='Latin Windows')

            plt.title(f"Trigram Cosine Similarity (Mapping Null & Latin Windows)\np_map={mapping_sim_stats['p_value']:.5f}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            plt.legend()
            plot_path_sim = results_dir / f"run_{timestamp}_trigram_hist.png"
            plt.savefig(plot_path_sim)
            plt.close()

            print(f"  Plots saved to {results_dir}")

        except ImportError:
            print("  Warning: Matplotlib not found. Skipping plots.")
        except Exception as e:
            print(f"  Error generating plots: {e}")

    # Print Summary to Console
    print("\n--- Summary ---")
    print(f"Entropy: Obs={obs_entropy:.4f} | Latin Mean={latin_entropy_stats['mean']:.4f}")
    print(f"IoC:     Obs={obs_ioc:.4f} | Latin Mean={latin_ioc_stats['mean']:.4f}")
    print(f"Gzip:    Obs={obs_gzip} | Text Shuffle Mean={gzip_stats['mean']:.1f} | p={gzip_stats['p_value']:.5f}")
    print(f"Trigram: Obs={obs_similarity:.4f} | Alphabet Null Mean={alphabet_shuffle_stats['mean']:.4f} | p={alphabet_shuffle_stats['p_value']:.5f}")
    print(f"         vs Glyph Null: Mean={glyph_map_sim_stats['mean']:.4f} | p={glyph_map_sim_stats['p_value']:.5f}")
    print(f"         vs Latin Windows: Mean={latin_sim_stats['mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Voynich Modular-23 Experiment CLI")
    parser.add_argument("--eva", default="data/voynich_eva_takahashi.txt", help="Path to EVA transcription")
    parser.add_argument("--latin", default="data/latin_reference.txt", help="Path to Latin reference")
    parser.add_argument("--n-iter", type=int, default=10000, help="Monte Carlo iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate histograms (requires matplotlib)")
    parser.add_argument("--no-raw", action="store_true", help="Do not save raw null samples in JSON")
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Fraction of data to use for testing (0.0 = use all)")
    parser.add_argument("--latin-windows", type=int, default=500, help="Number of Latin windows to sample")

    args = parser.parse_args()

    run_experiment(
        eva_path=args.eva,
        latin_path=args.latin,
        n_iter=args.n_iter,
        seed=args.seed,
        plot=args.plot,
        no_raw=args.no_raw,
        test_fraction=args.test_fraction,
        n_latin_windows=args.latin_windows
    )

if __name__ == "__main__":
    main()
