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

def run_experiment(
    eva_path: str,
    latin_path: str,
    n_iter: int,
    seed: int | None,
    plot: bool,
    no_raw: bool
):
    # 1. Setup
    if seed is not None:
        random.seed(seed)

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
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Decode
    decoder = dec.Mod23Decoder()
    decoded_words = [decoder.decode_word(w) for w in eva_words]
    decoded_text = "".join(decoded_words)

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

    # 5. Monte Carlo Simulations
    print(f"  Running {n_iter} Monte Carlo simulations...")

    null_gzip_samples = []
    null_sim_samples = []

    # Pre-calculate alphabet for shuffling
    alphabet = decoder.ALPHABET

    start_time = time.time()

    for i in range(n_iter):
        if n_iter >= 10 and (i + 1) % (n_iter // 10) == 0:
            print(f"    ... {i + 1}/{n_iter} iterations")

        # Null Model 1: Text Shuffle (Structure)
        # Test: Does the text have more structure (lower gzip) than a random shuffle?
        shuffled_text = nulls.shuffle_text(decoded_text)
        null_gzip_samples.append(met.gzip_size(shuffled_text))

        # Null Model 2: Alphabet Shuffle (Linguistic Affinity)
        # Test: Is the text closer to Latin (higher similarity) than a random substitution?
        # Note: We apply substitution to the DECODED text to preserve its structure but change labels.
        permuted_text = nulls.shuffle_alphabet_mapping(decoded_text, alphabet)
        perm_tri_counts = met.ngram_counter(permuted_text, 3)
        null_sim_samples.append(met.cosine_similarity(perm_tri_counts, ref_tri_counts))

    elapsed = time.time() - start_time
    print(f"  Simulation finished in {elapsed:.2f}s.")

    # 6. Statistics
    gzip_stats = st.calculate_stats(obs_gzip, null_gzip_samples, smaller_is_better=True)
    sim_stats = st.calculate_stats(obs_similarity, null_sim_samples, smaller_is_better=False)

    # 7. Construct Result JSON
    results = {
        "meta": {
            "timestamp": timestamp,
            "eva_path": eva_path,
            "latin_path": latin_path,
            "n_iter": n_iter,
            "seed": seed,
            "modulus": decoder.MOD,
            "decoder": "Mod23Decoder_v1"
        },
        "metrics": {
            "gzip": {
                "observed": obs_gzip,
                "null_mean": gzip_stats["mean"],
                "null_std": gzip_stats["std"],
                "z_score": gzip_stats["z_score"],
                "p_value_smaller": gzip_stats["p_value"]
            },
            "trigram_cosine": {
                "observed": obs_similarity,
                "null_mean": sim_stats["mean"],
                "null_std": sim_stats["std"],
                "z_score": sim_stats["z_score"],
                "p_value_greater": sim_stats["p_value"]
            },
            "entropy": {
                "observed": obs_entropy
            },
            "index_of_coincidence": {
                "observed": obs_ioc
            }
        }
    }

    if not no_raw:
        results["metrics"]["gzip"]["null_samples"] = null_gzip_samples
        results["metrics"]["trigram_cosine"]["null_samples"] = null_sim_samples

    # 8. Save Results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    out_file = results_dir / f"run_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {out_file}")

    # 9. Plotting (Optional)
    if plot:
        try:
            import matplotlib.pyplot as plt

            # Gzip Histogram
            plt.figure(figsize=(10, 6))
            plt.hist(null_gzip_samples, bins=50, color='gray', alpha=0.7, label='Null (Shuffled Text)')
            plt.axvline(obs_gzip, color='red', linestyle='dashed', linewidth=2, label=f'Observed ({obs_gzip})')
            plt.title(f"Gzip Compression Size (Null vs Observed)\np={gzip_stats['p_value']:.5f}")
            plt.xlabel("Compressed Size (bytes)")
            plt.ylabel("Frequency")
            plt.legend()
            plot_path_gzip = results_dir / f"run_{timestamp}_gzip_hist.png"
            plt.savefig(plot_path_gzip)
            print(f"  Plot saved to {plot_path_gzip}")
            plt.close()

            # Similarity Histogram
            plt.figure(figsize=(10, 6))
            plt.hist(null_sim_samples, bins=50, color='gray', alpha=0.7, label='Null (Random Alphabet)')
            plt.axvline(obs_similarity, color='blue', linestyle='dashed', linewidth=2, label=f'Observed ({obs_similarity:.4f})')
            plt.title(f"Trigram Cosine Similarity to Latin (Null vs Observed)\np={sim_stats['p_value']:.5f}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            plt.legend()
            plot_path_sim = results_dir / f"run_{timestamp}_trigram_hist.png"
            plt.savefig(plot_path_sim)
            print(f"  Plot saved to {plot_path_sim}")
            plt.close()

        except ImportError:
            print("  Warning: Matplotlib not found. Skipping plots.")
        except Exception as e:
            print(f"  Error generating plots: {e}")

    # Print Summary to Console
    print("\n--- Summary ---")
    print(f"Entropy: {obs_entropy:.4f}")
    print(f"IoC:     {obs_ioc:.4f}")
    print(f"Gzip:    Obs={obs_gzip} | Null Mean={gzip_stats['mean']:.1f} | p={gzip_stats['p_value']:.5f}")
    print(f"Trigram: Obs={obs_similarity:.4f} | Null Mean={sim_stats['mean']:.4f} | p={sim_stats['p_value']:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Voynich Modular-23 Experiment CLI")
    parser.add_argument("--eva", default="data/voynich_eva_takahashi.txt", help="Path to EVA transcription")
    parser.add_argument("--latin", default="data/latin_reference.txt", help="Path to Latin reference")
    parser.add_argument("--n-iter", type=int, default=10000, help="Monte Carlo iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Generate histograms (requires matplotlib)")
    parser.add_argument("--no-raw", action="store_true", help="Do not save raw null samples in JSON")

    args = parser.parse_args()

    run_experiment(
        eva_path=args.eva,
        latin_path=args.latin,
        n_iter=args.n_iter,
        seed=args.seed,
        plot=args.plot,
        no_raw=args.no_raw
    )

if __name__ == "__main__":
    main()
