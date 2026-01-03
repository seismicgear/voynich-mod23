"""
tokenize_eva.py
Learns a vocabulary from the Voynich text using simplified BPE.
"""
import collections
import re
from pathlib import Path
from data_loader import load_voynich_lines

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def learn_vocabulary(lines: list[list[str]], num_merges: int = 25) -> list[str]:
    # 1. Initialize Vocab
    vocab = collections.defaultdict(int)
    for line in lines:
        for word in line:
            # Treat every character as a token initially
            # e.g. "chedy" -> "c h e d y"
            chars = " ".join(list(word))
            vocab[chars] += 1

    # 2. Iteratively Merge
    print(f"Learning vocabulary ({num_merges} merges)...")
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        print(f"  Merge {i+1}: {best} -> {''.join(best)}")
        vocab = merge_vocab(best, vocab)

    # 3. Extract Tokens
    final_tokens = set()
    for word in vocab:
        for token in word.split():
            final_tokens.add(token)

    return sorted(list(final_tokens))

if __name__ == "__main__":
    # We learn from Voynich A as the primary source
    try:
        lines = load_voynich_lines("data/interlinear_full_words.txt", "A")
        tokens = learn_vocabulary(lines, num_merges=25)

        out_path = Path("data/vocab_a.txt")
        out_path.write_text("\n".join(tokens))
        print(f"\n[+] Saved {len(tokens)} tokens to {out_path}")
        print(f"Top tokens: {tokens[:10]}...")
    except FileNotFoundError:
        print("Error: data/interlinear_full_words.txt not found. Run setup_v2.py first.")
