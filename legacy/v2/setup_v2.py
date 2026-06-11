"""
setup_v2.py
Downloads the rich metadata Voynich file and NLTK corpora.
"""
import pathlib
import urllib.request
import nltk
from nltk.corpus import brown, udhr

# Using the URL found in previous investigation
VOYNICH_URL = "https://raw.githubusercontent.com/chirila/Voynich-public/master/Corpora/Voynich_texts/interlinear_full_words.txt"

def main():
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)

    print("--- 1. Downloading Voynich Data ---")
    dest = data_dir / "interlinear_full_words.txt"
    print(f"Fetching {dest.name}...")
    try:
        # User-agent sometimes needed for raw GitHub/Academic sites
        req = urllib.request.Request(
            VOYNICH_URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            data = response.read()
            dest.write_bytes(data)
        print(f"  -> Saved to {dest} ({len(data)/1024:.1f} KB)")
    except Exception as e:
        print(f"  -> Error downloading: {e}")
        print("  -> Please manually place 'interlinear_full_words.txt' in the data/ folder.")

    print("\n--- 2. Downloading NLTK Corpora ---")
    nltk.download('brown', quiet=True)
    nltk.download('udhr', quiet=True)

    # 1. English (Brown Corpus)
    print("Generating English baseline...")
    english_text = " ".join(brown.words()[:40000]).upper()
    english_clean = "".join([c for c in english_text if c.isalpha()])
    (data_dir / "english_brown.txt").write_text(english_clean)

    # 2. Italian (UDHR)
    print("Generating Italian baseline...")
    # 'Italian_Italian-Latin1' might be different in different NLTK versions, use 'Italian_Italiano-Latin1'
    italian_words = udhr.words('Italian_Italiano-Latin1')
    italian_text = "".join([w for w in italian_words if w.isalpha()]).upper()
    italian_full = (italian_text * 10)[:40000]
    (data_dir / "italian_sample.txt").write_text(italian_full)

    print("\n[+] Setup complete. Ready for Data Loading.")

if __name__ == "__main__":
    main()
