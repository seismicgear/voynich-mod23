"""
data_loader.py
Parses 'interlinear_full_words.txt' to reconstruct Voynich lines
filtered by Currier Language (A or B).
"""
import csv
import re
import pathlib

def natural_sort_key(folio_str):
    """
    Sorts folios naturally: f1r < f1v < f2r < f10r
    Expected format: "f" + digits + "r"/"v"
    """
    # Extract number and side
    m = re.match(r'f(\d+)([rv]?)', folio_str)
    if m:
        page_num = int(m.group(1))
        side = m.group(2) # 'r' or 'v'
        # 'r' comes before 'v', which is naturally true lexicographically
        return (page_num, side)
    return (9999, folio_str) # Fallback

def load_voynich_lines(path: str, target_language: str = "A") -> list[list[str]]:
    """
    Loads lines from the interlinear CSV, filtering by language.

    Args:
        path: Path to interlinear_full_words.txt
        target_language: 'A' or 'B' (Currier Language)

    Returns:
        List of lists (lines of tokens).
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Store lines as: dictionary[(folio, line_num)] = [word, word, ...]
    manuscript_lines = {}

    with open(p, 'r', encoding='utf-8') as f:
        # Sniff delimiter (Comma or Tab)
        first_line = f.readline()
        f.seek(0)
        dialect = csv.Sniffer().sniff(first_line)

        reader = csv.reader(f, dialect)
        headers = next(reader) # Skip header

        # User specified columns:
        # word (0), folio (2), language (6), line_number (11)
        # We verify indices just in case, but rely on your spec if strict.

        for row in reader:
            if not row: continue

            # Robust parsing
            try:
                word = row[0].strip().replace('"', '')
                folio = row[2].strip()
                lang = row[6].strip()
                line_num = row[11].strip()

                # Filter by Language
                if lang != target_language:
                    continue

                # Filter out garbage / null words
                if not word or word == "null":
                    continue

                # Key for grouping
                key = (folio, line_num)

                if key not in manuscript_lines:
                    manuscript_lines[key] = []

                manuscript_lines[key].append(word)

            except IndexError:
                continue # Skip malformed rows

    # Sorting
    # We need to sort by Folio (Natural) then Line Number (Integer)
    sorted_keys = sorted(
        manuscript_lines.keys(),
        key=lambda k: (natural_sort_key(k[0]), float(k[1]) if k[1].replace('.','',1).isdigit() else 999)
    )

    # Compile final list
    ordered_lines = []
    for key in sorted_keys:
        words = manuscript_lines[key]
        # Final sanity check: remove non-EVA garbage if present
        valid_words = [w for w in words if re.match(r'^[a-z0-9]+$', w)]
        if valid_words:
            ordered_lines.append(valid_words)

    print(f"Loaded {len(ordered_lines)} lines for Language {target_language}.")
    return ordered_lines

if __name__ == "__main__":
    # Test
    try:
        lines = load_voynich_lines("data/interlinear_full_words.txt", "A")
        print(f"Sample line 1: {lines[0]}")
        print(f"Sample line 2: {lines[1]}")
    except Exception as e:
        print(e)
