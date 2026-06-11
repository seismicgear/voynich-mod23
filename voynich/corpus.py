"""
corpus.py — data acquisition and parsing.

Voynich text: the "interlinear" word-level transcription compiled by
Claire Bowern's group (chirila/Voynich-public), which carries folio,
section, Currier-language and line metadata for every word.  The file
interleaves up to 18 transcribers per line; we pin to Takahashi ("H"),
the most complete single transcription, so lines are not duplicated.

Reference corpora (cleaned to lowercase a-z + spaces):
  latin    Thomas à Kempis, De Imitatione Christi (~1420s — contemporary
           with the manuscript's carbon dating) + Caesar, De Bello Gallico
  italian  Dante, La Divina Commedia (14th c.)
  english  King James Bible (1611)
"""

from __future__ import annotations

import csv
import os
import pathlib
import re
import unicodedata
import urllib.request
from dataclasses import dataclass

DATA_DIR = pathlib.Path(os.environ.get("VOYNICH_DATA_DIR", "data"))

VOYNICH_URL = (
    "https://raw.githubusercontent.com/chirila/Voynich-public/master/"
    "Corpora/Voynich_texts/interlinear_full_words.txt"
)
VOYNICH_FILE = "interlinear_full_words.txt"

_CLTK = "https://raw.githubusercontent.com/cltk/lat_text_latin_library/master"
REFERENCE_SOURCES: dict[str, list[str]] = {
    "latin": [
        f"{_CLTK}/kempis/kempis1.txt",
        f"{_CLTK}/kempis/kempis2.txt",
        f"{_CLTK}/kempis/kempis3.txt",
        f"{_CLTK}/kempis/kempis4.txt",
        f"{_CLTK}/caesar/gall1.txt",
    ],
    "italian": ["https://www.gutenberg.org/cache/epub/1012/pg1012.txt"],
    "english": ["https://www.gutenberg.org/cache/epub/10/pg10.txt"],
}

SECTIONS = {
    "H": "Herbal",
    "S": "Stars (recipes)",
    "B": "Biological",
    "P": "Pharmaceutical",
    "C": "Cosmological",
    "T": "Text-only",
    "Z": "Zodiac",
    "A": "Astronomical",
}


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def clean_reference_text(raw: str) -> str:
    """Strip Gutenberg boilerplate, fold accents to ASCII, keep a-z and
    word boundaries."""
    start = re.search(r"\*\*\* START OF.*?\*\*\*", raw)
    if start:
        raw = raw[start.end():]
    end = re.search(r"\*\*\* END OF.*?\*\*\*", raw)
    if end:
        raw = raw[: end.start()]
    # Fold accents (à -> a) and split ligatures (æ -> ae).
    raw = raw.replace("æ", "ae").replace("Æ", "AE").replace("œ", "oe")
    raw = unicodedata.normalize("NFKD", raw)
    raw = raw.encode("ascii", "ignore").decode("ascii").lower()
    # Classical 'v' for 'u' (IVLI -> iuli) is left alone: both letters
    # exist in the alphabet and the LM learns whatever convention the
    # corpus uses consistently.
    raw = re.sub(r"[^a-z]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


# ---- download management -------------------------------------------------

def voynich_path() -> pathlib.Path:
    return DATA_DIR / VOYNICH_FILE


def reference_path(language: str) -> pathlib.Path:
    return DATA_DIR / f"reference_{language}.txt"


def data_status() -> dict:
    status = {
        "voynich": voynich_path().exists(),
        "references": {
            lang: reference_path(lang).exists() for lang in REFERENCE_SOURCES
        },
    }
    status["ready"] = status["voynich"] and all(status["references"].values())
    return status


def ensure_voynich(force: bool = False) -> pathlib.Path:
    path = voynich_path()
    if force or not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_fetch(VOYNICH_URL))
    return path


def ensure_reference(language: str, force: bool = False) -> pathlib.Path:
    if language not in REFERENCE_SOURCES:
        raise ValueError(f"unknown reference language: {language}")
    path = reference_path(language)
    if force or not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        parts = []
        for url in REFERENCE_SOURCES[language]:
            raw = _fetch(url).decode("utf-8", errors="replace")
            parts.append(clean_reference_text(raw))
        path.write_text(" ".join(parts))
    return path


def ensure_all(force: bool = False) -> dict:
    ensure_voynich(force)
    for lang in REFERENCE_SOURCES:
        ensure_reference(lang, force)
    return data_status()


def load_reference(language: str) -> str:
    return ensure_reference(language).read_text()


# ---- Voynich parsing -----------------------------------------------------

@dataclass
class VoynichLine:
    folio: str
    line_number: str
    section: str
    language: str  # Currier language: 'A', 'B' or 'NA'
    words: list[str]


_WORD_RE = re.compile(r"^[a-z]+$")


def _folio_sort_key(folio: str):
    m = re.match(r"f(\d+)([rv]?)(\d*)", folio)
    if m:
        return (int(m.group(1)), m.group(2), int(m.group(3) or 0))
    return (10**6, folio, 0)


def load_lines(
    language: str | None = "A",
    section: str | None = None,
    transcriber: str = "H",
    path: pathlib.Path | None = None,
) -> list[VoynichLine]:
    """Parse the interlinear file into ordered lines of EVA words.

    language: Currier language filter ('A', 'B'), or None/'all' for both.
    section:  single-letter section code (see SECTIONS), or None for all.
    transcriber: which transcription to use; 'H' (Takahashi) is the most
                 complete.  Using one transcriber avoids the line
                 duplication inherent to the interlinear format.
    """
    path = path or ensure_voynich()
    if language in (None, "all", "ALL"):
        language = None

    grouped: dict[tuple, VoynichLine] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_word = col["word"]
        i_folio = col["folio"]
        i_section = col["section"]
        i_lang = col["language"]
        i_line = col["line_number"]
        i_trans = col["transcriber"]

        for row in reader:
            if len(row) <= i_trans:
                continue
            if row[i_trans] != transcriber:
                continue
            lang = row[i_lang]
            if language is not None and lang != language:
                continue
            sec = row[i_section]
            if section is not None and sec != section:
                continue
            word = row[i_word].strip().lower()
            if not _WORD_RE.match(word):
                continue
            key = (row[i_folio], row[i_line])
            line = grouped.get(key)
            if line is None:
                line = VoynichLine(
                    folio=row[i_folio],
                    line_number=row[i_line],
                    section=sec,
                    language=lang,
                    words=[],
                )
                grouped[key] = line
            line.words.append(word)

    def line_sort_key(item):
        (folio, line_no), _ = item
        try:
            ln = float(line_no)
        except ValueError:
            ln = 10**6
        return (_folio_sort_key(folio), ln)

    return [line for _, line in sorted(grouped.items(), key=line_sort_key)]
