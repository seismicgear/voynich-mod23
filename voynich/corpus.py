"""
corpus.py — data acquisition and parsing.

Voynich text: the "interlinear" word-level transcription compiled by
Claire Bowern's group (chirila/Voynich-public), which carries folio,
section, Currier-language and line metadata for every word.  The file
interleaves up to 18 transcribers per line; we pin to Takahashi ("H"),
the most complete single transcription, so lines are not duplicated.

Reference corpora cover the plausible 15th-century European candidates:
Romance, Germanic, Slavic, Hellenic, Uralic families plus Basque.  Each
source is labelled with its period; where a genuinely medieval text is
downloadable we use it (Kempis ~1420s, Froissart ~1370s, Kralická 1613,
Leizarraga 1571), otherwise the earliest robust text available.

All corpora are reduced to the solver's 26-letter alphabet.  Greek and
Cyrillic are transliterated with ONE-letter-per-letter tables — a
multi-letter romanization (theta -> "th") would smuggle fake digraph
statistics into the language model and corrupt the substitution
hypothesis being tested.
"""

from __future__ import annotations

import csv
import os
import pathlib
import re
import unicodedata
import urllib.request
from dataclasses import dataclass

from .paths import default_data_dir

DATA_DIR = default_data_dir()

VOYNICH_URL = (
    "https://raw.githubusercontent.com/chirila/Voynich-public/master/"
    "Corpora/Voynich_texts/interlinear_full_words.txt"
)
VOYNICH_FILE = "interlinear_full_words.txt"

_CLTK = "https://raw.githubusercontent.com/cltk/lat_text_latin_library/master"
_BIBLES = "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles"
_MORPHGNT = "https://raw.githubusercontent.com/morphgnt/sblgnt/master"
_PG = "https://www.gutenberg.org/cache/epub"


@dataclass(frozen=True)
class Reference:
    key: str
    label: str          # human-readable source + period
    family: str         # language family for grouping in the UI
    urls: tuple[str, ...]
    fmt: str = "gutenberg"   # gutenberg | text | bible_xml | morphgnt
    script: str = "latin"    # latin | greek | cyrillic


_REFERENCES = [
    # ---- Italic / Romance ------------------------------------------------
    Reference(
        "latin", "Latin — De Imitatione Christi (~1420s) + De Bello Gallico",
        "Italic",
        (f"{_CLTK}/kempis/kempis1.txt", f"{_CLTK}/kempis/kempis2.txt",
         f"{_CLTK}/kempis/kempis3.txt", f"{_CLTK}/kempis/kempis4.txt",
         f"{_CLTK}/caesar/gall1.txt"),
        fmt="text",
    ),
    Reference(
        "italian", "Italian — Dante, Divina Commedia (c. 1320)", "Romance",
        (f"{_PG}/1012/pg1012.txt",),
    ),
    Reference(
        "french", "Middle French — Froissart, Chroniques (c. 1370–1400)",
        "Romance", (f"{_PG}/50356/pg50356.txt",),
    ),
    Reference(
        "spanish", "Spanish — Cervantes, Don Quijote (1605)", "Romance",
        (f"{_PG}/2000/pg2000.txt",),
    ),
    Reference(
        "portuguese", "Portuguese — Camões, Os Lusíadas (1572)", "Romance",
        (f"{_PG}/3333/pg3333.txt",),
    ),
    Reference(
        "catalan", "Catalan — La orfaneta de Menargues (19th c. literary)",
        "Romance", (f"{_PG}/75136/pg75136.txt",),
    ),
    # ---- Germanic ---------------------------------------------------------
    Reference(
        "english", "English — King James Bible (1611)", "Germanic",
        (f"{_PG}/10/pg10.txt",),
    ),
    Reference(
        "german", "German — Luther Bible tradition (16th c. base)", "Germanic",
        (f"{_BIBLES}/German.xml",), fmt="bible_xml",
    ),
    Reference(
        "dutch", "Dutch — Bible (modern translation)", "Germanic",
        (f"{_BIBLES}/Dutch.xml",), fmt="bible_xml",
    ),
    # ---- Slavic -----------------------------------------------------------
    Reference(
        "czech", "Czech — Bible kralická (1613)", "Slavic",
        (f"{_BIBLES}/Czech.xml",), fmt="bible_xml",
    ),
    Reference(
        "polish", "Polish — Mickiewicz, Pan Tadeusz (1834)", "Slavic",
        (f"{_PG}/31536/pg31536.txt",),
    ),
    Reference(
        "russian", "Russian — Synodal Bible (1876)", "Slavic",
        (f"{_BIBLES}/Russian.xml",), fmt="bible_xml", script="cyrillic",
    ),
    # ---- Hellenic ---------------------------------------------------------
    Reference(
        "greek", "Greek — Koine New Testament (Byzantine standard)", "Hellenic",
        tuple(
            f"{_MORPHGNT}/{book}-morphgnt.txt"
            for book in ("61-Mt", "62-Mk", "63-Lk", "64-Jn", "65-Ac",
                         "66-Ro", "67-1Co", "79-Heb", "87-Re")
        ),
        fmt="morphgnt", script="greek",
    ),
    # ---- Uralic -----------------------------------------------------------
    Reference(
        "hungarian", "Hungarian — Jókai, Az arany ember (1872)", "Uralic",
        (f"{_PG}/56591/pg56591.txt", f"{_PG}/56592/pg56592.txt"),
    ),
    Reference(
        "finnish", "Finnish — Kalevala (1849, archaic oral tradition)", "Uralic",
        (f"{_PG}/7000/pg7000.txt",),
    ),
    # ---- Celtic (the Italo-Celtic / Byblos-theory test bed) ---------------
    Reference(
        "welsh", "Welsh — O.M. Edwards & contemporaries (19th c. literary)",
        "Celtic",
        (f"{_PG}/3680/pg3680.txt", f"{_PG}/57981/pg57981.txt",
         f"{_PG}/67424/pg67424.txt"),
    ),
    Reference(
        "irish", "Irish — Ua Laoghaire, Niamh (Munster Irish, 1907)", "Celtic",
        (f"{_PG}/50913/pg50913.txt",),
    ),
    # ---- Germanic periphery -------------------------------------------------
    Reference(
        "icelandic", "Icelandic — 19th c. translations (closest to Old Norse)",
        "Germanic",
        (f"{_PG}/17025/pg17025.txt", f"{_PG}/16846/pg16846.txt"),
    ),
    # ---- Albanian -----------------------------------------------------------
    Reference(
        "albanian", "Albanian — Bible (Adriatic / Venetian Albania)",
        "Albanian", (f"{_BIBLES}/Albanian.xml",), fmt="bible_xml",
    ),
    # ---- Turkic / Iranian (Silk Road, Ottoman sphere) ----------------------
    Reference(
        "turkish", "Turkish — Bible (Ottoman-era contact language)",
        "Turkic", (f"{_BIBLES}/Turkish.xml",), fmt="bible_xml",
    ),
    Reference(
        "persian", "Persian — Bible (Silk Road lingua franca)",
        "Iranian", (f"{_BIBLES}/Farsi.xml",), fmt="bible_xml", script="arabic",
    ),
    # ---- Semitic (Mediterranean and Red Sea trade) -------------------------
    Reference(
        "arabic", "Arabic — Bible (Mediterranean trade lingua franca)",
        "Semitic", (f"{_BIBLES}/Arabic.xml",), fmt="bible_xml", script="arabic",
    ),
    Reference(
        "hebrew", "Hebrew — Bible (Hauer & Kondrak's top candidate)",
        "Semitic", (f"{_BIBLES}/Hebrew.xml",), fmt="bible_xml", script="hebrew",
    ),
    Reference(
        "amharic", "Amharic — Bible (Ethiopia; embassy to Rome 1441)",
        "Semitic", (f"{_BIBLES}/Amharic.xml",), fmt="bible_xml",
        script="ethiopic",
    ),
    # ---- African (Indian Ocean trade coast) --------------------------------
    Reference(
        "somali", "Somali — Bible (Mogadishu / Indian Ocean trade)",
        "Cushitic", (f"{_BIBLES}/Somali.xml",), fmt="bible_xml",
    ),
    # ---- Asian (Silk Road / Indian Ocean) ----------------------------------
    Reference(
        "hindi", "Hindi — Bible (Indo-Aryan; Indian Ocean trade)",
        "Indo-Aryan", (f"{_BIBLES}/Hindi.xml",), fmt="bible_xml",
        script="devanagari",
    ),
    Reference(
        "chinese", "Chinese — Bible as toneless pinyin syllables (Stolfi's comparison)",
        "Sinitic", (f"{_BIBLES}/Chinese.xml",), fmt="bible_xml",
        script="chinese",
    ),
    # ---- Isolate ----------------------------------------------------------
    Reference(
        "basque", "Basque — Leizarraga New Testament (1571)", "Isolate",
        (f"{_BIBLES}/Basque-NT.xml",), fmt="bible_xml",
    ),
]

REFERENCE_SOURCES: dict[str, Reference] = {r.key: r for r in _REFERENCES}

FAMILY_ORDER = [
    "Italic", "Romance", "Celtic", "Germanic", "Slavic", "Hellenic",
    "Albanian", "Uralic", "Turkic", "Iranian", "Indo-Aryan", "Semitic",
    "Cushitic", "Sinitic", "Isolate",
]


def reference_catalog() -> list[dict]:
    """References grouped by family, in display order, for the GUI."""
    groups = []
    for family in FAMILY_ORDER:
        items = [
            {"key": r.key, "label": r.label}
            for r in _REFERENCES
            if r.family == family
        ]
        if items:
            groups.append({"family": family, "items": items})
    return groups


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


# ---- transliteration -------------------------------------------------------

# One Latin letter per Greek letter (24 letters onto 26 slots).  Eta -> h,
# theta -> q, chi -> c, psi -> y, omega -> w; the rest are conventional.
GREEK_TO_LATIN = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "h",
    "θ": "q", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
    "φ": "f", "χ": "c", "ψ": "y", "ω": "w",
}

# One Latin letter per Cyrillic letter where possible.  Rare/soft letters
# are merged (yu -> u, ya -> a, shcha -> sha) and hard/soft signs dropped;
# pre-reform letters (yat, fita, izhitsa, i-decimal) fold to their modern
# equivalents.  zh -> j, ts -> c, ch -> q, sh/shch -> w, kh -> h.
CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "j", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "q", "ш": "w", "щ": "w", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "u", "я": "a",
    # pre-1918 orthography and Church Slavonic extras
    "ѣ": "e", "і": "i", "ѳ": "f", "ѵ": "i", "ѕ": "z", "є": "e",
    "ѫ": "u", "ѭ": "u", "ѧ": "a", "ѩ": "a",
}

# One Latin letter per Arabic letter where possible; emphatic consonants
# merge with their plain counterparts (s./s, t./t, d./d, z./dh/z) and the
# Persian additions (pe/che/zhe/gaf) are included.  kh -> x, sh -> w,
# ghayn -> g, 'ayn -> e, qaf -> q.  Harakat are combining marks and are
# stripped automatically.
ARABIC_TO_LATIN = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ء": "", "ؤ": "u", "ئ": "i",
    "ب": "b", "ت": "t", "ث": "c", "ج": "j", "ح": "h", "خ": "x",
    "د": "d", "ذ": "z", "ر": "r", "ز": "z", "س": "s", "ش": "w",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "e", "غ": "g",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ة": "t", "و": "u", "ي": "i", "ى": "a", "ـ": "",
    # Persian
    "پ": "p", "چ": "c", "ژ": "j", "گ": "g", "ک": "k", "ی": "i",
}

# One Latin letter per Hebrew letter (final forms folded); niqqud are
# combining marks and are stripped automatically.  het -> x, tsadi -> c,
# shin -> w, ayin -> e, vav -> u, yod -> i.
HEBREW_TO_LATIN = {
    "א": "a", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "u",
    "ז": "z", "ח": "x", "ט": "t", "י": "i", "כ": "k", "ך": "k",
    "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s",
    "ע": "e", "פ": "p", "ף": "p", "צ": "c", "ץ": "c", "ק": "q",
    "ר": "r", "ש": "w", "ת": "t", "־": " ",
}

# Latin-script letters that NFKD does not decompose to ASCII.
_LATIN_FOLDS = {
    "ß": "ss", "æ": "ae", "œ": "oe", "ł": "l", "ø": "o", "đ": "d",
    "þ": "t", "ð": "d", "ŋ": "n", "ı": "i",
}

# Devanagari dependent vowel signs (which REPLACE the consonant's
# inherent 'a') and other signs, by the last word of the Unicode name.
_DEVANAGARI_VOWEL_SIGNS = {
    "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e",
    "AI": "ai", "O": "o", "AU": "au", "R": "r", "RR": "r", "L": "l",
}


def _name_syllable(ch: str, prefix: str) -> str | None:
    """Romanization embedded in a Unicode character name, e.g.
    'ETHIOPIC SYLLABLE QA' -> 'qa'.  Only the last word of the name is
    the syllable — earlier words are phonetic qualifiers ('ETHIOPIC
    SYLLABLE GLOTTAL A' is just 'a')."""
    name = unicodedata.name(ch, "")
    if name.startswith(prefix):
        tail = name[len(prefix):].strip().lower().split()[-1]
        return re.sub(r"[^a-z]", "", tail)
    return None


def _transliterate_ethiopic(text: str) -> str:
    """Ethiopic fidel -> Latin CV syllables via Unicode names (the names
    carry the standard romanization)."""
    out = []
    cache: dict[str, str] = {}
    for ch in text:
        mapped = cache.get(ch)
        if mapped is None:
            syl = _name_syllable(ch, "ETHIOPIC SYLLABLE")
            if syl is not None:
                mapped = syl
            elif ch.isspace() or unicodedata.category(ch).startswith("P"):
                mapped = " "
            else:
                mapped = " "
            cache[ch] = mapped
        out.append(mapped)
    return "".join(out)


def _transliterate_devanagari(text: str) -> str:
    """Devanagari -> Latin via Unicode names, handling the abugida's
    inherent vowel: consonants carry 'a', dependent vowel signs replace
    it, virama deletes it."""
    out: list[str] = []

    def drop_inherent_a() -> None:
        if out and out[-1].endswith("a") and len(out[-1]) > 1:
            out[-1] = out[-1][:-1]

    for ch in text:
        name = unicodedata.name(ch, "")
        if name.startswith("DEVANAGARI LETTER "):
            tail = name[len("DEVANAGARI LETTER "):].strip().lower()
            tail = re.sub(r"[^a-z]", "", tail)
            # Consonant names have no vowel ending in the name itself
            # (KA includes the inherent a already); independent vowels
            # (A, AA, I, ...) come through as-is.
            out.append(tail)
        elif name.startswith("DEVANAGARI VOWEL SIGN "):
            sign = name[len("DEVANAGARI VOWEL SIGN "):].strip()
            drop_inherent_a()
            out.append(_DEVANAGARI_VOWEL_SIGNS.get(
                sign, re.sub(r"[^a-z]", "", sign.lower())))
        elif name == "DEVANAGARI SIGN VIRAMA":
            drop_inherent_a()
        elif name == "DEVANAGARI SIGN ANUSVARA":
            out.append("n")
        elif name == "DEVANAGARI SIGN VISARGA":
            out.append("h")
        elif "a" <= ch.lower() <= "z":
            out.append(ch.lower())
        else:
            out.append(" ")
    return "".join(out)


def _transliterate_chinese(text: str) -> str:
    """Chinese characters -> toneless pinyin syllables, one syllable per
    'word' (Stolfi's comparison of Voynichese word grammar to Mandarin
    syllable structure operates at exactly this level)."""
    try:
        from pypinyin import lazy_pinyin
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the Chinese reference needs the 'pypinyin' package "
            "(pip install pypinyin)"
        ) from exc
    return " ".join(lazy_pinyin(text))


def transliterate(text: str, script: str) -> str:
    """Map non-Latin text to the solver's Latin alphabet, one letter per
    letter where the source script allows it.  Unmapped characters
    become spaces."""
    if script == "latin":
        return text
    if script == "chinese":
        return _transliterate_chinese(text)
    if script == "ethiopic":
        return _transliterate_ethiopic(text)
    if script == "devanagari":
        return _transliterate_devanagari(text)
    table = {
        "greek": GREEK_TO_LATIN,
        "cyrillic": CYRILLIC_TO_LATIN,
        "arabic": ARABIC_TO_LATIN,
        "hebrew": HEBREW_TO_LATIN,
    }[script]
    # NFD strips Greek tonos/polytonic accents and Cyrillic stress marks
    # into combining characters, which the loop below skips.
    decomposed = unicodedata.normalize("NFD", text.lower())
    out = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        mapped = table.get(ch)
        if mapped is not None:
            out.append(mapped)
        elif ch.isspace():
            out.append(" ")
        elif "a" <= ch <= "z":
            out.append(ch)  # already-Latin passages (titles, numbers context)
        else:
            out.append(" ")
    return "".join(out)


# ---- cleaning ---------------------------------------------------------------

def _strip_gutenberg(raw: str) -> str:
    start = re.search(r"\*\*\* START OF.*?\*\*\*", raw)
    if start:
        raw = raw[start.end():]
    end = re.search(r"\*\*\* END OF.*?\*\*\*", raw)
    if end:
        raw = raw[: end.start()]
    return raw


def _extract_bible_xml(raw: str) -> str:
    """Verse text from a bible-corpus XML file."""
    return " ".join(re.findall(r"<seg[^>]*>([^<]+)</seg>", raw))


def _extract_morphgnt(raw: str) -> str:
    """Word forms (column 5: punctuation-stripped) from MorphGNT files."""
    words = []
    for line in raw.splitlines():
        cols = line.split()
        if len(cols) >= 7:
            words.append(cols[4])
    return " ".join(words)


def clean_reference_text(raw: str, script: str = "latin") -> str:
    """Strip Gutenberg boilerplate, transliterate if needed, fold accents
    to ASCII, keep a-z and word boundaries."""
    raw = _strip_gutenberg(raw)
    raw = transliterate(raw, script)
    for src, dst in _LATIN_FOLDS.items():
        raw = raw.replace(src, dst).replace(src.upper(), dst.upper())
    raw = unicodedata.normalize("NFKD", raw)
    raw = raw.encode("ascii", "ignore").decode("ascii").lower()
    raw = re.sub(r"[^a-z]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def _clean_source(raw: str, ref: Reference) -> str:
    if ref.fmt == "bible_xml":
        raw = _extract_bible_xml(raw)
    elif ref.fmt == "morphgnt":
        raw = _extract_morphgnt(raw)
    elif ref.fmt == "gutenberg":
        raw = _strip_gutenberg(raw)
    return clean_reference_text(raw, ref.script)


# ---- download management ----------------------------------------------------

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
    ref = REFERENCE_SOURCES.get(language)
    if ref is None:
        raise ValueError(f"unknown reference language: {language}")
    path = reference_path(language)
    if force or not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        parts = []
        for url in ref.urls:
            raw = _fetch(url).decode("utf-8", errors="replace")
            parts.append(_clean_source(raw, ref))
        cleaned = " ".join(parts)
        if len(cleaned) < 50_000:
            raise RuntimeError(
                f"reference '{language}' cleaned to only {len(cleaned)} chars "
                "— source layout may have changed"
            )
        path.write_text(cleaned)
    return path


def ensure_all(force: bool = False) -> dict:
    ensure_voynich(force)
    errors = {}
    for lang in REFERENCE_SOURCES:
        try:
            ensure_reference(lang, force)
        except Exception as exc:  # keep going; report what failed
            errors[lang] = str(exc)
    status = data_status()
    if errors:
        status["errors"] = errors
    return status


def load_reference(language: str) -> str:
    return ensure_reference(language).read_text()


# ---- Voynich parsing ----------------------------------------------------

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
