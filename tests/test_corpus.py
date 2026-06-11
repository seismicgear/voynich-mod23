import pytest

from voynich.corpus import (
    FAMILY_ORDER,
    REFERENCE_SOURCES,
    _extract_bible_xml,
    _extract_morphgnt,
    clean_reference_text,
    load_lines,
    reference_catalog,
    transliterate,
)


def test_load_lines_orders_and_groups(interlinear_path):
    lines = load_lines(language="A", path=interlinear_path)
    assert len(lines) == 60
    assert lines[0].folio == "f1r"
    assert lines[0].words[0] == "fachys"
    # Natural folio order: f2r must not come before f1v
    folios = [l.folio for l in lines]
    assert folios.index("f1r") < folios.index("f1v")
    for line in lines:
        assert line.language == "A"
        assert all(w.isalpha() and w.islower() for w in line.words)


def test_load_lines_single_transcriber_no_duplicates(interlinear_path):
    lines = load_lines(language="A", path=interlinear_path)
    first = lines[0].words
    # The interlinear file repeats each line once per transcriber; with a
    # single transcriber pinned, a line must not contain itself twice.
    half = len(first) // 2
    assert first[:half] != first[half : 2 * half] or half == 0


def test_language_filter(interlinear_path):
    assert load_lines(language="B", path=interlinear_path) == []


def test_clean_reference_text():
    raw = (
        "junk *** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "Hæc Vita, præcépta — 123!\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X *** junk"
    )
    assert clean_reference_text(raw) == "haec vita praecepta"


def test_clean_handles_non_decomposing_letters():
    # ß and ł do not decompose under NFKD; they need explicit folds.
    assert clean_reference_text("Groß słowo") == "gross slowo"


def test_greek_transliteration_is_one_letter_per_letter():
    # λόγος -> logos; θεός -> qeos (theta is ONE letter, never 'th')
    assert transliterate("λόγος", "greek") == "logos"
    assert transliterate("Θεός ψυχή", "greek") == "qeos yuch"
    # polytonic accents and final sigma
    assert transliterate("ἄνθρωπος", "greek") == "anqrwpos"


def test_cyrillic_transliteration():
    assert transliterate("живёт", "cyrillic") == "jivet"
    assert transliterate("Человѣкъ", "cyrillic") == "qelovek"  # pre-reform yat
    assert transliterate("щука шёл", "cyrillic") == "wuka wel"


def test_reference_registry_covers_requested_families():
    families = {r.family for r in REFERENCE_SOURCES.values()}
    assert {
        "Romance", "Celtic", "Germanic", "Slavic", "Hellenic", "Uralic",
        "Turkic", "Iranian", "Indo-Aryan", "Semitic", "Cushitic", "Sinitic",
        "Isolate",
    } <= families
    assert set(FAMILY_ORDER) >= families
    for expected in ("latin", "french", "spanish", "portuguese", "czech",
                     "polish", "russian", "basque", "greek", "hungarian",
                     "finnish", "german", "dutch", "catalan",
                     "welsh", "irish", "icelandic", "albanian", "turkish",
                     "persian", "arabic", "hebrew", "amharic", "somali",
                     "hindi", "chinese"):
        assert expected in REFERENCE_SOURCES
    catalog = reference_catalog()
    n_items = sum(len(g["items"]) for g in catalog)
    assert n_items == len(REFERENCE_SOURCES)


def test_arabic_transliteration():
    assert transliterate("سلام", "arabic") == "slam"
    # harakat (combining vowel marks) are stripped
    assert transliterate("بِسْمِ", "arabic") == "bsm"


def test_hebrew_transliteration():
    # shalom: shin -> w, lamed -> l, vav -> u, final mem -> m
    assert transliterate("שלום", "hebrew") == "wlum"
    # niqqud stripped, final forms folded
    assert transliterate("מֶלֶךְ", "hebrew") == "mlk"


def test_ethiopic_transliteration():
    # Unicode names carry the romanization; qualifier words are dropped
    assert transliterate("ሀ", "ethiopic") == "ha"
    assert transliterate("አ", "ethiopic") == "a"  # GLOTTAL A -> a


def test_devanagari_transliteration():
    # ka + vowel sign i -> ki (inherent a replaced)
    assert transliterate("कि", "devanagari") == "ki"
    # ka + virama -> k (inherent a deleted)
    assert transliterate("क्", "devanagari") == "k"
    assert transliterate("नमस्ते", "devanagari") == "namaste"


def test_chinese_transliteration():
    pytest.importorskip("pypinyin")
    out = transliterate("光", "chinese")
    assert out == "guang"


def test_bible_xml_extractor():
    raw = "<x><seg id='1'>In principio</seg> junk <seg>creavit Deus</seg></x>"
    assert _extract_bible_xml(raw) == "In principio creavit Deus"


def test_morphgnt_extractor():
    raw = (
        "010101 N- ----NSF- Βίβλος Βίβλος βίβλος βίβλος\n"
        "010101 N- ----GSF- γενέσεως, γενέσεως γενέσεως γένεσις\n"
        "short line\n"
    )
    assert _extract_morphgnt(raw) == "Βίβλος γενέσεως"
