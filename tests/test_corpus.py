from voynich.corpus import clean_reference_text, load_lines


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
