from run_experiment import load_latin


def test_load_latin_removes_non_letters(tmp_path):
    latin_file = tmp_path / "latin.txt"
    latin_file.write_text("Ave, Maria! 123\nGloria.")

    loaded = load_latin(str(latin_file))

    assert loaded == "AVEMARIAGLORIA"
