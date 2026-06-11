import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def english_text() -> str:
    return (FIXTURES / "english_sample.txt").read_text()


@pytest.fixture(scope="session")
def interlinear_path() -> pathlib.Path:
    return FIXTURES / "interlinear_sample.txt"


@pytest.fixture
def offline_data_dir(tmp_path, english_text, interlinear_path, monkeypatch):
    """A data dir satisfying corpus.data_status() without network access."""
    from voynich import corpus

    (tmp_path / corpus.VOYNICH_FILE).write_text(interlinear_path.read_text())
    for lang in corpus.REFERENCE_SOURCES:
        (tmp_path / f"reference_{lang}.txt").write_text(english_text)
    monkeypatch.setattr(corpus, "DATA_DIR", tmp_path)
    return tmp_path
