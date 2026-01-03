import pathlib
import pytest
from run_experiment import main

def test_experiment_smoke(tmp_path, monkeypatch):
    """
    Smoke test to ensure the experiment pipeline runs end-to-end
    without errors using a tiny toy corpus.
    """
    # Create dummy data files
    eva_file = tmp_path / "eva.txt"
    latin_file = tmp_path / "latin.txt"

    # Tiny toy EVA text
    eva_file.write_text("qo qo qo\nqokedy chedy\n")

    # Tiny Latin reference
    latin_file.write_text("AVEMARIAGLORIA")

    # Override arguments to main by calling it directly
    # We use a small n_iter to keep the test fast
    try:
        main(
            eva_path=str(eva_file),
            latin_path=str(latin_file),
            n_iter=10,
            seed=42
        )
    except Exception as e:
        pytest.fail(f"Experiment main() raised an exception: {e}")
