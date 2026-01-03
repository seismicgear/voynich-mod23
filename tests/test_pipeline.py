import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from experiment.cli import run_experiment

@pytest.fixture
def temp_env():
    # Create a temp dir
    old_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)

    # Create dummy data
    data_dir = Path("data")
    data_dir.mkdir()
    (data_dir / "voynich_eva_takahashi.txt").write_text("qokeedy q o")
    (data_dir / "latin_reference.txt").write_text("ABC DEF GHI")

    yield tmp_dir

    # Cleanup
    os.chdir(old_cwd)
    shutil.rmtree(tmp_dir)

def test_pipeline_smoke(temp_env):
    """
    Runs the full experiment with minimal iterations to check for crashes
    and output file existence.
    """
    run_experiment(
        eva_path="data/voynich_eva_takahashi.txt",
        latin_path="data/latin_reference.txt",
        n_iter=10,
        seed=42,
        plot=False,
        no_raw=False
    )

    # Check results created
    results_dir = Path("results")
    assert results_dir.exists()

    files = list(results_dir.glob("run_*.json"))
    assert len(files) == 1

    # Check JSON content
    with open(files[0]) as f:
        data = json.load(f)

    assert "metrics" in data
    assert "gzip" in data["metrics"]
    assert len(data["metrics"]["gzip"]["null_samples"]) == 10
