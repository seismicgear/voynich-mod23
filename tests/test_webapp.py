import time

import pytest

from voynich.webapp import create_app


@pytest.fixture
def client(offline_data_dir):
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _wait_for_run(client, run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").get_json()
        if run["status"] != "running":
            return run
        time.sleep(0.3)
    pytest.fail("run did not finish in time")


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Voynich Decipherment Workbench" in resp.data


def test_data_status(client):
    status = client.get("/api/data/status").get_json()
    assert status["ready"] is True
    assert status["voynich"] is True


def test_run_validation_errors(client):
    resp = client.post("/api/runs", json={"kind": "solve", "config": {"reference": "klingon"}})
    assert resp.status_code == 400
    resp = client.post("/api/runs", json={"kind": "nonsense", "config": {}})
    assert resp.status_code == 400
    resp = client.get("/api/runs/999")
    assert resp.status_code == 404


def test_benchmark_run_lifecycle(client):
    resp = client.post(
        "/api/runs",
        json={
            "kind": "benchmark",
            "config": {
                "reference": "english",
                "cipher_chars": 1500,
                "iterations": 4000,
                "restarts": 1,
                "order": 3,
                "seed": 5,
            },
        },
    )
    assert resp.status_code == 201
    run_id = resp.get_json()["id"]

    run = _wait_for_run(client, run_id)
    assert run["status"] == "done", run.get("error")
    assert run["result"]["accuracy"] > 0.5
    assert len(run["history"]) > 0

    runs = client.get("/api/runs").get_json()
    assert any(r["id"] == run_id for r in runs)


def test_solve_run_lifecycle(client, tmp_path, monkeypatch):
    import voynich.pipeline as pipeline

    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path / "results")
    resp = client.post(
        "/api/runs",
        json={
            "kind": "solve",
            "config": {
                "currier_language": "A",
                "reference": "english",
                "hypothesis": "simple",
                "order": 3,
                "bpe_merges": 10,
                "iterations": 2000,
                "restarts": 1,
                "seed": 5,
            },
        },
    )
    assert resp.status_code == 201
    run_id = resp.get_json()["id"]

    run = _wait_for_run(client, run_id)
    assert run["status"] == "done", run.get("error")
    result = run["result"]
    assert "scores" in result and "verdict" in result
    assert result["scores"]["test_heldout"] > result["scores"]["random_key_floor"]
    assert len(result["decoded_sample"]) > 0
    assert (tmp_path / "results").exists()


def test_stop_endpoint(client):
    resp = client.post(
        "/api/runs",
        json={
            "kind": "benchmark",
            "config": {
                "reference": "english",
                "cipher_chars": 2000,
                "iterations": 500000,
                "restarts": 1,
                "order": 3,
            },
        },
    )
    run_id = resp.get_json()["id"]
    time.sleep(0.5)
    assert client.post(f"/api/runs/{run_id}/stop").get_json()["ok"] is True
    run = _wait_for_run(client, run_id)
    assert run["status"] in ("stopped", "done")
