"""
app.py — Flask routes for the workbench GUI.
"""

from __future__ import annotations

import threading

from flask import Flask, jsonify, render_template, request

from .. import corpus
from ..pipeline import DEFAULT_CONFIG
from .runs import RunManager


def create_app() -> Flask:
    app = Flask(__name__)
    manager = RunManager()
    download_state = {"status": "idle", "error": None}
    download_lock = threading.Lock()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            sections=corpus.SECTIONS,
            references=list(corpus.REFERENCE_SOURCES),
            defaults=DEFAULT_CONFIG,
        )

    # ---- data -----------------------------------------------------------

    @app.get("/api/data/status")
    def data_status():
        status = corpus.data_status()
        status["download"] = download_state
        return jsonify(status)

    @app.post("/api/data/download")
    def data_download():
        with download_lock:
            if download_state["status"] == "running":
                return jsonify({"ok": True, "already": True})
            download_state["status"] = "running"
            download_state["error"] = None

        def work():
            try:
                corpus.ensure_all()
                download_state["status"] = "done"
            except Exception as exc:
                download_state["status"] = "error"
                download_state["error"] = str(exc)

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"ok": True})

    # ---- runs -----------------------------------------------------------

    @app.get("/api/runs")
    def list_runs():
        return jsonify(manager.list_runs())

    @app.post("/api/runs")
    def start_run():
        body = request.get_json(force=True) or {}
        kind = body.get("kind", "solve")
        config = body.get("config", {})
        if not corpus.data_status()["ready"]:
            return jsonify({"error": "Data not downloaded yet — use the Data tab."}), 409
        try:
            config = _clean_config(kind, config)
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        if kind == "solve":
            run_id = manager.start_solve(config)
        elif kind == "benchmark":
            run_id = manager.start_benchmark(config)
        else:
            return jsonify({"error": f"unknown run kind: {kind}"}), 400
        return jsonify({"id": run_id}), 201

    @app.get("/api/runs/<int:run_id>")
    def get_run(run_id: int):
        run = manager.get(run_id)
        if run is None:
            return jsonify({"error": "no such run"}), 404
        return jsonify(run)

    @app.post("/api/runs/<int:run_id>/stop")
    def stop_run(run_id: int):
        return jsonify({"ok": manager.stop(run_id)})

    return app


def _clean_config(kind: str, raw: dict) -> dict:
    """Validate and coerce user-supplied run config."""
    cfg: dict = {}
    seed = raw.get("seed")
    cfg["seed"] = int(seed) if seed not in (None, "") else None
    cfg["reference"] = str(raw.get("reference", "latin"))
    if cfg["reference"] not in corpus.REFERENCE_SOURCES:
        raise ValueError(f"unknown reference: {cfg['reference']}")
    cfg["order"] = int(raw.get("order", 4))
    if cfg["order"] not in (3, 4):
        raise ValueError("order must be 3 or 4")
    cfg["iterations"] = max(100, min(int(raw.get("iterations", 60000)), 2_000_000))
    cfg["restarts"] = max(1, min(int(raw.get("restarts", 3)), 20))

    if kind == "solve":
        lang = str(raw.get("currier_language", "A"))
        if lang not in ("A", "B", "all"):
            raise ValueError("currier_language must be A, B or all")
        cfg["currier_language"] = lang
        section = raw.get("section") or None
        if section is not None and section not in corpus.SECTIONS:
            raise ValueError(f"unknown section: {section}")
        cfg["section"] = section
        hyp = str(raw.get("hypothesis", "simple"))
        if hyp not in ("simple", "positional"):
            raise ValueError("hypothesis must be simple or positional")
        cfg["hypothesis"] = hyp
        cfg["bpe_merges"] = max(0, min(int(raw.get("bpe_merges", 30)), 200))
    else:
        cfg["cipher_chars"] = max(500, min(int(raw.get("cipher_chars", 4000)), 50_000))
    return cfg


def main(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)
