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
            catalog=corpus.reference_catalog(),
            n_references=len(corpus.REFERENCE_SOURCES),
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
        try:
            config = _clean_config(kind, config)
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400
        missing = _missing_data(kind, config)
        if missing:
            return jsonify({
                "error": "Missing data for this run: "
                + ", ".join(missing)
                + " — download it from the Data tab."
            }), 409
        if kind == "solve":
            run_id = manager.start_solve(config)
        elif kind == "benchmark":
            run_id = manager.start_benchmark(config)
        elif kind == "sweep":
            run_id = manager.start_sweep(config)
        elif kind == "diagnostics":
            run_id = manager.start_diagnostics(config)
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


def _missing_data(kind: str, cfg: dict) -> list[str]:
    """Names of the data files THIS run needs that are absent.  A run
    against one reference must not be blocked because an unrelated
    corpus failed to download."""
    missing = []
    if kind in ("solve", "sweep", "diagnostics") and not corpus.voynich_path().exists():
        missing.append("Voynich transcription")
    if kind in ("solve", "benchmark"):
        refs = [cfg["reference"]]
    elif kind == "sweep":
        refs = cfg.get("references") or list(corpus.REFERENCE_SOURCES)
    else:  # diagnostics compares against these by default
        refs = cfg.get("references", ["latin", "english"])
    missing += [
        f"reference '{r}'" for r in refs if not corpus.reference_path(r).exists()
    ]
    return missing


def _clean_config(kind: str, raw: dict) -> dict:
    """Validate and coerce user-supplied run config."""
    from ..pipeline import HYPOTHESES, LOCKABLE_HYPOTHESES

    cfg: dict = {}
    seed = raw.get("seed")
    cfg["seed"] = int(seed) if seed not in (None, "") else None

    if kind == "diagnostics":
        lang = str(raw.get("currier_language", "A"))
        if lang not in ("A", "B", "all"):
            raise ValueError("currier_language must be A, B or all")
        cfg["currier_language"] = lang
        section = raw.get("section") or None
        if section is not None and section not in corpus.SECTIONS:
            raise ValueError(f"unknown section: {section}")
        cfg["section"] = section
        cfg["window"] = max(2, min(int(raw.get("window", 15)), 100))
        cfg["seed"] = cfg["seed"] if cfg["seed"] is not None else 0
        return cfg

    cfg["order"] = int(raw.get("order", 4))
    if cfg["order"] not in (3, 4):
        raise ValueError("order must be 3 or 4")
    cfg["iterations"] = max(100, min(int(raw.get("iterations", 60000)), 2_000_000))
    cfg["restarts"] = max(1, min(int(raw.get("restarts", 3)), 20))

    if kind != "sweep":
        cfg["reference"] = str(raw.get("reference", "latin"))
        if cfg["reference"] not in corpus.REFERENCE_SOURCES:
            raise ValueError(f"unknown reference: {cfg['reference']}")

    if kind in ("solve", "sweep"):
        lang = str(raw.get("currier_language", "A"))
        if lang not in ("A", "B", "all"):
            raise ValueError("currier_language must be A, B or all")
        cfg["currier_language"] = lang
        section = raw.get("section") or None
        if section is not None and section not in corpus.SECTIONS:
            raise ValueError(f"unknown section: {section}")
        cfg["section"] = section
        hyp = str(raw.get("hypothesis", "simple"))
        if hyp not in HYPOTHESES:
            raise ValueError(f"hypothesis must be one of {HYPOTHESES}")
        cfg["hypothesis"] = hyp
        cfg["abjad"] = bool(raw.get("abjad", False))
        cfg["bpe_merges"] = max(0, min(int(raw.get("bpe_merges", 30)), 200))
        cfg["lock_rounds"] = max(0, min(int(raw.get("lock_rounds", 0) or 0), 5))
        if cfg["lock_rounds"] and hyp not in LOCKABLE_HYPOTHESES:
            raise ValueError(
                f"lock rounds require one of: {', '.join(LOCKABLE_HYPOTHESES)}"
            )
        reverse = str(raw.get("reverse", "none") or "none")
        if reverse not in ("none", "words", "lines"):
            raise ValueError("reverse must be none, words or lines")
        cfg["reverse"] = reverse
        cfg["control"] = bool(raw.get("control", False))
        cfg["allow_nulls"] = bool(raw.get("allow_nulls", False))
        if cfg["allow_nulls"] and hyp != "abbreviation":
            raise ValueError("allow_nulls requires the abbreviation hypothesis")

    if kind == "sweep":
        refs = raw.get("references")
        if refs is not None:
            if not isinstance(refs, list) or not refs:
                raise ValueError("references must be a non-empty list")
            for r in refs:
                if r not in corpus.REFERENCE_SOURCES:
                    raise ValueError(f"unknown reference: {r}")
            cfg["references"] = [str(r) for r in refs]

    if kind == "benchmark":
        cfg["cipher_chars"] = max(500, min(int(raw.get("cipher_chars", 4000)), 50_000))
        mode = str(raw.get("mode", "substitution"))
        if mode not in ("substitution", "abbreviation", "nulls", "anagram",
                        "nomenclator"):
            raise ValueError(
                "mode must be substitution, abbreviation, nulls, anagram "
                "or nomenclator"
            )
        cfg["mode"] = mode
    return cfg


def main(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)
