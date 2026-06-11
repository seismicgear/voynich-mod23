"""
runs.py — background run management for the GUI.

Each run executes in a daemon thread, publishing progress into a shared
dict guarded by a lock.  The annealer polls `should_stop` so runs can be
cancelled from the browser mid-flight.
"""

from __future__ import annotations

import itertools
import threading
import traceback
from datetime import datetime, timezone

from .. import corpus
from ..pipeline import save_report, solve_voynich
from ..synthetic import run_benchmark

MAX_HISTORY_POINTS = 600


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[int, dict] = {}
        self._ids = itertools.count(1)

    # ---- public API ----------------------------------------------------

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [self._summary(r) for r in self._runs.values()]

    def get(self, run_id: int) -> dict | None:
        with self._lock:
            run = self._runs.get(run_id)
            return self._detail(run) if run else None

    def stop(self, run_id: int) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run["status"] != "running":
                return False
            run["stop_event"].set()
            return True

    def start_solve(self, config: dict) -> int:
        return self._start("solve", config)

    def start_benchmark(self, config: dict) -> int:
        return self._start("benchmark", config)

    # ---- internals -------------------------------------------------------

    def _start(self, kind: str, config: dict) -> int:
        run_id = next(self._ids)
        run = {
            "id": run_id,
            "kind": kind,
            "config": config,
            "status": "running",
            "started": datetime.now(timezone.utc).isoformat(),
            "progress": {},
            "history": [],
            "result": None,
            "error": None,
            "stop_event": threading.Event(),
        }
        with self._lock:
            self._runs[run_id] = run
        thread = threading.Thread(target=self._work, args=(run,), daemon=True)
        thread.start()
        return run_id

    def _work(self, run: dict) -> None:
        def on_progress(p: dict) -> None:
            with self._lock:
                run["progress"] = p
                hist = run["history"]
                step = p["restart"] * p["total_iterations"] + p["iteration"]
                hist.append([step, p["best_score"]])
                if len(hist) > MAX_HISTORY_POINTS:
                    # Decimate to keep payloads small.
                    run["history"] = hist[::2]

        try:
            if run["kind"] == "solve":
                report = solve_voynich(
                    run["config"],
                    progress=on_progress,
                    should_stop=run["stop_event"].is_set,
                )
                path = save_report(report)
                report["saved_to"] = str(path)
            else:
                cfg = run["config"]
                text = corpus.load_reference(cfg.get("reference", "english"))
                report = run_benchmark(
                    text,
                    order=int(cfg.get("order", 4)),
                    cipher_chars=int(cfg.get("cipher_chars", 4000)),
                    iterations=int(cfg.get("iterations", 20000)),
                    restarts=int(cfg.get("restarts", 2)),
                    seed=cfg.get("seed"),
                    progress=on_progress,
                    should_stop=run["stop_event"].is_set,
                )
            with self._lock:
                run["result"] = report
                run["status"] = "stopped" if run["stop_event"].is_set() else "done"
        except Exception as exc:  # surface errors to the GUI
            with self._lock:
                run["status"] = "error"
                run["error"] = f"{type(exc).__name__}: {exc}"
                run["traceback"] = traceback.format_exc()

    @staticmethod
    def _summary(run: dict) -> dict:
        return {
            "id": run["id"],
            "kind": run["kind"],
            "status": run["status"],
            "started": run["started"],
            "config": run["config"],
            "progress": run["progress"],
            "error": run["error"],
        }

    @classmethod
    def _detail(cls, run: dict) -> dict:
        out = cls._summary(run)
        out["history"] = run["history"]
        result = run["result"]
        if result is not None:
            # History inside the result duplicates the live history; drop it.
            result = {k: v for k, v in result.items() if k != "history"}
        out["result"] = result
        return out
