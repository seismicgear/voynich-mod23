"""
webapp — browser GUI for the decipherment workbench.

A small Flask app: one page, a JSON API, and a thread-based run manager.
Runs (Voynich solves and synthetic benchmarks) execute in background
threads and stream progress through polling endpoints.
"""

from .app import create_app

__all__ = ["create_app"]
