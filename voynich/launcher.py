"""
launcher.py — desktop entry point.

Starts the workbench server on a local port and opens the browser.
This is what the packaged executables and the `voynich-workbench`
console script run.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voynich-workbench",
        description="Voynich Decipherment Workbench — local web GUI",
    )
    parser.add_argument("--port", type=int, default=0,
                        help="port to serve on (default: pick a free one)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args(argv)

    from .webapp import create_app

    port = args.port or _free_port()
    url = f"http://{args.host}:{port}/"
    print("Voynich Decipherment Workbench — built by Montgomery Kuykendall")
    print(f"Serving at {url}  (Ctrl+C to quit)")

    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, [url]).start()

    create_app().run(host=args.host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
