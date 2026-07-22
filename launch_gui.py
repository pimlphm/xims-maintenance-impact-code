"""Launch the MII-CODE standalone browser interface.

The launcher uses only Python's standard library. It serves the versioned static
GUI from ``docs/`` and, by default, binds to the loopback interface so local CSV
data remains in the browser.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
import webbrowser


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class QuietStaticHandler(SimpleHTTPRequestHandler):
    """Serve static assets while keeping routine requests out of the console."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def default_interface_root() -> Path:
    """Return the packaged standalone GUI directory."""

    return Path(__file__).resolve().parent / "docs"


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    root: Path | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, the standalone HTTP server."""

    interface_root = (root or default_interface_root()).resolve()
    index = interface_root / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"MII-CODE interface not found: {index}")
    handler = partial(QuietStaticHandler, directory=str(interface_root))
    return ThreadingHTTPServer((host, port), handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the standalone MII-CODE graphical interface.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address; defaults to loopback only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local TCP port; use 0 for an automatic port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the default browser automatically.")
    parser.add_argument("--root", type=Path, default=None, help="Alternative static interface directory for testing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = create_server(args.host, args.port, args.root)
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"MII-CODE standalone interface: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MII-CODE.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
