"""Starting the dashboard: pick a port, notice an already running predictor, open the browser.

Shared by ``predictor run`` and the end of ``predictor setup``. A predictor is recognised by its ``/health``
answer; anything else on the port is left alone and the next free port is used.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
import webbrowser

from predictor import bootstrap

MAX_PORT_TRIES = 20


def health_ok(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when a predictor answers on this port (the setup wizard has no /health, so it is not one)."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, preferred: int, tries: int = MAX_PORT_TRIES) -> tuple[int, str]:
    """(port, state): "free", "running" (a predictor already answers there) or "moved" (the next free port)."""
    if port_free(host, preferred):
        return preferred, "free"
    if health_ok(host, preferred):
        return preferred, "running"
    for port in range(preferred + 1, preferred + 1 + tries):
        if port_free(host, port):
            return port, "moved"
    raise OSError(f"no free port between {preferred} and {preferred + tries}")


def open_when_ready(host: str, port: int, timeout: float = 60.0) -> threading.Thread:
    """Open the browser as soon as /health answers (not after a fixed sleep)."""
    def wait() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if health_ok(host, port):
                webbrowser.open(f"http://{host}:{port}/")
                return
            time.sleep(0.3)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return thread


def start(host: str, port: int, *, open_browser: bool = False) -> int:
    """Serve the dashboard (blocking). Returns an exit code."""
    import uvicorn

    try:
        port, state = pick_port(host, port)
    except OSError as exc:
        print(f"cannot start: {exc}")
        return 1
    url = f"http://{host}:{port}/"
    if state == "running":
        print(f"Already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return 0
    if state == "moved":
        print(f"Port is taken by something else; using {port} instead.")
    print(f"Dashboard: {url}   (Ctrl+C to stop)")
    if open_browser:
        open_when_ready(host, port)
    uvicorn.run(bootstrap.build_app(), host=host, port=port)
    return 0
