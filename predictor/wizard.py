"""The browser setup wizard (``predictor setup``): a thin layer over ``predictor.workflow`` and ``data_fetch``.

It is a separate little app, never part of ``predictor serve``. It answers on localhost only (the server binds
127.0.0.1, and every request is also checked for a local client, a local Host header and a same-origin Origin),
and it stops itself when the user finishes, after which ``predictor setup`` starts the normal server on the
same port. Long steps run as one background job whose log the page polls.
"""

from __future__ import annotations

import threading
from importlib.resources import files
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from predictor import bootstrap, data_fetch, workflow
from predictor.core.doctor import has_errors, installed_train_extra

LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1", "testserver"}


class DemoAnswer(BaseModel):
    install: bool


class Jobs:
    """One background job at a time, with a log the page can poll."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {"name": None, "state": "idle", "log": [], "error": None}

    def start(self, name: str, work: Callable[[Callable[[str], None]], None]) -> bool:
        with self._lock:
            if self.state["state"] == "running":
                return False
            self.state = {"name": name, "state": "running", "log": [], "error": None}
        state = self.state

        def run() -> None:
            try:
                work(lambda line: state["log"].append(line))
                state["state"] = "done"
            except Exception as exc:  # noqa: BLE001 - shown to the user in the wizard
                state["error"] = str(exc) or type(exc).__name__
                state["state"] = "failed"

        threading.Thread(target=run, daemon=True).start()
        return True


def create_wizard_app(on_finish: Callable[[], None]) -> FastAPI:
    app = FastAPI(title="Predictor setup", docs_url=None, redoc_url=None, openapi_url=None)
    jobs = Jobs()
    answers: dict[str, Any] = {"demo": None}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = _hostname(request.headers.get("host") or "")
        origin = request.headers.get("origin")
        if (request.client and request.client.host not in LOCAL_CLIENTS) or host not in LOCAL_HOSTS or \
                (origin and urlparse(origin).netloc != request.headers.get("host")):
            return JSONResponse(status_code=403, content={"detail": "the setup wizard only answers on localhost"})
        return await call_next(request)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/setup")

    @app.get("/setup", include_in_schema=False)
    def page() -> HTMLResponse:
        return HTMLResponse(files("predictor").joinpath("setup.html").read_text(encoding="utf-8"))

    @app.get("/setup/api/status")
    def status(request: Request) -> dict[str, Any]:
        findings = workflow.doctor()
        return {"findings": [f.as_dict() for f in findings], "has_errors": has_errors(findings),
                "train_extra": installed_train_extra(), "demo_data": data_fetch.state(bootstrap.home()),
                "demo_answer": answers["demo"], "trainable": _safe(workflow.trainable),
                "trained": _safe(workflow.trained), "port": request.url.port}

    @app.post("/setup/api/demo-data")
    def demo_data(answer: DemoAnswer) -> dict[str, Any]:
        answers["demo"] = answer.install
        if not answer.install:
            return {"started": False}
        if not installed_train_extra():
            return JSONResponse(status_code=409, content={"detail": "install the training libraries first: "
                                                                    "pip install -e '.[train,datasets]'"})
        return {"started": jobs.start("demo-data", lambda log: data_fetch.fetch(bootstrap.home(), log=log))}

    @app.post("/setup/api/train")
    def train() -> dict[str, Any]:
        if not installed_train_extra():
            return JSONResponse(status_code=409, content={"detail": "the training libraries are not installed"})
        return {"started": jobs.start("train", lambda log: workflow.train_ids(workflow.trainable(), log))}

    @app.get("/setup/api/job")
    def job() -> dict[str, Any]:
        return {**jobs.state, "log": list(jobs.state["log"])}

    @app.post("/setup/api/finish")
    def finish() -> dict[str, str]:
        threading.Timer(0.5, on_finish).start()  # after this response has been sent
        return {"status": "starting"}

    return app


def _hostname(header: str) -> str:
    if header.startswith("["):  # [::1]:8000
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0] if ":" in header else header


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - broken manifests are reported by the doctor findings
        return []
