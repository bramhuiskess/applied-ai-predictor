"""Starting the predictor: port handling, --open, and `setup` going straight to the dashboard when set up."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from predictor import cli, serving, wizard, workflow
from test_dataset_manifest import make_dataset

HOST = "127.0.0.1"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("PREDICTOR_HOME", "PREDICTOR_USECASES_DIR", "PREDICTOR_MODELS_DIR", "PREDICTOR_DATASETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    use_case = tmp_path / "usecases" / "demo"
    use_case.mkdir(parents=True)
    (use_case / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "use_case_id": "demo",
        "data_source": {"adapter": "csv-file", "path": "data/demo.csv"}, "training": {"target": "a"}}))
    make_dataset(tmp_path)
    return tmp_path


def add_model(home):
    version = home / "models" / "demo" / "1"
    (version / "a").mkdir(parents=True)
    (version / "feature_schema.json").write_text("{}")
    (version / "use_case.json").write_text('{"outputs": ["a"]}')
    (version / "a" / "descriptor.json").write_text("{}")
    (version / "a" / "model.onnx").write_bytes(b"x")


class FakePredictor:
    """Something that answers /health like a predictor, on a free port."""

    def __enter__(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"status": "ok"}' if self.path == "/health" else b"{}"
                self.send_response(200 if self.path == "/health" else 404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer((HOST, 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def busy_port():
    sock = socket.socket()
    sock.bind((HOST, 0))
    sock.listen()
    return sock


def test_pick_port_free_busy_and_running():
    with busy_port() as sock:
        port = sock.getsockname()[1]
        chosen, state = serving.pick_port(HOST, port)
        assert state == "moved" and chosen > port and serving.port_free(HOST, chosen)
    assert serving.pick_port(HOST, port)[1] == "free"
    with FakePredictor() as fake:
        assert serving.pick_port(HOST, fake.port) == (fake.port, "running")
        assert serving.health_ok(HOST, fake.port)


def test_pick_port_gives_up_when_nothing_is_free(monkeypatch):
    monkeypatch.setattr(serving, "port_free", lambda host, port: False)
    monkeypatch.setattr(serving, "health_ok", lambda host, port, timeout=0.5: False)
    with pytest.raises(OSError, match="no free port"):
        serving.pick_port(HOST, 9000, tries=3)


def test_start_when_already_running_opens_the_browser_and_does_not_serve(monkeypatch, capsys):
    opened, served = [], []
    monkeypatch.setattr(serving.webbrowser, "open", opened.append)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: served.append(1))
    with FakePredictor() as fake:
        assert serving.start(HOST, fake.port, open_browser=True) == 0
    assert opened == [f"http://{HOST}:{fake.port}/"] and not served
    assert "Already running" in capsys.readouterr().out


def test_start_serves_with_a_banner_and_opens_when_asked(monkeypatch, capsys):
    served, opened = [], []
    monkeypatch.setattr("uvicorn.run", lambda app, host, port: served.append((host, port)))
    monkeypatch.setattr(serving.bootstrap, "build_app", lambda: object())
    monkeypatch.setattr(serving, "open_when_ready", lambda host, port, timeout=60.0: opened.append((host, port)))
    with busy_port() as sock:
        taken = sock.getsockname()[1]
        assert serving.start(HOST, taken, open_browser=True) == 0
    port = served[0][1]
    out = capsys.readouterr().out
    assert port != taken and "using" in out and f"Dashboard: http://{HOST}:{port}/" in out and "Ctrl+C" in out
    assert opened == [(HOST, port)]


def test_run_open_passes_the_flag(project, monkeypatch):
    calls = []
    monkeypatch.setattr(serving, "start", lambda host, port, open_browser=False: calls.append((port, open_browser)) or 0)
    assert cli.main(["run", "--open", "--port", "8123"]) == 0
    assert cli.main(["run"]) == 0
    assert calls == [(8123, True), (8000, False)]


def test_trained_and_is_set_up(project):
    assert workflow.trained() == [] and not workflow.is_set_up()
    add_model(project)
    assert workflow.trained() == ["demo"] and workflow.is_set_up()
    (project / "models" / "demo" / "1" / "a" / "model.onnx").unlink()  # an incomplete model does not count
    assert workflow.trained() == [] and not workflow.is_set_up()


def test_setup_on_a_set_up_project_just_starts(project, monkeypatch, capsys):
    add_model(project)
    started, wizards = [], []
    monkeypatch.setattr(serving, "start", lambda host, port, open_browser=False: started.append(open_browser) or 0)
    monkeypatch.setattr(cli, "_setup_browser", lambda args: wizards.append(1) or 0)
    assert cli.main(["setup"]) == 0
    assert started == [True] and not wizards
    assert "--reconfigure" in capsys.readouterr().out
    assert cli.main(["setup", "--reconfigure"]) == 0
    assert wizards == [1] and started == [True]


def test_setup_on_a_new_project_shows_the_wizard_and_cli_is_unchanged(project, monkeypatch):
    started, wizards, terminals = [], [], []
    monkeypatch.setattr(serving, "start", lambda *a, **k: started.append(1) or 0)
    monkeypatch.setattr(cli, "_setup_browser", lambda args: wizards.append(1) or 0)
    monkeypatch.setattr(cli, "_setup_terminal", lambda args: terminals.append(1) or 0)
    assert cli.main(["setup"]) == 0
    add_model(project)
    assert cli.main(["setup", "--cli"]) == 0
    assert wizards == [1] and terminals == [1] and not started


def test_wizard_status_reports_trained_use_cases_and_the_port(project):
    add_model(project)
    with TestClient(wizard.create_wizard_app(lambda: None)) as client:
        body = client.get("/setup/api/status").json()
    assert body["trained"] == ["demo"] and "port" in body
