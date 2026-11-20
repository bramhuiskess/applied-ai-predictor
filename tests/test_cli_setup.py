"""The setup commands and the browser wizard: the same results through both."""

import json

import pytest
from fastapi.testclient import TestClient

from predictor import cli, data_fetch, serving, wizard
from test_dataset_manifest import make_dataset


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("PREDICTOR_HOME", "PREDICTOR_USECASES_DIR", "PREDICTOR_MODELS_DIR", "PREDICTOR_DATASETS_DIR"):
        monkeypatch.delenv(var, raising=False)
    use_case = tmp_path / "usecases" / "demo"
    use_case.mkdir(parents=True)
    (use_case / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file", "path": "data/demo.csv"},
        "training": {"target": "a"}}))
    make_dataset(tmp_path)
    return tmp_path


def test_doctor_command_exit_code_and_json(project, capsys):
    assert cli.main(["doctor", "--json"]) == 0  # not trained is a warning, not an error
    findings = json.loads(capsys.readouterr().out)
    assert any(f["check"] == "usecase demo" and f["level"] == "warn" for f in findings)
    (project / "data" / "demo.csv").write_text("a,b,when\n1,x,2024-01-01\n")  # differs from dataset.json
    assert cli.main(["doctor"]) == 1


def test_home_option_changes_into_the_project(project, tmp_path_factory, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert cli.main(["--home", str(project), "doctor"]) == 0
    assert cli.main(["--home", str(project / "nope"), "doctor"]) == 2


def test_data_fetch_and_list_commands(project, capsys):
    tomato = data_fetch.DEMO_DATASETS[3]
    (project / tomato.files[0]).write_text("a,b\n1,2\n")
    assert cli.main(["data", "fetch", tomato.id]) == 0
    assert (project / "datasets" / tomato.id / "dataset.json").is_file()
    assert cli.main(["data", "list"]) == 0
    assert "tomato-season" in capsys.readouterr().out
    assert cli.main(["data", "fetch", "nope"]) == 2


def test_package_commands(project, tmp_path_factory, capsys):
    out = tmp_path_factory.mktemp("pkg") / "p"
    assert cli.main(["package", "build", "--out", str(out)]) == 0
    assert cli.main(["package", "verify", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out.split("wrote")[-1].split("\n", 1)[1])["ok"]
    (out / "usecases" / "demo" / "manifest.json").write_text("{}")
    assert cli.main(["package", "verify", str(out)]) == 1


def test_setup_terminal_flow_without_data_is_serve_only(project, capsys):
    (project / "data" / "demo.csv").unlink()  # no data here: nothing can be trained
    assert cli.main(["setup", "--cli", "--yes"]) == 0
    assert "predictor run" in capsys.readouterr().out
    assert not (project / "models").exists()  # nothing trained: no demo data, no models


def test_run_refuses_to_start_on_errors(project, capsys, monkeypatch):
    (project / "usecases" / "demo" / "manifest.json").write_text("{}")
    started = []
    monkeypatch.setattr(serving, "start", lambda *a, **k: started.append(1) or 0)
    assert cli.main(["run"]) == 1 and not started


def test_run_starts_when_ready(project, monkeypatch):
    started = []
    monkeypatch.setattr(serving, "start", lambda *a, **k: started.append(1) or 0)
    assert cli.main(["run"]) == 0 and started


# ------------------------------------------------------------------ wizard

@pytest.fixture
def client(project):
    finished = []
    with TestClient(wizard.create_wizard_app(lambda: finished.append(1))) as c:
        c.finished = finished
        yield c


def test_wizard_status_matches_the_doctor(project, client):
    from predictor import workflow

    status = client.get("/setup/api/status").json()
    assert status["findings"] == [f.as_dict() for f in workflow.doctor()]
    assert status["demo_answer"] is None and {d["id"] for d in status["demo_data"]} == set(data_fetch.known_ids())
    assert "Predictor setup" in client.get("/setup").text
    assert client.get("/", follow_redirects=False).headers["location"] == "/setup"


def test_wizard_demo_answer_no_starts_nothing(client):
    assert client.post("/setup/api/demo-data", json={"install": False}).json() == {"started": False}
    assert client.get("/setup/api/status").json()["demo_answer"] is False
    assert client.get("/setup/api/job").json()["state"] == "idle"


def test_wizard_demo_answer_yes_runs_a_fetch_job(client, monkeypatch):
    seen = []
    monkeypatch.setattr(data_fetch, "fetch", lambda home, ids=None, log=print, **kw: (log("fetching"), seen.append(home)))
    assert client.post("/setup/api/demo-data", json={"install": True}).json() == {"started": True}
    for _ in range(100):
        job = client.get("/setup/api/job").json()
        if job["state"] != "running":
            break
    assert job["state"] == "done" and job["log"] == ["fetching"] and seen


def test_wizard_finish_calls_back(client):
    import time

    assert client.post("/setup/api/finish").json() == {"status": "starting"}
    time.sleep(0.8)
    assert client.finished == [1]


def test_wizard_refuses_non_local_requests(client):
    assert client.get("/setup/api/status", headers={"host": "evil.example"}).status_code == 403
    assert client.post("/setup/api/finish", headers={"origin": "http://evil.example"}).status_code == 403
    assert client.get("/setup/api/status", headers={"host": "localhost:8000"}).status_code == 200
