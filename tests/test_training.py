import json

import pytest

from predictor import bootstrap
from predictor.trainer.errors import TrainingError
from predictor.trainer.run import init_use_case, train_use_case


def descriptor(project, use_case, version="1"):
    return json.loads((project / "models" / use_case / version / "descriptor.json").read_text())


@pytest.mark.parametrize("use_case", ["greenhouse-climate", "absence-forecast", "pump-failure", "pump-anomaly"])
def test_every_model_passes_the_onnx_parity_check(project, use_case):
    d = descriptor(project, use_case)
    assert d["parity"]["passed"] is True
    assert d["parity"]["max_abs_diff"] < 1e-3
    assert (project / "models" / use_case / "1" / "report.md").is_file()


def test_task_is_detected_automatically(project):
    d = descriptor(project, "pump-failure")
    assert d["task_type"] == "classification"
    assert d["classes"] == ["0", "1"]
    assert "pump_id" not in d["features"]


def test_lightgbm_beats_the_baseline_on_the_greenhouse_forecast(project):
    d = descriptor(project, "greenhouse-climate")
    assert d["split"]["chronological"] is True
    assert d["algorithm"] == "lightgbm"
    assert d["metrics"]["mae"] < d["reference"]["persistence_mae"]


def test_policy_is_applied_during_training(project):
    d = descriptor(project, "absence-forecast")
    assert any("headcount < 5" in note for note in d["notes"])
    assert "department_id" not in d["features"]


def test_training_refuses_a_dataset_with_forbidden_columns(project):
    registry = bootstrap.build_registry(project / "models")
    with pytest.raises(TrainingError, match="forbidden column"):
        train_use_case(project / "usecases" / "absence-forecast", registry,
                       project / "data" / "absence_with_diagnosis.csv")


def test_retraining_creates_a_new_version(scratch, project):
    registry = bootstrap.build_registry(scratch / "models")
    path, _ = init_use_case("pumps", scratch / "data" / "pumps.csv", scratch / "usecases", target="failure")
    assert train_use_case(path, registry)["model_version"] == "1"
    assert train_use_case(path, registry)["model_version"] == "2"
    assert registry.versions("pumps") == ["1", "2"]


def test_new_dataset_end_to_end_without_code_changes(scratch):
    """The core claim: init + train + serve a never-seen dataset with configuration only."""
    from fastapi.testclient import TestClient

    path, remarks = init_use_case("new-client", scratch / "data" / "pumps.csv", scratch / "usecases",
                                  target="failure")
    manifest = json.loads(path.read_text())
    assert manifest["training"]["drop_columns"] == ["pump_id"]  # identifier detected
    train_use_case(path, bootstrap.build_registry(scratch / "models"))

    client = TestClient(bootstrap.build_app(scratch / "usecases", scratch / "models"))
    features = {"pump_type": "A", "vibration_mm_s": 3.0, "pressure_bar": 6.0, "filter_ok": 1}
    response = client.post("/models/new-client/predict", json={"records": [{"features": features}]})
    assert response.status_code == 200, response.text


def test_init_flags_probable_personal_data(scratch):
    path, remarks = init_use_case("raw-absence", scratch / "data" / "absence_with_diagnosis.csv",
                                  scratch / "usecases", target="absence_pct")
    manifest = json.loads(path.read_text())
    assert manifest["policy"]["forbidden_fields"] == ["diagnosis"]
    assert manifest["policy"]["log_feature_values"] is False
    assert any("personal data" in r for r in remarks)


def test_target_leakage_is_flagged(scratch):
    import pandas as pd

    data = pd.read_csv(scratch / "data" / "pumps.csv")
    data["repair_ticket_opened"] = data["failure"]  # only known after the failure
    data.to_csv(scratch / "data" / "leaky.csv", index=False)
    path, _ = init_use_case("leaky", scratch / "data" / "leaky.csv", scratch / "usecases", target="failure")
    result = train_use_case(path, bootstrap.build_registry(scratch / "models"))
    assert any("leakage" in n and "repair_ticket_opened" in n for n in result["notes"])
