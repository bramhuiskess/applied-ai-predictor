"""Ticket 2: scenario variables ("levers") in manifest.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core.manifest import ManifestError, load_manifest
from predictor.core.validation import FeatureValidator
from predictor.trainer.errors import TrainingError
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_nursery  # noqa: E402


def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def greenhouse_data(path, n=600, seed=0):
    """Gas rises with the heating setpoint but falls with lamp hours (lamp heat)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "setpoint": rng.uniform(17, 23, n).round(1),
        "lamps": rng.uniform(0, 16, n).round(1),
        "outside": rng.uniform(0, 15, n).round(1),
        "screen": rng.integers(0, 2, n),
        "variety": rng.choice(["a", "b"], n),
    })
    df["gas"] = (100 * (df.setpoint - df.outside).clip(0) - 25 * df.lamps - 40 * df.screen + 600
                 + rng.normal(0, 40, n)).round(1)
    df.to_csv(path, index=False)


LEVERS = {
    "setpoint": {"label": "Temperature", "unit": "°C", "min": 17, "max": 25, "step": 0.5, "effects": {"gas": "increasing"}},
    "lamps": {"min": 1, "max": 15, "step": 0.5, "effects": {"gas": "decreasing"}},
}


def manifest(data, **scenario):
    return {"manifest_version": "1.0", "use_case_id": "kas",
            "data_source": {"adapter": "csv-file", "path": str(data)},
            "training": {"target": "gas", "task": "regression"},
            "scenario": {"levers": LEVERS, "outputs": {"gas": {"label": "Gas", "value_eur": -0.45}}, **scenario}}


# --------------------------------------------------------------------------- manifest validation

def broken(**changes):
    m = manifest("data.csv")
    for path, value in changes.items():
        target = m
        *parents, key = path.split(".")
        for p in parents:
            target = target[p]
        target[key] = value
    return m


@pytest.mark.parametrize("m, message", [
    (broken(**{"scenario.levers": {"setpoint": {"min": 25, "max": 17, "step": 1}}}), "must be smaller than max"),
    (broken(**{"scenario.levers": {"setpoint": {"min": 17, "max": 25, "step": 0}}}), "greater than 0"),
    (broken(**{"scenario.levers": {"setpoint": {"min": 0, "max": 1e9, "step": 1}}}), "too small for the range"),
    (broken(**{"scenario.levers": {"setpoint": {"min": 17, "max": 25, "step": 1, "effects": {"gas": "up"}}}}),
     "increasing"),
    (broken(**{"scenario.levers": {"setpoint": {"min": 17, "max": 25, "step": 1, "effects": {"plants": "increasing"}}}}),
     "unknown output 'plants'"),
    (broken(**{"scenario.levers": {"setpoint": {"min": 17, "max": 25, "step": 1, "efects": {}}}}), "Extra inputs"),
    (broken(**{"scenario.outputs": {"plants": {"value_eur": 2}}}), "unknown output 'plants'"),
    (broken(**{"scenario.levers": {}}), "at least 1"),
    (broken(**{"scenario.levers": {"gas": {"min": 0, "max": 1, "step": 1}}}), "is the target"),
    (broken(**{"training.drop_columns": ["lamps"]}), "is the drop_columns"),
    (broken(**{"training.time_column": "setpoint"}), "is the time_column"),
    (broken(**{"policy": {"forbidden_fields": ["lamps"]}}), "forbidden field"),
    (broken(**{"training": {"task": "anomaly"}}), "needs a target"),
])
def test_invalid_scenario_sections_fail_fast(tmp_path, m, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(write(tmp_path, m))


def test_valid_scenario_section_loads(tmp_path):
    loaded = load_manifest(write(tmp_path, manifest("data.csv")))
    assert loaded.training.outputs == ["gas"]
    assert loaded.scenario.levers["lamps"].effects == {"gas": "decreasing"}
    assert loaded.scenario.outputs["gas"].value_eur == -0.45


# --------------------------------------------------------------------------- training

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenario")
    data = root / "kas.csv"
    greenhouse_data(data)
    result = train_use_case(write(root, manifest(data)), bootstrap.build_registry(root / "models"))
    return root, data, result


def test_declared_effects_hold_for_every_row(trained):
    """Monotone constraints: for any situation, turning a lever up moves gas the declared way."""
    root, data, result = trained
    model = bootstrap.build_registry(root / "models").resolve("kas")
    validator = FeatureValidator(model.feature_schema, reject_unknown_fields=False)
    rows = pd.read_csv(data).drop(columns=["gas"]).head(200).to_dict("records")
    for lever, sign in [("setpoint", 1), ("lamps", -1)]:
        previous = None
        for value in np.linspace(*model.feature_schema["properties"][lever]["x-observed-range"], 15):
            matrix = validator.to_matrix([{**r, lever: float(value)} for r in rows])
            pred = np.asarray(model.runner.predict(matrix).values)
            if previous is not None:
                assert (sign * (pred - previous) >= -1e-3).all(), lever
            previous = pred
    assert result["scenario_constraints"]["effects"] == {"setpoint": "increasing", "lamps": "decreasing"}


def test_report_lists_the_levers(trained):
    _, _, result = trained
    report = Path(result["report"]).read_text()
    assert "## Scenario levers" in report and "| `lamps` | 1 - 15 | 0.5 | decreasing |" in report


def test_linear_model_that_contradicts_a_declared_effect_cannot_win(tmp_path):
    data = tmp_path / "kas.csv"
    greenhouse_data(data)
    m = manifest(data)
    # Deliberately wrong declaration: the data says lamps lower gas use.
    m["scenario"]["levers"]["lamps"]["effects"] = {"gas": "increasing"}
    result = train_use_case(write(tmp_path, m), bootstrap.build_registry(tmp_path / "models"))
    assert result["algorithm"] == "lightgbm"
    assert "linear_regression" in result["scenario_constraints"]["excluded_candidates"]
    assert any("not eligible" in note for note in result["notes"])
    # The exported LightGBM obeys the declaration even though the data disagrees.
    model = bootstrap.build_registry(tmp_path / "models").resolve("kas")
    validator = FeatureValidator(model.feature_schema, reject_unknown_fields=False)
    rows = pd.read_csv(data).drop(columns=["gas"]).head(100).to_dict("records")
    preds = [np.asarray(model.runner.predict(validator.to_matrix([{**r, "lamps": v} for r in rows])).values)
             for v in (1.0, 8.0, 15.0)]
    assert (preds[1] >= preds[0] - 1e-3).all() and (preds[2] >= preds[1] - 1e-3).all()


@pytest.mark.parametrize("lever, message", [
    ("does_not_exist", "not a model input"),
    ("variety", "categorical"),
])
def test_lever_must_be_a_numeric_model_input(tmp_path, lever, message):
    data = tmp_path / "kas.csv"
    greenhouse_data(data)
    m = manifest(data)
    m["scenario"]["levers"] = {lever: {"min": 0, "max": 1, "step": 1}}
    with pytest.raises(TrainingError, match=message):
        train_use_case(write(tmp_path, m), bootstrap.build_registry(tmp_path / "models"))


# --------------------------------------------------------------------------- serving

def test_model_metadata_describes_the_levers(trained):
    root, _, _ = trained
    client = TestClient(bootstrap.build_app(root / "usecases", root / "models"))
    scenario = client.get("/models/kas").json()["scenario"]
    levers = {lever["name"]: lever for lever in scenario["levers"]}
    assert levers["setpoint"]["label"] == "Temperature" and levers["setpoint"]["unit"] == "°C"
    assert levers["setpoint"]["beyond_training_data"] is True   # max 25, the data stops at 23
    assert levers["lamps"]["beyond_training_data"] is False
    assert levers["lamps"]["label"] == "lamps"                   # falls back to the column name
    [output] = scenario["outputs"]
    assert {k: output[k] for k in ("name", "label", "unit", "value_eur")} == {
        "name": "gas", "label": "Gas", "unit": None, "value_eur": -0.45}
    assert scenario["constraints"]["effects"]["lamps"] == "decreasing"


def test_use_cases_without_scenario_are_unchanged(client):
    assert client.get("/models/greenhouse-climate").json()["scenario"] is None


def test_startup_fails_when_a_lever_is_not_an_input_of_the_trained_model(tmp_path):
    data = tmp_path / "kas.csv"
    greenhouse_data(data)
    m = manifest(data)
    del m["scenario"]
    path = write(tmp_path, m)
    train_use_case(path, bootstrap.build_registry(tmp_path / "models"))
    m["scenario"] = {"levers": {"humidity": {"min": 50, "max": 90, "step": 5}}}
    write(tmp_path, m)  # manifest changed after training
    with pytest.raises(ManifestError, match="not an input of the trained model"):
        bootstrap.build_app(tmp_path / "usecases", tmp_path / "models")


def test_yes_no_lever_needs_a_yes_no_range(tmp_path):
    data = tmp_path / "kas.csv"
    greenhouse_data(data)
    m = manifest(data)
    m["scenario"]["levers"] = {"screen": {"min": 0, "max": 2, "step": 1}}
    train_use_case(write(tmp_path, m), bootstrap.build_registry(tmp_path / "models"))
    with pytest.raises(ManifestError, match="yes/no"):
        bootstrap.build_app(tmp_path / "usecases", tmp_path / "models")


# --------------------------------------------------------------------------- evidence for the real manifests

def test_declared_nursery_effects_match_the_answer_key():
    """Every direction declared in the nursery manifests holds for every simulated batch."""
    daily = knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=8)
    _, sim, batches = make_nursery.generate(daily, compartments=4, seed=9)
    deltas = {"temp_setpoint_c": 1.0, "light_hours": 2.0, "co2_ppm": 200.0, "ec_ms_cm": 0.3, "plant_density_m2": 5.0}
    for path in sorted((ROOT / "usecases").glob("nursery-*/manifest.json")):
        m = load_manifest(path)
        for name, lever in m.scenario.levers.items():
            for output, direction in lever.effects.items():
                sign = 1 if direction == "increasing" else -1
                for b in batches:
                    if name == "light_hours" and sim.compartments[b["compartment"]].lamp_type == "none":
                        continue
                    change = (sim.counterfactual(b, **{name: b[name] + deltas[name]})[output]
                              - sim.simulate(b)[output])
                    assert sign * change >= 0, (m.use_case_id, name, output, b["batch_id"], change)
