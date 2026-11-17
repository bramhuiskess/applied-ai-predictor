"""Ticket 4: calculating scenarios (POST /models/{id}/scenario and /scenario/sweep)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import repoint_nursery, write_nursery_tables
from predictor import bootstrap
from predictor.core.api import EXAMPLE_SCENARIO, EXAMPLE_SWEEP
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


@pytest.fixture(scope="module")
def kas(tmp_path_factory):
    """Gas (regression) and a yes/no 'too cold' alarm (classification), both with levers."""
    root = tmp_path_factory.mktemp("scenario-calc")
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({
        "setpoint": rng.uniform(17, 23, n).round(1),
        "lamps": rng.uniform(0, 16, n).round(1),
        "outside": rng.uniform(0, 15, n).round(1),
        "screen": rng.integers(0, 2, n),
    })
    df["gas"] = (100 * (df.setpoint - df.outside).clip(0) - 25 * df.lamps - 40 * df.screen + 600
                 + rng.normal(0, 30, n)).round(1)
    df["too_cold"] = ((df.setpoint + 0.1 * df.lamps + rng.normal(0, 0.8, n)) < 19).astype(int)
    df.to_csv(root / "kas.csv", index=False)
    levers = {"setpoint": {"label": "Temperature", "unit": "°C", "min": 16, "max": 26, "step": 0.5,
                           "effects": {"gas": "increasing"}},
              "lamps": {"min": 0, "max": 18, "step": 1, "effects": {"gas": "decreasing"}},
              "screen": {"min": 0, "max": 1, "step": 1}}
    base = {"manifest_version": "1.0", "data_source": {"adapter": "csv-file", "path": str(root / "kas.csv")}}
    registry = bootstrap.build_registry(root / "models")
    train_use_case(write(root, {**base, "use_case_id": "kas",
                                "training": {"target": "gas", "task": "regression", "drop_columns": ["too_cold"]},
                                "scenario": {"levers": levers}}), registry)
    train_use_case(write(root, {**base, "use_case_id": "kas-alarm",
                                "training": {"target": "too_cold", "task": "classification", "drop_columns": ["gas"]},
                                "scenario": {"levers": {k: {**v, "effects": {}} for k, v in levers.items()}}}), registry)
    return TestClient(bootstrap.build_app(root / "usecases", root / "models"))


BASE = {"setpoint": 20, "lamps": 8, "outside": 6.5, "screen": 0}


def run(client, scenarios, use_case="kas", base=None):
    return client.post(f"/models/{use_case}/scenario",
                       json={"base": {"record_id": "now", "features": base or BASE}, "scenarios": scenarios})


def sweep(client, use_case="kas", base=None, **params):
    return client.post(f"/models/{use_case}/scenario/sweep", json={"base": {"features": base or BASE}, **params})


def predict(client, features, use_case="kas"):
    return client.post(f"/models/{use_case}/predict", json={"records": [{"features": features}]}).json()["predictions"][0]


# --------------------------------------------------------------------------- calculate

def test_base_and_scenarios_are_calculated(kas):
    response = run(kas, [{"name": "warmer", "changes": {"setpoint": {"delta": 2}}},
                         {"name": "lamps off", "changes": {"lamps": 0}}])
    assert response.status_code == 200, response.text
    body = response.json()
    base = body["base"]["outputs"]["gas"]["value"]
    warmer, lamps_off = (s["outputs"]["gas"]["value"] for s in body["scenarios"])
    assert warmer > base        # declared: more heating, more gas
    assert lamps_off > base     # declared: less lamp heat, more gas
    assert body["scenarios"][0]["changes"][0] == {"lever": "setpoint", "label": "Temperature", "unit": "°C",
                                                  "from": 20, "to": 22}


def test_scenario_output_equals_a_normal_prediction(kas):
    """The scenario engine uses the same model and the same input handling as /predict."""
    body = run(kas, [{"name": "warmer", "changes": {"setpoint": 22}}]).json()
    assert body["base"]["outputs"]["gas"]["value"] == pytest.approx(predict(kas, BASE)["prediction"])
    scenario = body["scenarios"][0]
    assert scenario["outputs"]["gas"]["value"] == pytest.approx(predict(kas, scenario["features"])["prediction"])


def test_classification_scenarios_return_probabilities(kas):
    body = run(kas, [{"name": "colder", "changes": {"setpoint": 17}}], use_case="kas-alarm").json()
    base, colder = body["base"]["outputs"]["too_cold"], body["scenarios"][0]["outputs"]["too_cold"]
    assert set(base["probabilities"]) == {"0", "1"}
    assert colder["probabilities"]["1"] > base["probabilities"]["1"]


def test_calculation_refuses_what_check_refuses(kas):
    response = run(kas, [{"name": "x", "changes": {"outside": 1}}])
    assert response.status_code == 422 and response.json()["errors"][0]["scenario"] == "x"


def test_calculations_are_counted_in_the_metrics(kas):
    before = kas.get("/metrics").json().get("kas", {}).get("records_predicted", 0)
    run(kas, [{"name": "a", "changes": {"lamps": 2}}, {"name": "b", "changes": {"lamps": 4}}])
    assert kas.get("/metrics").json()["kas"]["records_predicted"] == before + 3  # base + 2 scenarios


# --------------------------------------------------------------------------- sweep

def test_sweep_defaults_to_the_whole_lever_range(kas):
    body = sweep(kas, lever="lamps").json()
    assert [p["value"] for p in body["points"]] == list(range(0, 19))
    assert body["lever"]["name"] == "lamps" and body["base"]["value"] == 8
    values = [p["outputs"]["gas"]["value"] for p in body["points"]]
    assert all(b <= a + 1e-6 for a, b in zip(values, values[1:]))  # declared decreasing: never goes up


def test_sweep_with_own_range_includes_the_end(kas):
    body = sweep(kas, lever="setpoint", **{"from": 18, "to": 21, "step": 0.75}).json()
    assert [p["value"] for p in body["points"]] == [18, 18.75, 19.5, 20.25, 21]


def test_sweep_flags_points_beyond_the_training_data(kas):
    points = sweep(kas, lever="setpoint").json()["points"]
    flags = {p["value"]: p["beyond_training_data"] for p in points}
    assert flags[16] and flags[26] and not flags[20]


def test_sweep_of_a_yes_no_lever(kas):
    body = sweep(kas, lever="screen").json()
    assert [p["value"] for p in body["points"]] == [0, 1]


@pytest.mark.parametrize("params, message", [
    ({"lever": "outside"}, "cannot be changed"),
    ({"lever": "setpoint", "from": 22, "to": 20}, "must go upwards"),
    ({"lever": "setpoint", "from": 10, "to": 20}, "within the allowed range 16 - 26"),
    ({"lever": "setpoint", "step": 0.01}, "more than 201 points"),
])
def test_invalid_sweeps_are_refused(kas, params, message):
    response = sweep(kas, **params)
    assert response.status_code == 422 and message in response.json()["errors"][0]["message"]


def test_sweep_step_must_be_positive(kas):
    assert sweep(kas, lever="setpoint", step=0).status_code == 422


def test_sweep_checks_the_base(kas):
    response = sweep(kas, lever="lamps", base={"setpoint": 20})
    assert response.status_code == 422 and response.json()["errors"][0]["scenario"] is None


# --------------------------------------------------------------------------- the real nursery, against the answer key

@pytest.fixture(scope="module")
def nursery(tmp_path_factory):
    root = tmp_path_factory.mktemp("nursery")
    daily = knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=4)
    df, sim, batches = make_nursery.generate(daily, compartments=10, seed=6)
    write_nursery_tables(root, df, daily, sim)
    registry = bootstrap.build_registry(root / "models")
    manifest = json.loads((ROOT / "usecases" / "nursery" / "manifest.json").read_text())
    repoint_nursery(manifest, root)
    train_use_case(write(root, manifest), registry)  # one run trains every output; the old ids work as aliases
    client = TestClient(bootstrap.build_app(root / "usecases", root / "models"))
    return client, df, sim, {b["batch_id"]: b for b in batches}


def test_documented_examples_work_on_the_real_manifest(nursery):
    client = nursery[0]
    assert client.post("/models/nursery-gas/scenario", json=EXAMPLE_SCENARIO).status_code == 200
    assert client.post("/models/nursery-gas/scenario/sweep", json=EXAMPLE_SWEEP).status_code == 200


def test_sweeps_follow_the_true_curve_of_the_simulator(nursery):
    """The temperature sweep against the simulator's answer for the last 40 (unseen) batches.

    Where temperature really matters (autumn to spring) the model must follow the true curve.
    In summer the true effect is ~0 (the sun keeps the greenhouse warm); the model still invents
    about a week there. That is a known limitation, guarded here so it cannot silently get worse.
    """
    client, df, sim, batches = nursery
    order = client.get("/models/nursery-harvest-weeks").json()["feature_schema"]["x-feature-order"]
    real_effect, invented = [], []
    for _, row in df.sort_values("start_date").tail(40).iterrows():
        features = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}
        body = client.post("/models/nursery-harvest-weeks/scenario/sweep",
                           json={"base": {"features": features}, "lever": "temp_setpoint_c", "from": 18, "to": 23}).json()
        predicted = np.array([p["outputs"]["weeks_to_harvest"]["value"] for p in body["points"]])
        true = np.array([sim.counterfactual(batches[row["batch_id"]], temp_setpoint_c=p["value"])["weeks_to_harvest"]
                         for p in body["points"]])
        if np.ptp(true) >= 0.5:
            real_effect.append(np.corrcoef(predicted, true)[0, 1])
        else:
            invented.append(np.ptp(predicted))
    assert len(real_effect) >= 15
    assert np.mean(real_effect) > 0.9, np.mean(real_effect)
    assert np.mean(invented) < 2.0, np.mean(invented)


# --------------------------------------------------------------------------- physical limits of the output

@pytest.fixture(scope="module")
def capped(tmp_path_factory):
    """Same greenhouse data, but gas may not exceed 1,000 (an artificial limit that the base breaks)."""
    root = tmp_path_factory.mktemp("capped")
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"setpoint": rng.uniform(17, 23, n).round(1), "outside": rng.uniform(0, 15, n).round(1)})
    df["gas"] = (100 * (df.setpoint - df.outside).clip(0) + rng.normal(0, 30, n)).round(1)
    df.to_csv(root / "gas.csv", index=False)
    train_use_case(write(root, {
        "manifest_version": "1.0", "use_case_id": "capped",
        "data_source": {"adapter": "csv-file", "path": str(root / "gas.csv")},
        "training": {"target": "gas", "task": "regression", "target_min": 0, "target_max": 1000},
        "scenario": {"levers": {"setpoint": {"min": 16, "max": 26, "step": 1}}},
    }), bootstrap.build_registry(root / "models"))
    return TestClient(bootstrap.build_app(root / "usecases", root / "models"))


def test_predictions_stay_within_the_physical_limits_everywhere(capped):
    hot = {"setpoint": 23, "outside": 0}  # the model says ~2,300
    p = predict(capped, hot, use_case="capped")
    assert p["prediction"] == 1000 and "above the possible maximum 1000" in p["warnings"][0]
    body = run(capped, [{"name": "warm", "changes": {"setpoint": 24}}], use_case="capped", base=hot).json()
    for row in [body["base"], body["scenarios"][0]]:
        assert row["outputs"]["gas"]["value"] == 1000 and "warning" in row["outputs"]["gas"]
    points = sweep(capped, lever="setpoint", use_case="capped", base={"setpoint": 18, "outside": 14}).json()["points"]
    values = [p["outputs"]["gas"]["value"] for p in points]
    assert min(values) >= 0 and max(values) <= 1000


@pytest.mark.parametrize("training, message", [
    ({"target": "y", "task": "regression", "target_min": 5, "target_max": 1}, "must be smaller"),
    ({"target": "y", "task": "classification", "target_min": 0}, "only apply to regression"),
])
def test_invalid_limits_fail_fast(tmp_path, training, message):
    from predictor.core.manifest import ManifestError, load_manifest
    path = write(tmp_path, {"manifest_version": "1.0", "use_case_id": "x",
                            "data_source": {"adapter": "csv-file"}, "training": training})
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)
