"""Certainty per scenario: how many comparable cases the training data has."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import predictor.core.certainty as certainty_module
from predictor import bootstrap
from predictor.core.certainty import SupportIndex, build_support, certainty
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_nursery  # noqa: E402


# --------------------------------------------------------------------------- the index

def grid_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.integers(0, 3, n)])


def index(matrix, importance=None):
    return SupportIndex(build_support(matrix, ["number", "number", "categorical"], ["a", "b", "kind"],
                                      [None, None, ["x", "y", "z"]], importance))


def test_a_typical_case_has_about_ten_comparable_cases():
    matrix = grid_data()
    counts = index(matrix).comparable(matrix)
    assert 8 <= np.median(counts) <= 14


def test_far_away_has_no_comparable_cases():
    assert index(grid_data()).comparable(np.array([[50.0, 50.0, 0.0]])) == [0]


def test_another_category_counts_as_different():
    matrix = grid_data()
    same = index(matrix).comparable(np.array([[5.0, 5.0, 0.0]]))[0]
    unknown = index(matrix).comparable(np.array([[5.0, 5.0, 7.0]]))[0]  # a category not in the data
    assert unknown < same


def test_inputs_the_model_ignores_barely_matter():
    """With all importance on 'a', a different 'b' should hardly reduce the count."""
    matrix = grid_data()
    idx = index(matrix, importance={"a": 0.98, "b": 0.0, "kind": 0.0})
    near = idx.comparable(np.array([[5.0, 5.0, 1.0]]))[0]
    other_b = idx.comparable(np.array([[5.0, 9.5, 1.0]]))[0]
    assert other_b >= 0.5 * near


def test_large_training_sets_are_subsampled_and_counts_scaled(monkeypatch):
    monkeypatch.setattr(certainty_module, "MAX_POINTS", 200)
    matrix = grid_data(n=800)
    data = build_support(matrix, ["number", "number", "categorical"], ["a", "b", "kind"], [None, None, ["x", "y", "z"]], None)
    assert len(data["points"]) == 200 and data["scale"] == 4.0 and data["rows"] == 800
    assert 5 <= np.median(SupportIndex(data).comparable(matrix)) <= 25


@pytest.mark.parametrize("count, level, text", [
    (0, "low", "Uncertain: no comparable batches"),
    (1, "low", "only 1 comparable batch in"),
    (5, "medium", "Based on only 5 comparable batches"),
    (25, "high", "Based on 25 comparable batches in the history."),
])
def test_levels_and_messages(count, level, text):
    c = certainty(count, "batches")
    assert c["level"] == level and text in c["message"] and c["comparable_cases"] == count


# --------------------------------------------------------------------------- training and the API

def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def kas(tmp_path_factory):
    root = tmp_path_factory.mktemp("certainty")
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({"setpoint": rng.uniform(18, 22, n).round(1), "outside": rng.uniform(0, 15, n).round(1)})
    df["gas"] = (100 * (df.setpoint - df.outside).clip(0) + rng.normal(0, 30, n)).round(1)
    df.to_csv(root / "kas.csv", index=False)
    base = {"manifest_version": "1.0", "data_source": {"adapter": "csv-file", "path": str(root / "kas.csv")},
            "training": {"target": "gas", "task": "regression"}}
    scenario = {"levers": {"setpoint": {"min": 14, "max": 26, "step": 1}}, "case_label": "days"}
    registry = bootstrap.build_registry(root / "models")
    results = {
        "kas": train_use_case(write(root, {**base, "use_case_id": "kas", "scenario": scenario}), registry),
        "kas-private": train_use_case(write(root, {**base, "use_case_id": "kas-private", "scenario": scenario,
                                                   "policy": {"contains_personal_data": True}}), registry),
        "kas-plain": train_use_case(write(root, {**base, "use_case_id": "kas-plain"}), registry),
    }
    return root, TestClient(bootstrap.build_app(root / "usecases", root / "models")), results


def test_the_map_is_only_stored_where_needed_and_allowed(kas):
    root, _, results = kas
    stored = {uc: (root / "models" / uc / "1" / "support.json").is_file() for uc in results}
    assert stored == {"kas": True, "kas-private": False, "kas-plain": False}
    assert results["kas"]["certainty"]["available"] is True
    assert results["kas-private"]["certainty"] == {"available": False,
                                                   "reason": "not stored for use cases with personal data"}


def test_scenarios_and_sweeps_report_certainty(kas):
    _, client, _ = kas
    body = client.post("/models/kas/scenario", json={
        "base": {"features": {"setpoint": 20, "outside": 7}},
        "scenarios": [{"name": "normal", "changes": {"setpoint": 21}},
                      {"name": "never seen", "changes": {"setpoint": 14}}]}).json()
    normal, never = body["scenarios"]
    assert body["base"]["certainty"]["level"] != "low"  # a typical day: about ten comparable days
    assert normal["certainty"]["comparable_cases"] > never["certainty"]["comparable_cases"]
    assert never["certainty"]["level"] == "low"
    assert never["summary"].endswith("so this is mostly extrapolation.") and "comparable days" in never["summary"]
    points = client.post("/models/kas/scenario/sweep", json={"base": {"features": {"setpoint": 20, "outside": 7}},
                                                             "lever": "setpoint"}).json()["points"]
    levels = {p["value"]: p["certainty"]["level"] for p in points}
    assert levels[20] != "low" and levels[14] == "low" and levels[26] == "low"


def test_no_certainty_without_the_map(kas):
    _, client, _ = kas
    s = client.post("/models/kas-private/scenario", json={
        "base": {"features": {"setpoint": 20, "outside": 7}},
        "scenarios": [{"name": "x", "changes": {"setpoint": 21}}]}).json()["scenarios"][0]
    assert s["certainty"] is None and "comparable" not in s["summary"]


# --------------------------------------------------------------------------- the answer key

def test_higher_certainty_means_a_smaller_error_in_the_effect(tmp_path):
    """Sweeps through the real engine on unseen batches, against the simulator's true effects."""
    df, sim, batches = make_nursery.generate(knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=7),
                                             compartments=10, seed=13)
    df.to_csv(tmp_path / "nursery.csv", index=False)
    manifest = json.loads((ROOT / "usecases" / "nursery-harvest-weeks" / "manifest.json").read_text())
    manifest["data_source"]["path"] = str(tmp_path / "nursery.csv")
    train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    client = TestClient(bootstrap.build_app(tmp_path / "usecases", tmp_path / "models"))
    order = client.get("/models/nursery-harvest-weeks").json()["feature_schema"]["x-feature-order"]
    by_id = {b["batch_id"]: b for b in batches}
    level = df["weeks_to_harvest"].mean()
    errors = {"low": [], "medium": [], "high": []}
    for _, row in df.sort_values("start_date").tail(40).iterrows():
        features = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}
        for lever in ("temp_setpoint_c", "co2_ppm", "plant_density_m2"):
            body = client.post("/models/nursery-harvest-weeks/scenario/sweep",
                               json={"base": {"features": features}, "lever": lever}).json()
            b = by_id[row["batch_id"]]
            base_true = sim.simulate(b)["weeks_to_harvest"]
            for p in body["points"]:
                true = sim.counterfactual(b, **{lever: p["value"]})["weeks_to_harvest"] - base_true
                errors[p["certainty"]["level"]].append(abs(p["difference"]["weeks_to_harvest"]["change"] - true) / level)
    assert all(len(v) >= 30 for v in errors.values()), {k: len(v) for k, v in errors.items()}
    assert np.mean(errors["high"]) < np.mean(errors["low"]) * 0.7, {k: np.mean(v) for k, v in errors.items()}
