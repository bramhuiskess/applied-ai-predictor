"""A range around every prediction: measured on rows the model did not learn from."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core import ranges
from predictor.trainer.run import train_use_case


# --------------------------------------------------------------------------- the rule

def test_the_half_width_covers_the_asked_share_of_unseen_rows():
    errors = list(range(1, 101))                      # 1..100
    assert ranges.conformal_half_width(errors, 0.8) == 81      # ceil(101 * 0.8) = 81st smallest
    assert ranges.conformal_half_width([1.0] * 5) is None      # too few rows to say anything


def test_less_certain_is_wider_and_never_narrower():
    errors = [1.0] * 40 + [3.0] * 40 + [2.0] * 40
    levels = ["high"] * 40 + ["medium"] * 40 + ["low"] * 40    # by chance "low" came out better than "medium"
    spec = ranges.build(errors, levels)
    assert spec["by_certainty"] == {"low": 3.0, "medium": 3.0, "high": 1.0}
    assert ranges.half_width(spec, "high") == 1.0 and ranges.half_width(spec, None) == spec["half_width"]


def test_a_level_with_too_few_rows_uses_the_overall_width():
    spec = ranges.build([1.0] * 50 + [9.0] * 5, ["high"] * 50 + ["low"] * 5)
    assert "by_certainty" not in spec and ranges.half_width(spec, "low") == spec["half_width"]


def test_the_range_stays_within_what_is_physically_possible():
    spec = {"level": 0.8, "half_width": 5.0}
    assert ranges.around(spec, 2.0, None, [0, None]) == {"low": 0, "high": 7.0, "level": 0.8}
    assert ranges.around(None, 2.0) is None


# --------------------------------------------------------------------------- training and the API

def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def kas(tmp_path_factory):
    root = tmp_path_factory.mktemp("ranges")
    rng = np.random.default_rng(4)
    n = 900
    df = pd.DataFrame({"setpoint": rng.uniform(18, 22, n).round(1), "outside": rng.uniform(0, 15, n).round(1)})
    df["gas"] = (10 * (df.setpoint - df.outside).clip(0) + rng.normal(0, 8, n)).clip(0).round(1)
    df["broken"] = (df.gas > 120).astype(int)
    df.to_csv(root / "kas.csv", index=False)
    source = {"manifest_version": "1.0", "data_source": {"adapter": "csv-file", "path": str(root / "kas.csv")}}
    scenario = {"levers": {"setpoint": {"min": 14, "max": 26, "step": 1}}, "case_label": "days"}
    registry = bootstrap.build_registry(root / "models")
    results = {
        "kas": train_use_case(write(root, {**source, "use_case_id": "kas", "scenario": scenario,
                                           "training": {"target": "gas", "task": "regression", "target_min": 0,
                                                        "drop_columns": ["broken"]}}), registry),
        "kas-plain": train_use_case(write(root, {**source, "use_case_id": "kas-plain",
                                                 "training": {"target": "gas", "task": "regression", "drop_columns": ["broken"]}}), registry),
        "kas-broken": train_use_case(write(root, {**source, "use_case_id": "kas-broken",
                                                  "training": {"target": "broken", "task": "classification", "drop_columns": ["gas"]}}), registry),
    }
    return TestClient(bootstrap.build_app(root / "usecases", root / "models")), results


def test_training_measures_the_range_and_checks_it_on_the_test_rows(kas):
    _, results = kas
    r = results["kas"]["range"]
    assert r["level"] == 0.8 and r["calibration_rows"] >= 500
    assert 8 < r["half_width"] < 14                      # noise sd 8: 80% of a normal error lies within 1.28 sd
    assert 0.72 <= r["test_coverage"] <= 0.88            # close to the promised 80% on rows it never saw
    assert "80% range around a prediction" in Path(results["kas"]["report"]).read_text()
    assert "range" not in results["kas-broken"]          # a yes/no output has probabilities instead


def test_predictions_and_scenarios_come_with_their_range(kas):
    client, results = kas
    features = {"setpoint": 20, "outside": 7}
    p = client.post("/models/kas/predict", json={"records": [{"features": features}]}).json()["predictions"][0]
    assert p["range"]["low"] < p["prediction"] < p["range"]["high"] and p["range"]["level"] == 0.8
    body = client.post("/models/kas/scenario", json={"base": {"features": features},
                                                     "scenarios": [{"name": "never seen", "changes": {"setpoint": 14}}]}).json()
    base, far = body["base"]["outputs"]["gas"], body["scenarios"][0]["outputs"]["gas"]
    assert base["range"]["low"] < base["value"] < base["range"]["high"]
    width = lambda o: o["range"]["high"] - o["range"]["low"]
    assert width(far) >= width(base) - 1e-6              # few comparable days: at least as wide
    plain = client.post("/models/kas-plain/predict", json={"records": [{"features": features}]}).json()["predictions"][0]
    assert plain["range"]["high"] - plain["range"]["low"] == pytest.approx(2 * results["kas-plain"]["range"]["half_width"], abs=1e-4)
    no = client.post("/models/kas-broken/predict", json={"records": [{"features": features}]}).json()["predictions"][0]
    assert "range" not in no


def test_a_range_never_goes_below_the_possible_minimum(kas):
    client, _ = kas
    body = client.post("/models/kas/scenario", json={"base": {"features": {"setpoint": 18, "outside": 15}},
                                                     "scenarios": [{"name": "cold", "changes": {"setpoint": 14}}]}).json()
    cold = body["scenarios"][0]["outputs"]["gas"]
    assert cold["value"] < 15 and cold["range"]["low"] == 0      # gas cannot be negative


def test_a_level_without_its_own_rows_is_never_narrower_than_a_more_certain_one():
    spec = {"level": 0.8, "half_width": 5.0, "by_certainty": {"medium": 6.0, "high": 4.0}}
    assert ranges.half_width(spec, "low") == 6.0
