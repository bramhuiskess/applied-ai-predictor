"""Inputs nobody knows on the day of the question (the sun ahead): a usual value, an outlook, a wider range."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core import ranges
from predictor.core.manifest import ManifestError, load_manifest
from predictor.trainer.run import train_use_case

ASSUMPTION = {"sun": {"label": "Sun in the coming months", "typical": 1.0, "low": 0.9, "high": 1.1,
                      "low_label": "a dark year", "high_label": "a sunny year"}}


def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def manifest(root, **scenario):
    return {"manifest_version": "1.0", "use_case_id": "crop", "data_source": {"adapter": "csv-file", "path": str(root / "crop.csv")},
            "training": {"target": "harvest", "task": "regression", "target_min": 0},
            "scenario": {"levers": {"lamps": {"min": 0, "max": 16, "step": 1}}, "case_label": "seasons", **scenario}}


@pytest.fixture(scope="module")
def crop(tmp_path_factory):
    """Harvest follows the lamps and the sun of the season, which is only known afterwards."""
    root = tmp_path_factory.mktemp("assume")
    rng = np.random.default_rng(8)
    n = 1200
    df = pd.DataFrame({"lamps": rng.uniform(0, 16, n).round(0), "sun": rng.normal(1.0, 0.07, n).round(3)})
    df["harvest"] = (10 + 0.3 * df.lamps + 20 * (df.sun - 1) + rng.normal(0, 0.3, n)).round(2)
    df.to_csv(root / "crop.csv", index=False)
    result = train_use_case(write(root, manifest(root, assumptions=ASSUMPTION)), bootstrap.build_registry(root / "models"))
    return TestClient(bootstrap.build_app(root / "usecases", root / "models")), result


def test_left_out_means_a_usual_year_with_an_outlook_and_a_wider_range(crop):
    client, result = crop
    asked = client.post("/models/crop/predict", json={"records": [{"features": {"lamps": 8}}]}).json()["predictions"][0]
    assert asked["prediction"] == pytest.approx(12.4, abs=0.3)                       # 10 + 0.3 x 8 in a usual year
    assert asked["outlook"]["a dark year"] == pytest.approx(10.4, abs=0.4) and asked["outlook"]["a sunny year"] == pytest.approx(14.4, abs=0.4)
    width = result["range"]["half_width"]
    assert asked["range"]["low"] == pytest.approx(asked["outlook"]["a dark year"] - width, abs=0.15)
    assert asked["range"]["high"] == pytest.approx(asked["outlook"]["a sunny year"] + width, abs=0.15)
    assert asked["assumed"]["sun"]["value"] == 1.0 and "a dark year (0.9)" in asked["assumed"]["sun"]["note"]


def test_given_means_no_outlook_and_the_narrow_range(crop):
    client, result = crop
    known = client.post("/models/crop/predict", json={"records": [{"features": {"lamps": 8, "sun": 1.08}}]}).json()["predictions"][0]
    assert "outlook" not in known and "assumed" not in known
    assert known["prediction"] == pytest.approx(14.0, abs=0.3)
    assert known["range"]["high"] - known["range"]["low"] == pytest.approx(2 * result["range"]["half_width"], abs=0.2)


def test_scenarios_carry_the_assumption_and_the_plan_finder_uses_the_usual_year(crop):
    client, _ = crop
    body = client.post("/models/crop/scenario", json={"base": {"features": {"lamps": 8}},
                                                      "scenarios": [{"name": "more lamps", "changes": {"lamps": 12}}]}).json()
    assert body["assumed"]["sun"]["label"] == "Sun in the coming months" and body["base"]["features"]["sun"] == 1.0
    more = body["scenarios"][0]
    assert more["outputs"]["harvest"]["outlook"]["a sunny year"] > more["outputs"]["harvest"]["value"] > more["outputs"]["harvest"]["outlook"]["a dark year"]
    assert more["difference"]["harvest"]["change"] == pytest.approx(1.2, abs=0.25)   # the change itself does not depend on the weather
    best = client.post("/models/crop/scenario/best", json={"base": {"features": {"lamps": 8}},
                                                           "objective": {"output": "harvest", "goal": "highest"}}).json()
    assert best["best"]["plan"]["lamps"] == 16
    described = client.get("/models/crop").json()["scenario"]["assumptions"]
    assert described[0]["name"] == "sun" and described[0]["low_label"] == "a dark year"


def test_the_range_spans_every_assumption():
    spec = {"level": 0.8, "half_width": 1.0}
    assert ranges.around(spec, 10.0, also=[8.0, 13.0]) == {"low": 7.0, "high": 14.0, "level": 0.8}
    assert ranges.around(spec, 10.0) == {"low": 9.0, "high": 11.0, "level": 0.8}


@pytest.mark.parametrize("scenario, message", [
    ({"assumptions": {"sun": {**ASSUMPTION["sun"], "low": 1.2}}}, "low < typical < high"),
    ({"assumptions": {"lamps": ASSUMPTION["sun"]}}, "both a lever and an assumption"),
])
def test_manifest_rules(tmp_path, scenario, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(write(tmp_path, manifest(tmp_path, **scenario)) / "manifest.json")


def test_an_assumption_must_be_an_input_of_the_model(crop, tmp_path):
    pd.DataFrame({"lamps": np.arange(200) % 16, "harvest": np.arange(200) % 16 * 0.3 + 10.0}).to_csv(tmp_path / "crop.csv", index=False)
    train_use_case(write(tmp_path, manifest(tmp_path)), bootstrap.build_registry(tmp_path / "models"))
    write(tmp_path, manifest(tmp_path, assumptions=ASSUMPTION))                       # the model was trained without "sun"
    with pytest.raises(ManifestError, match="assumption 'sun' is not a numeric input"):
        bootstrap.build_app(tmp_path / "usecases", tmp_path / "models")
