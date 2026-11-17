"""The best-plan finder: try every setting of the levers, advise the best one if it is worth mentioning."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core import plan as planning
from predictor.core.manifest import ScenarioConfig
from predictor.trainer.run import train_use_case


# --------------------------------------------------------------------------- the grid

def test_a_lever_is_tried_on_its_own_steps_and_thinned_when_there_are_too_many():
    assert planning.lever_values(18, 22, 1, at_most=10) == [18, 19, 20, 21, 22]
    thinned = planning.lever_values(0, 100, 1, at_most=5)
    assert thinned == [0, 25, 50, 75, 100]                       # evenly, and the ends are kept


def test_the_grid_is_every_combination_within_the_limit():
    scenario = ScenarioConfig.model_validate({"levers": {"a": {"min": 0, "max": 2, "step": 1}, "b": {"min": 10, "max": 20, "step": 5}}})
    plans = planning.grid(scenario, ["a", "b"])
    assert len(plans) == 9 and {"a": 2, "b": 15} in plans
    wide = ScenarioConfig.model_validate({"levers": {n: {"min": 0, "max": 100, "step": 1} for n in "abc"}})
    assert len(planning.grid(wide, ["a", "b", "c"])) <= planning.MAX_PLANS


def test_the_typical_error_of_the_net_result_combines_the_priced_outputs():
    margins, prices = {"harvest": 0.6, "gas": 50.0, "weeks": 1.0}, {"harvest": 3.0, "gas": -0.02, "weeks": None}
    assert planning.typical_error_of_objective(margins, prices, None) == pytest.approx((1.8 ** 2 + 1.0 ** 2) ** 0.5)
    assert planning.typical_error_of_objective(margins, prices, {"output": "weeks", "goal": "lowest"}) == 1.0


# --------------------------------------------------------------------------- the API

def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def farm(tmp_path_factory):
    """Yield is best around 21 degrees and with more lamp hours; gas rises with the setpoint."""
    root = tmp_path_factory.mktemp("plan")
    rng = np.random.default_rng(6)
    n = 1500
    df = pd.DataFrame({"setpoint": rng.uniform(17, 25, n).round(1), "lamps": rng.uniform(0, 16, n).round(0),
                       "outside": rng.uniform(0, 15, n).round(1)})
    df["harvest"] = (30 - 0.8 * (df.setpoint - 21) ** 2 + 0.5 * df.lamps + rng.normal(0, 0.5, n)).round(2)
    df["gas"] = (10 * (df.setpoint - df.outside).clip(0) + rng.normal(0, 3, n)).round(1)
    df.to_csv(root / "farm.csv", index=False)
    levers = {"setpoint": {"min": 17, "max": 25, "step": 0.5}, "lamps": {"min": 0, "max": 16, "step": 2}}
    base = {"manifest_version": "1.0", "data_source": {"adapter": "csv-file", "path": str(root / "farm.csv")}}
    registry = bootstrap.build_registry(root / "models")
    train_use_case(write(root, {**base, "use_case_id": "farm",
                                "training": {"outputs": {"harvest": {"unit": "kg"}, "gas": {"unit": "m3", "target_min": 0}}},
                                "scenario": {"levers": levers, "case_label": "days",
                                             "outputs": {"harvest": {"label": "Harvest", "value_eur": 2.0},
                                                         "gas": {"label": "Gas", "value_eur": -0.05}}}}), registry)
    train_use_case(write(root, {**base, "use_case_id": "farm-unpriced",
                                "training": {"target": "harvest", "task": "regression", "drop_columns": ["gas"]},
                                "scenario": {"levers": {"setpoint": {"min": 12, "max": 30, "step": 1}}}}), registry)
    return TestClient(bootstrap.build_app(root / "usecases", root / "models"))


BASE = {"features": {"setpoint": 18, "lamps": 4, "outside": 7}}


def best(client, use_case="farm", **options):
    return client.post(f"/models/{use_case}/scenario/best", json={"base": BASE, **options})


def test_it_finds_the_plan_with_the_highest_net_result(farm):
    body = best(farm).json()
    plan = body["best"]["plan"]
    assert 20 <= plan["setpoint"] <= 22 and plan["lamps"] == 16      # the top of the yield curve, all lamps
    assert body["best"]["net_eur"] > 15 and body["best"]["gain"] == pytest.approx(body["best"]["net_eur"])
    assert body["searched"]["plans"] == 17 * 9 and body["objective"] == {"output": "net_eur", "goal": "highest"}
    assert body["advice"].startswith("Best plan for the highest net result in euros")
    gains = [a["gain"] for a in body["alternatives"]]
    assert gains == sorted(gains, reverse=True) and gains[0] <= body["best"]["gain"]
    assert "range" in body["best"]["outputs"]["harvest"] and body["best"]["certainty"]["level"] in ("low", "medium", "high")


def test_the_objective_can_be_one_output(farm):
    body = best(farm, objective={"output": "gas", "goal": "lowest"}).json()
    assert body["best"]["plan"]["setpoint"] == 17                     # the least gas: as cold as allowed
    body = best(farm, objective={"output": "harvest", "goal": "highest"}, levers=["setpoint"]).json()
    assert set(body["best"]["plan"]) == {"setpoint"} and 20 <= body["best"]["plan"]["setpoint"] <= 22


def test_a_gain_too_small_to_tell_apart_is_not_advised(farm):
    at_the_top = {"features": {"setpoint": 21, "lamps": 16, "outside": 7}}
    body = farm.post("/models/farm/scenario/best", json={"base": at_the_top}).json()
    assert body["best"] is None and body["advice"].startswith("Keep the current settings")
    assert body["searched"]["min_gain"] == pytest.approx(planning.MIN_GAIN_SHARE * body["searched"]["typical_error"], rel=1e-3)
    # the caller can set the bar himself
    assert best(farm, min_gain=1000).json()["best"] is None
    assert best(farm, min_gain=1000).json()["best_below_min_gain"]["gain"] > 15
    assert best(farm, min_gain=0).json()["best"] is not None


def test_plans_outside_the_training_data_are_left_out(farm):
    body = best(farm, "farm-unpriced", objective={"output": "harvest", "goal": "highest"}).json()
    assert body["searched"]["left_out_outside_training_data"] >= 4    # the lever runs from 12 to 30, the data from 17 to 25
    assert 17 <= body["best"]["plan"]["setpoint"] <= 25


def test_a_certainty_limit_is_optional(farm):
    free, strict = best(farm).json(), best(farm, min_certainty="high").json()
    assert free["searched"]["left_out_low_certainty"] == 0
    assert strict["searched"]["trusted"] <= free["searched"]["trusted"]
    if strict["best"]:
        assert strict["best"]["certainty"]["level"] == "high"


@pytest.mark.parametrize("use_case, options, message", [
    ("farm-unpriced", {}, "no output of this use case has a price"),
    ("farm", {"levers": ["outside"]}, "cannot be changed in a scenario"),
    ("farm", {"objective": {"output": "profit", "goal": "highest"}}, "is not an output"),
])
def test_clear_errors(farm, use_case, options, message):
    response = best(farm, use_case, **options)
    assert response.status_code == 422 and message in json.dumps(response.json())
