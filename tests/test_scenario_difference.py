"""Ticket 5: the difference between base and scenario (units, %, euros, margin of error)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core.manifest import ManifestError, ScenarioConfig, load_manifest
from predictor.core.scenario import compare, summarise
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_nursery  # noqa: E402


def config(outputs):
    return ScenarioConfig.model_validate({"levers": {"setpoint": {"min": 15, "max": 25, "step": 1}}, "outputs": outputs})


REGRESSION = {"task_type": "regression", "target_unit": "m³", "metrics": {"mae": 100.0}}


# --------------------------------------------------------------------------- the calculation

def test_change_percent_and_margin():
    c = compare(config({}), REGRESSION, {}, {"gas": {"value": 1000.0}}, {}, {"gas": {"value": 850.0}})
    gas = c["outputs"]["gas"]
    assert gas == {"base": 1000.0, "scenario": 850.0, "change": -150.0, "change_pct": -15.0, "eur": None,
                   "margin": 100.0, "exceeds_margin": True}
    assert c["net_eur"] is None


def test_a_change_smaller_than_the_margin_is_not_a_clear_effect():
    gas = compare(config({}), REGRESSION, {}, {"gas": {"value": 1000.0}}, {}, {"gas": {"value": 1050.0}})["outputs"]["gas"]
    assert gas["exceeds_margin"] is False


def test_no_percentage_from_zero():
    gas = compare(config({}), REGRESSION, {}, {"gas": {"value": 0.0}}, {}, {"gas": {"value": 40.0}})["outputs"]["gas"]
    assert gas["change"] == 40.0 and gas["change_pct"] is None


def test_fixed_value_in_euros():
    c = compare(config({"plants": {"value_eur": 2.10}}), REGRESSION, {}, {"plants": {"value": 10_000.0}},
                {}, {"plants": {"value": 10_500.0}})
    assert c["outputs"]["plants"]["eur"] == 1050.0 and c["net_eur"] == 1050.0


def test_price_from_the_row_itself_as_a_cost():
    """Gas is paid at the grower's own price from the input; a cost that drops is worth money."""
    cfg = config({"gas": {"price_from": "gas_price", "is_cost": True}})
    c = compare(cfg, REGRESSION, {"gas_price": 0.40}, {"gas": {"value": 1000.0}},
                {"gas_price": 0.40}, {"gas": {"value": 800.0}})
    assert c["outputs"]["gas"]["eur"] == 80.0  # 200 m3 less at 0.40


def test_each_row_uses_its_own_price():
    cfg = config({"gas": {"price_from": "gas_price", "is_cost": True}})
    c = compare(cfg, REGRESSION, {"gas_price": 0.40}, {"gas": {"value": 1000.0}},
                {"gas_price": 0.50}, {"gas": {"value": 1000.0}})
    assert c["outputs"]["gas"]["change"] == 0 and c["outputs"]["gas"]["eur"] == -100.0


def test_net_worth_adds_up_all_outputs():
    cfg = config({"plants": {"value_eur": 2.0}, "gas": {"price_from": "gas_price", "is_cost": True}})
    base = {"plants": {"value": 1000.0}, "gas": {"value": 1000.0}}
    now = {"plants": {"value": 1100.0}, "gas": {"value": 1300.0}}
    c = compare(cfg, REGRESSION, {"gas_price": 0.5}, base, {"gas_price": 0.5}, now)
    assert c["net_eur"] == 200.0 - 150.0
    text = summarise(cfg, REGRESSION, "warmer", c)
    assert text.endswith("Net worth +€50.")


def test_classification_compares_the_chance_of_yes():
    cls = {"task_type": "classification"}
    before = {"failure": {"value": "0", "probabilities": {"0": 0.9, "1": 0.1}}}
    after = {"failure": {"value": "1", "probabilities": {"0": 0.35, "1": 0.65}}}
    c = compare(config({}), cls, {}, before, {}, after)["outputs"]["failure"]
    assert c["label_changed"] and c["positive_class"] == "1"
    assert c["probability_change"] == {"0": -0.55, "1": 0.55}
    assert "goes from 10% to 65%" in summarise(config({}), cls, "worn tool", {"outputs": {"failure": c}, "net_eur": None})


@pytest.mark.parametrize("now, expected", [
    (850.0, "warmer: gas use 15.0% lower (-150 m³, worth +€60)."),
    (1050.0, "warmer: gas use 5.0% higher (+50.0 m³, worth -€20); this is within the model's typical error of "
             "±100 m³, so treat it as no clear effect."),
    (1000.0, "warmer: gas use does not change."),
])
def test_summary_in_plain_language(now, expected):
    cfg = config({"gas": {"label": "Gas use", "price_from": "gas_price", "is_cost": True}})
    c = compare(cfg, REGRESSION, {"gas_price": 0.4}, {"gas": {"value": 1000.0}}, {"gas_price": 0.4}, {"gas": {"value": now}})
    assert summarise(cfg, REGRESSION, "warmer", c) == expected


# --------------------------------------------------------------------------- manifest

@pytest.mark.parametrize("output, message", [
    ({"value_eur": 1, "price_from": "price"}, "either 'value_eur' or 'price_from'"),
    ({"is_cost": True}, "'is_cost' goes with 'price_from'"),
    ({"price_from": "gas"}, "is an output, not an input"),
])
def test_invalid_money_settings_fail_fast(tmp_path, output, message):
    directory = tmp_path / "usecases" / "kas"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "manifest_version": "1.0", "use_case_id": "kas", "data_source": {"adapter": "csv-file"},
        "training": {"target": "gas"},
        "scenario": {"levers": {"setpoint": {"min": 15, "max": 25, "step": 1}}, "outputs": {"gas": output}}}))
    with pytest.raises(ManifestError, match=message):
        load_manifest(directory)


# --------------------------------------------------------------------------- the API, on the real nursery manifests

@pytest.fixture(scope="module")
def nursery(tmp_path_factory):
    root = tmp_path_factory.mktemp("nursery-diff")
    df, sim, batches = make_nursery.generate(knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=5),
                                             compartments=10, seed=12)
    df.to_csv(root / "nursery.csv", index=False)
    registry = bootstrap.build_registry(root / "models")
    for use_case in ("nursery-gas", "nursery-harvest-weeks"):
        manifest = json.loads((ROOT / "usecases" / use_case / "manifest.json").read_text())
        manifest["data_source"]["path"] = str(root / "nursery.csv")
        directory = root / "usecases" / use_case
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        train_use_case(directory, registry)
    client = TestClient(bootstrap.build_app(root / "usecases", root / "models"))
    return client, df, sim, {b["batch_id"]: b for b in batches}


def features_of(client, use_case, row):
    order = client.get(f"/models/{use_case}").json()["feature_schema"]["x-feature-order"]
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order}


def test_scenarios_return_difference_money_and_a_summary(nursery):
    client, df, _, _ = nursery
    row = df[df["start_week"].between(40, 48)].iloc[-1]
    body = client.post("/models/nursery-gas/scenario", json={
        "base": {"features": features_of(client, "nursery-gas", row)},
        "scenarios": [{"name": "more light", "changes": {"light_hours": {"delta": 3}}}],
    }).json()
    s = body["scenarios"][0]
    gas = s["difference"]["gas_m3"]
    assert gas["change"] == pytest.approx(s["outputs"]["gas_m3"]["value"] - body["base"]["outputs"]["gas_m3"]["value"])
    assert gas["change"] <= 0  # declared: more lamp light never costs gas
    assert gas["eur"] == pytest.approx(-gas["change"] * row["gas_price_eur_m3"], abs=0.01)
    assert s["net_eur"] == gas["eur"]
    assert s["summary"].startswith("more light: gas use")


def test_sweep_points_carry_the_difference(nursery):
    client, df, _, _ = nursery
    row = df.iloc[-5]
    body = client.post("/models/nursery-harvest-weeks/scenario/sweep", json={
        "base": {"features": features_of(client, "nursery-harvest-weeks", row)}, "lever": "co2_ppm"}).json()
    base = body["base"]["outputs"]["weeks_to_harvest"]["value"]
    for p in body["points"]:
        d = p["difference"]["weeks_to_harvest"]
        assert d["change"] == pytest.approx(p["outputs"]["weeks_to_harvest"]["value"] - base)
        assert d["exceeds_margin"] == (abs(d["change"]) >= d["margin"])


def test_model_metadata_shows_how_money_is_counted(nursery):
    client = nursery[0]
    [output] = client.get("/models/nursery-gas").json()["scenario"]["outputs"]
    assert output["price_from"] == "gas_price_eur_m3" and output["is_cost"] is True
    assert output["margin"] > 0


def test_clear_effects_point_the_right_way(nursery):
    """The margin rule against the answer key: when the engine says 'clear effect', it is right."""
    client, df, sim, batches = nursery
    right = []
    for _, row in df.sort_values("start_date").tail(60).iterrows():
        features = features_of(client, "nursery-harvest-weeks", row)
        requested = [{"name": f"{lever} {d:+}", "changes": {lever: {"delta": d}}}
                     for lever, d in [("temp_setpoint_c", 1.0), ("temp_setpoint_c", -1.0), ("co2_ppm", 200.0)]
                     if 17 <= features["temp_setpoint_c"] + (d if lever == "temp_setpoint_c" else 0) <= 23
                     and features["co2_ppm"] + (d if lever == "co2_ppm" else 0) <= 1100]
        body = client.post("/models/nursery-harvest-weeks/scenario",
                           json={"base": {"features": features}, "scenarios": requested}).json()
        base_true = sim.simulate(batches[row["batch_id"]])["weeks_to_harvest"]
        for s in body["scenarios"]:
            d = s["difference"]["weeks_to_harvest"]
            if not d["exceeds_margin"]:
                continue
            lever, delta = s["name"].split()[0], float(s["name"].split()[1])
            b = batches[row["batch_id"]]
            true = sim.counterfactual(b, **{lever: b[lever] + delta})["weeks_to_harvest"] - base_true
            right.append(np.sign(d["change"]) == np.sign(round(true, 6)))
    assert len(right) >= 20
    assert np.mean(right) >= 0.85, np.mean(right)


@pytest.mark.parametrize("eur, text", [(2292.4, "+€2,292"), (-20.0, "-€20"), (-0.00266, "-€0.0027"),
                                       (0.04, "+€0.04"), (0.0, "€0")])
def test_small_amounts_keep_their_sign_and_size(eur, text):
    """Per m² a degree warmer costs fractions of a cent: that must not read as '+€0'."""
    from predictor.core.scenario import _eur_text, _round_eur
    assert _eur_text(_round_eur(eur)) == text
