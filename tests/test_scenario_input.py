"""Ticket 3: scenario input (POST /models/{id}/scenario/check)."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core.api import EXAMPLE_SCENARIO
from predictor.trainer.run import train_use_case


@pytest.fixture(scope="module")
def kas(tmp_path_factory):
    """A small greenhouse use case with three levers, one of them yes/no, and a privacy rule."""
    root = tmp_path_factory.mktemp("scenario-input")
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        "setpoint": rng.uniform(17, 23, n).round(1),
        "lamps": rng.uniform(0, 16, n).round(1),
        "outside": rng.uniform(0, 15, n).round(1),
        "screen": rng.integers(0, 2, n),
    })
    df["gas"] = (100 * (df.setpoint - df.outside).clip(0) - 25 * df.lamps - 40 * df.screen + 600
                 + rng.normal(0, 30, n)).round(1)
    df.to_csv(root / "kas.csv", index=False)
    manifest = {
        "manifest_version": "1.0", "use_case_id": "kas",
        "data_source": {"adapter": "csv-file", "path": str(root / "kas.csv")},
        "training": {"target": "gas", "task": "regression"},
        "policy": {"forbidden_fields": ["grower_name"]},
        "scenario": {"levers": {
            "setpoint": {"label": "Temperature", "unit": "°C", "min": 16, "max": 26, "step": 0.5},
            "lamps": {"min": 0, "max": 18, "step": 0.5},
            "screen": {"label": "Energy screen", "min": 0, "max": 1, "step": 1},
        }},
    }
    (root / "usecases" / "kas").mkdir(parents=True)
    (root / "usecases" / "kas" / "manifest.json").write_text(json.dumps(manifest))
    train_use_case(root / "usecases" / "kas", bootstrap.build_registry(root / "models"))
    return TestClient(bootstrap.build_app(root / "usecases", root / "models"))


BASE = {"setpoint": 20.1, "lamps": 8, "outside": 6.5, "screen": 0}


def check(client, scenarios, base=None, use_case="kas"):
    body = {"base": {"record_id": "now", "features": base or BASE}, "scenarios": scenarios}
    return client.post(f"/models/{use_case}/scenario/check", json=body)


def errors(response):
    assert response.status_code == 422, response.text
    return response.json()["errors"]


# --------------------------------------------------------------------------- happy path

def test_set_delta_and_shorthand_resolve_to_full_rows(kas):
    response = check(kas, [
        {"name": "warmer", "changes": {"setpoint": {"delta": 0.2}}},
        {"name": "lamps on 12", "changes": {"lamps": {"set": 12}}},
        {"name": "screen + 22 degrees", "changes": {"screen": 1, "setpoint": 22}},
    ])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["base"] == {"record_id": "now", "features": BASE, "warnings": []}
    warmer, lamps, screen = body["scenarios"]
    assert warmer["features"] == {**BASE, "setpoint": 20.3}  # not 20.300000000000001
    assert warmer["changes"] == [{"lever": "setpoint", "label": "Temperature", "unit": "°C", "from": 20.1, "to": 20.3}]
    assert lamps["features"]["lamps"] == 12 and lamps["changes"][0]["label"] == "lamps"
    assert screen["features"] == {**BASE, "screen": 1, "setpoint": 22}
    assert [c["lever"] for c in screen["changes"]] == ["screen", "setpoint"]
    assert all(s["warnings"] == [] for s in body["scenarios"])


def test_documented_example_works_on_the_real_nursery_manifest(tmp_path):
    """The example in /docs must keep working for the real nursery-gas use case."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import knmi
    import make_nursery

    df, _, _ = make_nursery.generate(knmi.synthetic_daily("2015-06-01", "2025-12-31"), compartments=4)
    df.to_csv(tmp_path / "nursery.csv", index=False)
    manifest = json.loads((root / "usecases" / "nursery-gas" / "manifest.json").read_text())
    manifest["data_source"]["path"] = str(tmp_path / "nursery.csv")
    (tmp_path / "usecases" / "nursery-gas").mkdir(parents=True)
    (tmp_path / "usecases" / "nursery-gas" / "manifest.json").write_text(json.dumps(manifest))
    train_use_case(tmp_path / "usecases" / "nursery-gas", bootstrap.build_registry(tmp_path / "models"))
    client = TestClient(bootstrap.build_app(tmp_path / "usecases", tmp_path / "models"))
    response = client.post("/models/nursery-gas/scenario/check", json=EXAMPLE_SCENARIO)
    assert response.status_code == 200, response.text
    assert [s["name"] for s in response.json()["scenarios"]] == ["1 degree warmer", "more light, less heat"]


def test_warnings_for_values_beyond_the_data_and_for_no_change(kas):
    body = check(kas, [
        {"name": "very warm", "changes": {"setpoint": 25.5}},
        {"name": "same as now", "changes": {"lamps": 8}},
    ]).json()
    very_warm, same = body["scenarios"]
    assert any("outside the training range" in w for w in very_warm["warnings"])
    assert same["warnings"] == ["this scenario changes nothing compared to the base"]


# --------------------------------------------------------------------------- what is refused

@pytest.mark.parametrize("changes, field, message", [
    ({"outside": 10}, "outside", "cannot be changed in a scenario; you can change: setpoint, lamps, screen"),
    ({"humidity": 80}, "humidity", "cannot be changed"),
    ({"setpoint": 27}, "setpoint", "would become 27 (set to 27), outside the allowed range 16 - 26"),
    ({"setpoint": {"delta": 6}}, "setpoint", "would become 26.1 (20.1 +6), outside the allowed range"),
    ({"screen": {"delta": -1}}, "screen", "would become -1"),
    ({"screen": 0.5}, "screen", "is not one of [0, 1]"),
])
def test_invalid_changes_name_the_scenario_and_the_field(kas, changes, field, message):
    found = errors(check(kas, [{"name": "try", "changes": changes}]))
    assert all(e["scenario"] == "try" and e["field"] == field for e in found)
    assert any(message in e["message"] for e in found)


def test_all_problems_are_reported_at_once(kas):
    found = errors(check(kas, [
        {"name": "a", "changes": {"outside": 1}},
        {"name": "b", "changes": {"setpoint": 99}},
        {"name": "c", "changes": {"lamps": 3}},
    ]))
    assert [(e["scenario"], e["field"]) for e in found] == [("a", "outside"), ("b", "setpoint")]


def test_duplicate_names_are_refused(kas):
    found = errors(check(kas, [{"name": "Plan A", "changes": {"lamps": 3}},
                               {"name": "plan a ", "changes": {"lamps": 4}}]))
    assert found[0]["field"] == "name" and "used twice" in found[0]["message"]


@pytest.mark.parametrize("scenarios", [
    [],                                                              # nothing to calculate
    [{"name": f"s{i}", "changes": {"lamps": 1}} for i in range(21)],  # more than 20
    [{"name": "x", "changes": {}}],                                   # no changes
    [{"name": "", "changes": {"lamps": 1}}],                          # no name
    [{"name": "x", "changes": {"lamps": {"set": 3, "delta": 1}}}],    # both
    [{"name": "x", "changes": {"lamps": {}}}],                        # neither
    [{"name": "x", "changes": {"lamps": {"sett": 3}}}],               # typo
    [{"name": "x", "changes": {"lamps": "veel"}}],                    # not a number
])
def test_malformed_requests_are_refused(kas, scenarios):
    assert check(kas, scenarios).status_code == 422


def test_nan_is_refused(kas):
    body = json.dumps({"base": {"features": BASE}, "scenarios": [{"name": "x", "changes": {"lamps": 1}}]})
    body = body.replace('"lamps": 1}', '"lamps": NaN}')
    response = kas.post("/models/kas/scenario/check", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_schema_version_is_checked(kas):
    body = {"schema_version": "2.0", "base": {"features": BASE}, "scenarios": [{"name": "x", "changes": {"lamps": 1}}]}
    assert kas.post("/models/kas/scenario/check", json=body).status_code == 422


# --------------------------------------------------------------------------- the base

def test_invalid_base_is_reported_without_a_scenario_name(kas):
    found = errors(check(kas, [{"name": "x", "changes": {"lamps": 1}}], base={"setpoint": 20, "lamps": 8}))
    assert {e["field"] for e in found} == {"outside", "screen"}
    assert all(e["scenario"] is None for e in found)


def test_privacy_policy_applies_to_the_base(kas):
    [error] = errors(check(kas, [{"name": "x", "changes": {"lamps": 1}}], base={**BASE, "grower_name": "Jan"}))
    assert error["field"] == "grower_name" and "forbidden" in error["message"]


# --------------------------------------------------------------------------- use cases

def test_use_case_without_levers_explains_what_to_do(client):
    from conftest import GREENHOUSE
    response = check(client, [{"name": "x", "changes": {"heating_on": 1}}], base=GREENHOUSE,
                     use_case="greenhouse-climate")
    assert response.json()["detail"] == "invalid scenario request"
    assert "add a 'scenario' section" in errors(response)[0]["message"]


def test_unknown_and_untrained_use_cases(client):
    assert check(client, [{"name": "x", "changes": {"a": 1}}], use_case="nope").status_code == 404
    assert check(client, [{"name": "x", "changes": {"a": 1}}], use_case="untrained").status_code == 409


def test_scenario_endpoint_is_documented(kas):
    spec = kas.get("/openapi.json").json()
    operation = spec["paths"]["/models/{use_case_id}/scenario/check"]["post"]
    assert "levers" in operation["description"]
    schema = spec["components"]["schemas"]["ScenarioRequest"]
    assert schema["examples"][0]["scenarios"][0]["name"] == "1 degree warmer"
