"""Joins end to end: training with lookup tables, the snapshot in the model version, and the service filling joined inputs.

Uses the real nursery-gas manifest (compartment key join + weather window join) on generated data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import repoint_nursery, write_nursery_tables
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.core.manifest import ManifestError
from predictor.trainer.errors import TrainingError
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_nursery  # noqa: E402

DERIVED = ["area_m2", "lamp_type", "u_value", "outside_temp_avg_c", "radiation_avg_j_cm2"]


def write(root: Path, manifest: dict) -> Path:
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def nursery(tmp_path_factory):
    root = tmp_path_factory.mktemp("nursery-joins")
    daily = knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=21)
    df, sim, _ = make_nursery.generate(daily, compartments=6, seed=31)
    write_nursery_tables(root, df, daily, sim)
    manifest = repoint_nursery(json.loads((ROOT / "usecases" / "nursery-gas" / "manifest.json").read_text()), root)
    registry = bootstrap.build_registry(root / "models")
    result = train_use_case(write(root, manifest), registry)
    client = TestClient(bootstrap.build_app(root / "usecases", root / "models"))
    return {"root": root, "df": df, "daily": daily, "sim": sim, "client": client, "result": result, "manifest": manifest}


def order(client) -> list[str]:
    return client.get("/models/nursery-gas").json()["feature_schema"]["x-feature-order"]


def full_features(client, row) -> dict:
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in order(client)}


def join_inputs_only(client, row) -> dict:
    """What a caller sends when the system looks the rest up: no weather, no compartment attributes."""
    features = {k: v for k, v in full_features(client, row).items() if k not in DERIVED}
    return {**features, "compartment": row["compartment"], "start_date": row["start_date"]}


def predict(client, features: dict) -> dict:
    response = client.post("/models/nursery-gas/predict", json={"records": [{"features": features}]})
    assert response.status_code == 200, response.text
    return response.json()["predictions"][0]


# --------------------------------------------------------------------------- training

def test_the_model_version_carries_the_join_snapshot(nursery):
    model_dir = nursery["root"] / "models" / "nursery-gas" / "1"
    assert (model_dir / "joins.json").is_file()
    assert (model_dir / "lookups" / "weather.npz").is_file() and (model_dir / "lookups" / "compartment.npz").is_file()
    spec = json.loads((model_dir / "joins.json").read_text())
    assert [j["name"] for j in spec["joins"]] == ["compartment", "weather"]


def test_joined_columns_become_model_features_and_join_only_inputs_do_not(nursery):
    features = nursery["result"]["features"]
    assert set(DERIVED) <= set(features)
    assert "compartment" not in features and "start_date" not in features


def test_descriptor_and_report_describe_the_joins(nursery):
    joins = {j["name"]: j for j in nursery["result"]["joins"]}
    assert joins["compartment"]["matched"] == 1.0 and joins["weather"]["matched"] == 1.0
    assert len(joins["weather"]["sha256"]) == 64 and joins["weather"]["columns"] == ["outside_temp_avg_c", "radiation_avg_j_cm2"]
    assert nursery["result"]["dataset"]["rows"] == len(nursery["df"])
    report = Path(nursery["result"]["report"]).read_text()
    assert "## Joins" in report and "window [start_date, start_date +10 weeks)" in report and "compartment = compartment" in report


def test_joined_training_data_equals_the_columns_that_were_baked_in(nursery):
    """The joins reproduce the old pre-joined CSV exactly, so moving them into the manifest changes nothing."""
    from predictor.core.manifest import load_manifest
    from predictor.trainer.join import apply_joins
    import pandas as pd

    manifest = load_manifest(nursery["root"] / "usecases" / "nursery-gas")
    base = pd.read_csv(nursery["root"] / "nursery_base.csv")
    joined = apply_joins(manifest, base).df
    full = pd.read_csv(nursery["root"] / "nursery.csv")
    for column in DERIVED:
        assert (joined[column].to_numpy() == full[column].to_numpy()).all(), column


def test_training_stops_with_a_readable_error_when_keys_do_not_match(nursery, tmp_path):
    compartments = (nursery["root"] / "compartments.csv").read_text().splitlines()
    (tmp_path / "compartments.csv").write_text("\n".join(compartments[:3]) + "\n")  # only two of six compartments
    manifest = json.loads(json.dumps(nursery["manifest"]))
    manifest["joins"][0]["source"]["path"] = str(tmp_path / "compartments.csv")
    with pytest.raises(TrainingError, match=r"(?s)join 'compartment' failed its checks.*not in the lookup.*'K03'"):
        train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))


def test_check_joins_command_reports_ok_and_failure(nursery, tmp_path, capsys, monkeypatch):
    from predictor.cli import main

    monkeypatch.setenv("PREDICTOR_USECASES_DIR", str(nursery["root"] / "usecases"))
    assert main(["check-joins"]) == 0
    assert "join weather: OK" in capsys.readouterr().out
    bad = json.loads(json.dumps(nursery["manifest"]))
    bad["joins"][1]["window"]["end"]["offset"] = 100000  # windows far beyond the weather data
    path = write(tmp_path, bad)
    assert main(["check-joins", str(path / "manifest.json")]) == 1
    assert "FAILED" in capsys.readouterr().out


# --------------------------------------------------------------------------- serving

def test_the_service_fills_the_joined_inputs_and_predicts_like_with_full_features(nursery):
    client, df = nursery["client"], nursery["df"]
    for _, row in df.iloc[::97].iterrows():
        assert predict(client, join_inputs_only(client, row))["prediction"] == pytest.approx(
            predict(client, full_features(client, row))["prediction"], abs=1e-6)


def test_scenarios_work_from_join_inputs_and_show_the_derived_values(nursery):
    client, df = nursery["client"], nursery["df"]
    row = df[df["start_week"].between(40, 48)].iloc[-1]
    body = client.post("/models/nursery-gas/scenario", json={
        "base": {"features": join_inputs_only(client, row)},
        "scenarios": [{"name": "warmer", "changes": {"temp_setpoint_c": {"delta": 1}}}]}).json()
    shown = body["base"]["features"]
    assert shown["outside_temp_avg_c"] == row["outside_temp_avg_c"] and shown["u_value"] == row["u_value"]
    assert "compartment" not in shown and "start_date" not in shown
    assert body["scenarios"][0]["features"]["outside_temp_avg_c"] == row["outside_temp_avg_c"]  # inherited, not re-joined
    assert body["scenarios"][0]["features"]["temp_setpoint_c"] == row["temp_setpoint_c"] + 1


def test_a_value_the_caller_sends_wins_over_the_lookup(nursery):
    client, df = nursery["client"], nursery["df"]
    row = df.iloc[100]
    colder = {**join_inputs_only(client, row), "outside_temp_avg_c": float(row["outside_temp_avg_c"]) - 5}
    body = client.post("/models/nursery-gas/scenario", json={
        "base": {"features": colder}, "scenarios": [{"name": "x", "changes": {"co2_ppm": {"delta": 10}}}]}).json()
    assert body["base"]["features"]["outside_temp_avg_c"] == pytest.approx(float(row["outside_temp_avg_c"]) - 5)
    assert body["base"]["features"]["radiation_avg_j_cm2"] == row["radiation_avg_j_cm2"]  # the rest is still looked up


def test_the_sweep_goes_through_the_same_fill(nursery):
    client, df = nursery["client"], nursery["df"]
    body = client.post("/models/nursery-gas/scenario/sweep", json={
        "base": {"features": join_inputs_only(client, df.iloc[50])}, "lever": "temp_setpoint_c"})
    assert body.status_code == 200 and len(body.json()["points"]) > 3


def test_a_start_date_beyond_the_weather_data_gets_climatology_and_a_warning(nursery):
    client, df = nursery["client"], nursery["df"]
    features = {**join_inputs_only(client, df.iloc[10]), "start_date": "2026-11-02"}
    item = predict(client, features)
    assert any("not observed" in w and "other year(s)" in w for w in item["warnings"])
    body = client.post("/models/nursery-gas/scenario", json={
        "base": {"features": features}, "scenarios": [{"name": "x", "changes": {"co2_ppm": {"delta": 10}}}]}).json()
    assert any("other year(s)" in w for w in body["base"]["warnings"])


def test_bad_join_inputs_are_rejected_with_the_field_and_a_suggestion(nursery):
    client, df = nursery["client"], nursery["df"]
    good = join_inputs_only(client, df.iloc[10])
    wrong_key = client.post("/models/nursery-gas/predict", json={"records": [{"features": {**good, "compartment": "k01"}}]})
    assert wrong_key.status_code == 422
    error = wrong_key.json()["errors"][0]
    assert error["field"] == "compartment" and "differs only in case" in error["message"]
    no_date = {k: v for k, v in good.items() if k != "start_date"}
    response = client.post("/models/nursery-gas/predict", json={"records": [{"features": no_date}]})
    assert response.status_code == 422 and response.json()["errors"][0]["field"] == "start_date"
    scenario = client.post("/models/nursery-gas/scenario", json={
        "base": {"features": {**good, "start_date": "soon"}}, "scenarios": [{"name": "x", "changes": {"co2_ppm": 800}}]})
    assert scenario.status_code == 422 and scenario.json()["errors"][0]["field"] == "start_date"
    assert scenario.json()["errors"][0]["scenario"] is None


def test_unknown_fields_are_still_rejected(nursery):
    client, df = nursery["client"], nursery["df"]
    features = {**join_inputs_only(client, df.iloc[10]), "colour": "red"}
    response = client.post("/models/nursery-gas/predict", json={"records": [{"features": features}]})
    assert response.status_code == 422


def test_batch_prediction_also_fills_the_joins(nursery):
    client, df = nursery["client"], nursery["df"]
    lines = "\n".join(json.dumps({"record_id": str(i), "features": join_inputs_only(client, df.iloc[i * 40])}) for i in range(4))
    response = client.post("/models/nursery-gas/predict/batch", content=lines, headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 200
    rows = [json.loads(line) for line in response.text.splitlines()]
    assert len(rows) == 4 and all("prediction" in r for r in rows)


def test_describe_tells_clients_what_can_be_sent_instead(nursery):
    body = nursery["client"].get("/models/nursery-gas").json()
    inputs = {i["name"]: i for i in body["inputs"]}
    assert inputs["outside_temp_avg_c"]["derived_by"] == "weather" and inputs["outside_temp_avg_c"]["required"] is False
    assert "derived_by" not in inputs["temp_setpoint_c"]
    joins = {j["name"]: j for j in body["joins"]}
    assert joins["weather"]["accepts"] == ["start_date"] and joins["weather"]["fallback"] == "climatology"
    assert joins["compartment"]["accepts"] == ["compartment"]
    assert set(joins["compartment"]["fills"]) == {"area_m2", "lamp_type", "u_value"}


def test_a_scenario_lever_on_a_joined_column_stops_the_service_at_startup(nursery, tmp_path):
    manifest = json.loads(json.dumps(nursery["manifest"]))
    manifest.pop("joins")  # the model still has them: they come from the version, not from the manifest
    manifest["scenario"] = {"levers": {"u_value": {"min": 1, "max": 10, "step": 0.5}}}
    write(tmp_path, manifest)
    with pytest.raises(ManifestError, match="u_value' is calculated by join 'compartment'"):
        bootstrap.build_service(tmp_path / "usecases", nursery["root"] / "models")


def test_a_model_without_joins_is_not_affected(project):
    """The shared fixture project has no joins: nothing is added to its describe output or requests."""
    client = TestClient(bootstrap.build_app(project / "usecases", project / "models"))
    body = client.get("/models/pump-failure").json()
    assert body["joins"] == [] and all("derived_by" not in i for i in body["inputs"])


# --------------------------------------------------------------------------- review findings

def test_rows_without_a_match_are_dropped_and_reported_when_enough_rows_match(nursery, tmp_path):
    import pandas as pd

    lookup = pd.read_csv(nursery["root"] / "compartments.csv")
    gone = lookup["compartment"].iloc[1]
    lookup[lookup["compartment"] != gone].to_csv(tmp_path / "compartments.csv", index=False)
    base = pd.read_csv(nursery["root"] / "nursery_base.csv")
    unmatched = int((base["compartment"] == gone).sum())
    manifest = json.loads(json.dumps(nursery["manifest"]))
    manifest["joins"][0]["source"]["path"] = str(tmp_path / "compartments.csv")
    manifest["joins"][0]["min_match"] = 0.5
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    entry = result["joins"][0]
    assert entry["dropped_rows"] == unmatched > 0 and entry["matched"] == pytest.approx(1 - unmatched / len(base))
    assert result["dataset"]["rows"] == len(base) and result["dataset"]["rows_after_joins"] == len(base) - unmatched
    assert any(f"dropped {unmatched} of {len(base)} rows" in note for note in result["notes"])
    assert "found no match and will be dropped" in " ".join(entry["warnings"])


def test_a_base_column_with_the_name_of_a_new_column_stops_training(nursery, tmp_path):
    import pandas as pd

    base = pd.read_csv(nursery["root"] / "nursery_base.csv")
    base["u_value"] = 1.0  # a stale copy of a column the join creates
    base.to_csv(tmp_path / "nursery_base.csv", index=False)
    manifest = json.loads(json.dumps(nursery["manifest"]))
    manifest["data_source"]["path"] = str(tmp_path / "nursery_base.csv")
    with pytest.raises(TrainingError, match="already exist in the base table"):
        train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))


def test_chained_joins_train_in_order(tmp_path):
    import pandas as pd

    from predictor.core.manifest import Manifest
    from predictor.trainer.join import apply_joins

    pd.DataFrame({"plot": ["p1", "p2", "p1"], "x": [1, 2, 3]}).to_csv(tmp_path / "base.csv", index=False)
    pd.DataFrame({"plot": ["p1", "p2"], "zone": ["z1", "z2"]}).to_csv(tmp_path / "plots.csv", index=False)
    pd.DataFrame({"zone": ["z1", "z2"], "ph": [5.5, 7.0]}).to_csv(tmp_path / "zones.csv", index=False)
    manifest = Manifest.model_validate({
        "manifest_version": "1.0", "use_case_id": "chain", "data_source": {"adapter": "csv-file", "path": str(tmp_path / "base.csv")},
        "joins": [
            {"name": "plots", "source": {"adapter": "csv-file", "path": str(tmp_path / "plots.csv")}, "keys": {"plot": "plot"},
             "aggregate": {"zone": {"column": "zone", "op": "first"}}},
            {"name": "zones", "source": {"adapter": "csv-file", "path": str(tmp_path / "zones.csv")}, "keys": {"zone": "zone"},
             "aggregate": {"ph": {"column": "ph", "op": "first"}}}],
        "training": {"target": "x"}})
    applied = apply_joins(manifest, pd.read_csv(tmp_path / "base.csv"))
    assert applied.df["ph"].tolist() == [5.5, 7.0, 5.5] and not applied.failed


def test_the_registry_writes_bytes_and_subdirectories_and_refuses_paths_outside_the_version(tmp_path):
    from predictor.adapters.fs_registry import FilesystemRegistry

    registry = FilesystemRegistry(tmp_path, {"fake": lambda path, descriptor: object()})
    meta = dict(descriptor={"runner": "fake", "task_type": "regression"}, feature_schema={"properties": {}},
                artifact_name="model.bin", artifact=b"m")
    version = registry.publish("uc", **meta, extra_files={"lookups/a.npz": b"\x00\x01\x02", "notes.txt": "hello"})
    assert (tmp_path / "uc" / version / "lookups" / "a.npz").read_bytes() == b"\x00\x01\x02"
    for bad in ("../escape.txt", "/tmp/escape.txt", "lookups/../../escape.txt"):
        with pytest.raises(ValueError, match="must stay inside"):
            registry.publish("uc", **meta, extra_files={bad: "x"})
    assert registry.versions("uc") == [version]  # nothing half-written is left behind
    assert [p.name for p in (tmp_path / "uc").iterdir() if p.name.startswith(".")] == []
    assert not (tmp_path / "escape.txt").exists()


def test_a_damaged_join_snapshot_stops_the_service_with_a_clear_error(nursery, tmp_path):
    import shutil

    shutil.copytree(nursery["root"] / "models", tmp_path / "models")
    version = tmp_path / "models" / "nursery-gas" / "1"
    (version / "lookups" / "weather.npz").write_bytes(b"not a numpy file")
    with pytest.raises(LookupError, match="cannot load the join snapshot"):
        bootstrap.build_service(nursery["root"] / "usecases", tmp_path / "models")
    (version / "lookups" / "weather.npz").unlink()
    with pytest.raises(LookupError, match="cannot load the join snapshot"):
        bootstrap.build_service(nursery["root"] / "usecases", tmp_path / "models")
    (version / "joins.json").write_text("{broken")
    with pytest.raises(LookupError, match="cannot load the join snapshot"):
        bootstrap.build_service(nursery["root"] / "usecases", tmp_path / "models")


def test_the_policy_sees_the_raw_request_and_the_filled_in_values(nursery, tmp_path):
    areas = [int(line.split(",")[1]) for line in (nursery["root"] / "compartments.csv").read_text().splitlines()[1:]]
    strict = json.loads(json.dumps(nursery["manifest"]))
    strict["policy"] = {**strict["policy"], "forbidden_fields": ["secret"], "min_group_size": {"field": "area_m2", "min": max(areas) + 1}}
    write(tmp_path, strict)
    client = TestClient(bootstrap.build_app(tmp_path / "usecases", nursery["root"] / "models"))
    good = join_inputs_only(nursery["client"], nursery["df"].iloc[10])
    forbidden = client.post("/models/nursery-gas/predict", json={"records": [{"features": {**good, "secret": 1}}]})
    assert forbidden.status_code == 422 and forbidden.json()["errors"][0]["field"] == "secret"
    assert "forbidden by the data-protection policy" in forbidden.json()["errors"][0]["message"]
    # area_m2 is not in the request: the group size rule must still apply to the value that was looked up
    small = client.post("/models/nursery-gas/predict", json={"records": [{"features": good}]})
    assert small.status_code == 422 and small.json()["errors"][0]["field"] == "area_m2"
    assert "groups smaller than" in small.json()["errors"][0]["message"]


def test_odd_join_inputs_over_http_are_validation_errors_not_server_errors(nursery):
    client = nursery["client"]
    good = join_inputs_only(client, nursery["df"].iloc[10])
    for value in (["2024-01-01", "2024-01-02"], {"d": 1}, 20240101, "9999-12-31", None, 3.5):
        response = client.post("/models/nursery-gas/predict", json={"records": [{"features": {**good, "start_date": value}}]})
        assert response.status_code == 422, (value, response.status_code)
        assert response.json()["errors"][0]["field"] == "start_date"
    null_derived = client.post("/models/nursery-gas/predict", json={"records": [{"features": {**good, "u_value": None}}]})
    assert null_derived.status_code == 200  # null counts as not sent: it is looked up


def test_check_joins_cli_suggests_joins_and_survives_missing_files(nursery, tmp_path, capsys, monkeypatch):
    from predictor.cli import main

    base, lookup = nursery["root"] / "nursery_base.csv", nursery["root"] / "compartments.csv"
    assert main(["check-joins", "--suggest", str(base), str(lookup)]) == 0
    out = capsys.readouterr().out
    assert "base.compartment = lookup.compartment" in out and "one row per key (a key join)" in out
    assert main(["check-joins", "--suggest", str(tmp_path / "nope.csv"), str(lookup)]) == 2
    assert "cannot read the files" in capsys.readouterr().err
    assert main(["check-joins", "nursery-gas", "--suggest", str(base), str(lookup)]) == 2
    broken = json.loads(json.dumps(nursery["manifest"]))
    broken["data_source"]["path"] = str(tmp_path / "missing.csv")
    path = write(tmp_path, broken)
    assert main(["check-joins", str(path / "manifest.json")]) == 1
    assert "cannot check" in capsys.readouterr().err


def test_a_missing_key_column_is_not_reported_as_an_empty_table(nursery, tmp_path, capsys):
    from predictor.cli import main

    broken = json.loads(json.dumps(nursery["manifest"]))
    broken["joins"][0]["keys"] = {"nonexistent": "compartment"}
    path = write(tmp_path, broken)
    assert main(["check-joins", str(path / "manifest.json")]) == 1
    out = capsys.readouterr().out
    assert "no column(s) ['nonexistent']" in out and "no rows" not in out


def test_the_arable_split_tables_reproduce_the_season_columns(tmp_path):
    """The arable use case joins its season weather from a table per province; the old columns come out identical."""
    import make_arable
    import pandas as pd

    from predictor.core.manifest import load_manifest
    from predictor.trainer.join import apply_joins

    provinces = ["Drenthe", "Flevoland"]
    weather = {p: knmi.synthetic_daily("1994-01-01", "2025-12-31", seed=i) for i, p in enumerate(provinces)}
    crops = [name for name, _, _ in make_arable.CROPS.values()]
    rows = [{"year": year, "province": p, "crop": crop, "harvested_ha": 10.0, "yield_t_ha": 5.0 + (year % 7) + len(crop) / 10}
            for year in range(1994, 2026) for p in provinces for crop in crops]
    df = make_arable.build(pd.DataFrame(rows), weather)
    assert len(df) > 100
    tables = make_arable.split_tables(df, weather)
    tables["base"].to_csv(tmp_path / "arable_base.csv", index=False)
    tables["weather"].to_csv(tmp_path / "weather_provinces.csv", index=False)
    manifest = json.loads((ROOT / "usecases" / "arable-yield" / "manifest.json").read_text())
    manifest["data_source"]["path"] = str(tmp_path / "arable_base.csv")
    manifest["joins"][0]["source"]["path"] = str(tmp_path / "weather_provinces.csv")
    applied = apply_joins(load_manifest(write(tmp_path, manifest)), pd.read_csv(tmp_path / "arable_base.csv"))
    assert len(applied.df) == len(df) and applied.consumed == ["season_start", "season_end"]
    for column in make_arable.JOINED_COLUMNS:
        assert (applied.df[column].to_numpy() == df[column].to_numpy()).all(), column
