"""Several outputs in one use case: manifest rules, one model per output, `algorithms`, the registry layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictor import bootstrap
from predictor.adapters.fs_registry import FilesystemRegistry, PublishedOutput
from predictor.core.manifest import ALGORITHMS, Manifest, ManifestError, load_all_manifests, load_manifest
from predictor.trainer import models
from predictor.trainer.errors import TrainingError
from predictor.trainer.run import train_use_case


def table(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({
        "row_id": np.arange(n), "x1": x1.round(3), "x2": x2.round(3), "x3": x3.round(3),
        "kind": rng.choice(["a", "b", "c"], n),
        "y_linear": (2 * x1 - x2 + rng.normal(0, 0.3, n)).round(3),
        "y_positive": np.abs(x1 * x3 + rng.normal(0, 0.2, n)).round(3),
        "y_flag": (x1 + x2 + rng.normal(0, 0.5, n) > 0).astype(int),
    })


OUTPUTS = {
    "y_linear": {"unit": "units", "algorithms": ["linear_regression"]},          # forced
    "y_positive": {"unit": "m3", "target_min": 0, "algorithms": ["lightgbm", "blend"]},  # restricted contest
    "y_flag": {"task": "classification"},                                         # default contest
}


def manifest_for(root: Path, **over) -> dict:
    return {"manifest_version": "1.0", "use_case_id": "multi", "data_source": {"adapter": "csv-file", "path": str(root / "multi.csv")},
            "training": {"drop_columns": ["row_id"], "outputs": OUTPUTS}, **over}


def write(root: Path, manifest: dict) -> Path:
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi")
    table().to_csv(root / "multi.csv", index=False)
    registry = bootstrap.build_registry(root / "models")
    result = train_use_case(write(root, manifest_for(root)), registry)
    return {"root": root, "result": result, "registry": registry}


# --------------------------------------------------------------------------- manifest rules

BASE = {"manifest_version": "1.0", "use_case_id": "demo", "data_source": {"adapter": "csv-file"}}


def manifest(training: dict, **over) -> Manifest:
    return Manifest.model_validate({**BASE, "training": training, **over})


def test_outputs_and_the_single_target_give_the_same_view():
    several = manifest({"outputs": OUTPUTS, "drop_columns": ["row_id"]})
    assert several.training.output_names == ["y_linear", "y_positive", "y_flag"]
    one = manifest({"target": "y", "task": "regression", "target_max": 5, "units": {"y": "kg"}})
    config = one.training.output_configs()["y"]
    assert (one.training.output_names, config.task, config.target_max, config.unit) == (["y"], "regression", 5, "kg")
    assert manifest({"outputs": {"a": {}}, "units": {"a": "kg"}}).training.output_configs()["a"].unit == "kg"


def test_an_output_is_seen_as_a_single_target_without_the_other_outputs():
    view = manifest({"outputs": OUTPUTS, "drop_columns": ["row_id"]}).training.for_output("y_positive")
    assert (view.target, view.task, view.target_min, view.outputs) == ("y_positive", "auto", 0, {})
    assert view.drop_columns == ["row_id", "y_linear", "y_flag"]  # outputs are never features of each other
    assert view.units["y_positive"] == "m3" and view.output_names == ["y_positive"]
    assert manifest({"outputs": OUTPUTS}).training.for_output("y_flag").task == "classification"


@pytest.mark.parametrize(
    "training, message",
    [
        ({"target": "y", "outputs": {"a": {}}}, "either 'target' .* or 'outputs'"),
        ({"outputs": {"a": {}}, "task": "regression"}, "set 'task' per output"),
        ({"outputs": {"a": {}}, "target_min": 0}, "per output"),
        ({"outputs": {"a": {}}, "horizon_steps": 2, "time_column": "t"}, "horizon_steps"),
        ({"outputs": {"a": {}}, "drop_columns": ["a"]}, "is in drop_columns"),
        ({"outputs": {"a": {}}, "time_column": "a"}, "is in time_column"),
        ({"outputs": {"a": {}}, "rename_columns": {"a": "b"}}, "is in rename_columns"),
        ({"outputs": {"a": {"target_min": 5, "target_max": 1}}}, "must be smaller"),
        ({"outputs": {"a": {"task": "classification", "target_min": 0}}}, "only apply to regression"),
        ({"outputs": {"a": {"algorithms": []}}}, "at least 1 item"),
        ({"outputs": {"a": {"task": "regression", "algorithms": ["logistic_regression"]}}}, r"unknown model\(s\) \['logistic_regression'\]"),
        ({"outputs": {"a": {"algorithms": ["magic"]}}}, r"unknown model\(s\) \['magic'\] \(valid: "),
    ],
)
def test_output_rules(training, message):
    with pytest.raises(ValueError, match=message):
        manifest(training)


def test_default_algorithms_are_checked_against_the_tasks_of_the_outputs():
    assert manifest({"outputs": OUTPUTS}, model={"algorithms": ["lightgbm"]}).model.algorithms == ["lightgbm"]
    with pytest.raises(ValueError, match="model.algorithms: unknown model"):
        manifest({"outputs": OUTPUTS}, model={"algorithms": ["isolation_forest"]})
    with pytest.raises(ValueError, match="model.algorithms: unknown model"):  # no target = anomaly detection
        manifest({}, model={"algorithms": ["lightgbm"]})
    assert manifest({}, model={"algorithms": ["isolation_forest"]})


def test_aliases_must_be_slugs_unique_and_name_an_output():
    ok = manifest({"outputs": OUTPUTS}, aliases={"demo-linear": "y_linear", "demo-flag": "y_flag"})
    assert ok.aliases["demo-flag"] == "y_flag"
    for aliases, message in (({"Bad Id": "y_linear"}, "lowercase slug"), ({"demo": "y_linear"}, "id of this use case itself"),
                             ({"x-y": "nope"}, "unknown output 'nope'"),
                             ({"demo-a": "y_linear", "demo-b": "y_linear"}, "only one alias")):
        with pytest.raises(ValueError, match=message):
            manifest({"outputs": OUTPUTS}, aliases=aliases)


def test_outputs_may_not_be_scenario_levers_or_join_columns():
    with pytest.raises(ValueError, match="is the target"):
        manifest({"outputs": {"a": {}}}, joins=[{"name": "j", "source": {"adapter": "csv-file", "path": "x"},
                                                 "keys": {"k": "k"}, "aggregate": {"a": {"column": "c", "op": "first"}}}])
    with pytest.raises(ValueError, match="is the target"):
        manifest({"outputs": {"a": {}, "b": {}}}, scenario={"levers": {"a": {"min": 0, "max": 1, "step": 0.1}}})


def test_scenario_rules_check_every_output():
    scenario = {"levers": {"x": {"min": 0, "max": 1, "step": 0.1, "effects": {"b": "increasing"}}}, "outputs": {"a": {}, "b": {}}}
    assert manifest({"outputs": {"a": {}, "b": {}}}, scenario=scenario)
    with pytest.raises(ValueError, match="unknown output 'c'"):
        manifest({"outputs": {"a": {}, "b": {}}}, scenario={**scenario, "outputs": {"c": {}}})


def test_ids_and_aliases_of_different_use_cases_may_not_clash(tmp_path):
    two_aliases, alias_and_id = tmp_path / "a", tmp_path / "b"
    for use_case_id in ("one", "two"):
        write(two_aliases, {**BASE, "use_case_id": use_case_id, "training": {"outputs": {"a": {}}}, "aliases": {"shared": "a"}})
    with pytest.raises(ManifestError, match="id 'shared' is used by both 'one' and 'two'"):
        load_all_manifests(two_aliases / "usecases")
    write(alias_and_id, {**BASE, "use_case_id": "three", "training": {"target": "y"}})
    write(alias_and_id, {**BASE, "use_case_id": "four", "training": {"outputs": {"a": {}}}, "aliases": {"three": "a"}})
    with pytest.raises(ManifestError, match="id 'three' is used by both 'four' and 'three'"):
        load_all_manifests(alias_and_id / "usecases")


def test_the_algorithm_names_in_the_manifest_match_the_models_the_trainer_has():
    for task, names in ALGORITHMS.items():
        assert tuple(models.candidates(task)) == names


def test_candidates_can_be_limited_by_name():
    assert list(models.candidates("regression", allowed=["lightgbm"])) == ["lightgbm"]
    assert list(models.candidates("regression", allowed=["blend", "linear_regression"])) == ["linear_regression", "blend"]
    with pytest.raises(TrainingError, match=r"\['logistic_regression'\] are not available for a 'regression' output"):
        models.candidates("regression", allowed=["logistic_regression"])


# --------------------------------------------------------------------------- training

def test_one_run_publishes_one_version_with_a_model_per_output(trained):
    version = trained["root"] / "models" / "multi" / "1"
    assert json.loads((version / "use_case.json").read_text())["outputs"] == ["y_linear", "y_positive", "y_flag"]
    for output in OUTPUTS:
        assert (version / output / "model.onnx").is_file() and (version / output / "descriptor.json").is_file()
    assert (version / "feature_schema.json").is_file() and (version / "report.md").is_file()
    assert not (version / "descriptor.json").exists() and trained["registry"].versions("multi") == ["1"]


def test_outputs_are_never_features_of_each_other_and_share_their_inputs(trained):
    descriptors = trained["result"]["outputs"]
    assert descriptors["y_linear"]["features"] == descriptors["y_positive"]["features"] == descriptors["y_flag"]["features"]
    assert set(descriptors["y_linear"]["features"]) == {"x1", "x2", "x3", "kind"}
    schema = json.loads((trained["root"] / "models" / "multi" / "1" / "feature_schema.json").read_text())
    assert set(schema["x-feature-order"]) == {"x1", "x2", "x3", "kind"}


def test_each_output_has_its_own_task_unit_bounds_and_model(trained):
    d = trained["result"]["outputs"]
    assert [d[o]["output"] for o in OUTPUTS] == list(OUTPUTS)
    assert (d["y_linear"]["task_type"], d["y_positive"]["task_type"], d["y_flag"]["task_type"]) == ("regression", "regression", "classification")
    assert (d["y_linear"]["target_unit"], d["y_positive"]["target_unit"]) == ("units", "m3")
    assert d["y_positive"]["target_bounds"] == [0, None] and d["y_linear"]["target_bounds"] is None
    assert d["y_flag"]["classes"] == ["0", "1"]


def test_algorithms_force_or_restrict_the_contest_per_output(trained):
    d = trained["result"]["outputs"]
    assert list(d["y_linear"]["candidates"]) == ["linear_regression"] and d["y_linear"]["algorithm"] == "linear_regression"
    assert set(d["y_positive"]["candidates"]) == {"lightgbm", "blend"} and d["y_positive"]["algorithm"] in {"lightgbm", "blend"}
    assert set(d["y_flag"]["candidates"]) == {"logistic_regression", "lightgbm"}  # nothing set: today's contest


def test_the_result_keeps_the_flat_keys_of_the_first_output(trained):
    result = trained["result"]
    first = result["outputs"]["y_linear"]
    assert result["output"] == "y_linear" and result["algorithm"] == first["algorithm"] and result["metrics"] == first["metrics"]
    assert result["model_version"] == "1" and set(result["outputs"]) == set(OUTPUTS)


def test_the_report_has_one_section_per_output_and_the_joins_only_once(trained):
    report = Path(trained["result"]["report"]).read_text()
    assert report.count("# Training report: multi v1") == 1
    for output in OUTPUTS:
        assert f"## Output: {output}" in report
    assert "3 outputs, each with its own model" in report and "### Candidates (test split)" in report


def test_the_default_models_come_from_the_model_section(tmp_path):
    table().to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path, model={"algorithms": ["lightgbm"]})
    manifest["training"]["outputs"] = {"y_linear": {}, "y_positive": {"algorithms": ["linear_regression"]}}
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    assert list(result["outputs"]["y_linear"]["candidates"]) == ["lightgbm"]  # the default
    assert list(result["outputs"]["y_positive"]["candidates"]) == ["linear_regression"]  # the output wins


def test_a_model_that_does_not_exist_for_the_detected_task_stops_training_before_anything_is_trained(tmp_path):
    table().to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path)
    manifest["training"]["outputs"] = {"y_positive": {}, "y_linear": {"algorithms": ["logistic_regression"]}}  # 'auto' = regression
    registry = bootstrap.build_registry(tmp_path / "models")
    with pytest.raises(TrainingError, match=r"output 'y_linear' is a 'regression' output, so the model\(s\) \['logistic_regression'\] are not available"):
        train_use_case(write(tmp_path, manifest), registry)
    assert registry.versions("multi") == []  # y_positive (first) was not trained either: the check comes first


def test_the_default_models_are_cut_down_to_the_task_of_each_output(tmp_path):
    table().to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path, model={"algorithms": ["linear_regression", "logistic_regression"]})
    manifest["training"]["outputs"] = {"y_linear": {}, "y_flag": {"task": "classification"}}
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    assert list(result["outputs"]["y_linear"]["candidates"]) == ["linear_regression"]
    assert list(result["outputs"]["y_flag"]["candidates"]) == ["logistic_regression"]


def test_a_default_that_leaves_an_output_without_a_model_is_rejected_when_the_manifest_loads():
    outputs = {"a": {"task": "regression"}, "b": {"task": "classification"}}
    assert manifest({"outputs": outputs}, model={"algorithms": ["lightgbm"]})  # exists for both tasks
    with pytest.raises(ValueError, match=r"has no model for output 'b' \(a 'classification' output"):
        manifest({"outputs": outputs}, model={"algorithms": ["linear_regression"]})
    assert manifest({"outputs": {**outputs, "b": {**outputs["b"], "algorithms": ["logistic_regression"]}}},
                    model={"algorithms": ["linear_regression"]})  # its own list wins


@pytest.mark.parametrize(
    "training, message",
    [
        ({"outputs": {"a/b": {}}}, "used as a folder name"),
        ({"outputs": {"lookups": {}}}, "used as a folder name"),
        ({"outputs": {"use_case.json": {}}}, "used as a folder name"),
        ({"outputs": {"a": {"baseline": {"column": "b", "label": "x"}}, "b": {}}}, "'b' is an output, not an input"),
        ({"outputs": {"a": {}}, "test_group": {"column": "a", "values": ["x"]}}, "'a' is an output, not an input"),
    ],
)
def test_more_output_rules(training, message):
    with pytest.raises(ValueError, match=message):
        manifest(training)


def test_outputs_that_drop_different_rows_still_share_one_encoding_of_the_inputs(tmp_path):
    """y_b has no value where kind == 'a'. Profiled alone, y_b would number the kinds b=0, c=1 while the published
    schema says a=0, b=1, c=2: serving would read every 'b' as 'c'."""
    df = table()
    df["y_a"] = (df["x1"] * 2).round(3)
    df["y_b"] = np.where(df["kind"] == "b", 100.0, np.where(df["kind"] == "c", 200.0, np.nan))
    df.to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path)
    manifest["training"] = {"drop_columns": ["row_id", "y_linear", "y_positive", "y_flag"],
                            "outputs": {"y_a": {"task": "regression", "algorithms": ["linear_regression"]},
                                        "y_b": {"task": "regression", "algorithms": ["lightgbm"]}}}
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    assert result["outputs"]["y_a"]["features"] == result["outputs"]["y_b"]["features"]
    schema = json.loads((tmp_path / "models" / "multi" / "1" / "feature_schema.json").read_text())
    assert schema["properties"]["kind"]["enum"] == ["a", "b", "c"]  # the encoding both models were trained with
    client = TestClient(bootstrap.build_app(tmp_path / "usecases", tmp_path / "models"))
    for kind, expected in (("b", 100.0), ("c", 200.0)):
        item = client.post("/models/multi/predict", json={"records": [{"features": {**FEATURES, "kind": kind}}]}).json()["predictions"][0]
        assert item["outputs"]["y_b"]["prediction"] == pytest.approx(expected, abs=1.0), kind


def test_single_output_profiling_is_untouched_by_the_shared_profile(tmp_path):
    """One output: no shared profile, so the inputs are profiled after dropping the rows without a target, as before."""
    df = table()
    df.loc[df["kind"] == "a", "y_linear"] = np.nan
    df.to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path)
    manifest["training"] = {"drop_columns": ["row_id", "y_positive", "y_flag"], "target": "y_linear"}
    train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    schema = json.loads((tmp_path / "models" / "multi" / "1" / "feature_schema.json").read_text())
    assert schema["properties"]["kind"]["enum"] == ["b", "c"]  # unchanged behaviour for a single output


def test_outputs_that_miss_different_rows_still_get_the_same_inputs(tmp_path):
    """x4 only varies where y_positive has no value. Profiled per output it would be constant (dropped) for y_positive
    but kept for y_linear; profiled once on all rows both outputs use it."""
    df = table()
    df["x4"] = np.random.default_rng(1).normal(0, 1, len(df)).round(3)
    df.loc[df.index % 2 == 0, "y_positive"] = np.nan
    df.loc[df["y_positive"].notna(), "x4"] = 7.0
    df.to_csv(tmp_path / "multi.csv", index=False)
    manifest = manifest_for(tmp_path)
    manifest["training"]["outputs"] = {"y_linear": {}, "y_positive": {}}
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    features = {name: d["features"] for name, d in result["outputs"].items()}
    assert features["y_linear"] == features["y_positive"] and "x4" in features["y_positive"]


def test_a_use_case_with_a_single_target_is_published_in_the_same_layout(project):
    registry = bootstrap.build_registry(project / "models")
    version = project / "models" / "pump-failure" / registry.versions("pump-failure")[-1]
    assert json.loads((version / "use_case.json").read_text())["outputs"] == ["failure"]
    assert (version / "failure" / "model.onnx").is_file()
    loaded = registry.resolve("pump-failure")
    assert list(loaded.outputs) == ["failure"] and loaded.descriptor is loaded.primary.descriptor
    assert loaded.only("failure").outputs.keys() == loaded.outputs.keys()


# --------------------------------------------------------------------------- registry

def test_a_version_from_before_outputs_existed_is_read_as_one_output(tmp_path):
    registry = FilesystemRegistry(tmp_path, {"fake": lambda path, descriptor: ("runner", path.name)})
    old = tmp_path / "uc" / "1"
    old.mkdir(parents=True)
    (old / "feature_schema.json").write_text(json.dumps({"properties": {}}))
    (old / "descriptor.json").write_text(json.dumps({"runner": "fake", "artifact": "model.onnx", "model_version": "1",
                                                      "target_name": "gas", "task_type": "regression"}))
    (old / "model.onnx").write_bytes(b"m")
    assert registry.versions("uc") == ["1"]
    loaded = registry.resolve("uc")
    assert list(loaded.outputs) == ["gas"] and loaded.version == "1" and loaded.runner == ("runner", "model.onnx")


def test_publish_refuses_an_output_name_that_would_collide_with_a_shared_file(tmp_path):
    registry = FilesystemRegistry(tmp_path, {"fake": lambda path, descriptor: object()})
    meta = {"descriptor": {"runner": "fake"}, "artifact": b"m"}
    for name in ("lookups", "report.md", "feature_schema.json", "use_case.json", ".hidden", "a/b", ""):
        with pytest.raises(ValueError, match="cannot be used as a folder name"):
            registry.publish("uc", outputs={name: PublishedOutput(**meta)}, feature_schema={}, extra_files={"lookups/a.npz": b"x"})
    with pytest.raises(ValueError, match="at least one output"):
        registry.publish("uc", outputs={}, feature_schema={})
    assert registry.versions("uc") == []
    assert not [p for p in (tmp_path / "uc").iterdir() if p.name.startswith(".")]  # nothing half-written


def test_every_manifest_in_the_repository_is_still_valid_and_has_its_outputs():
    manifests = load_all_manifests(Path(__file__).resolve().parents[1] / "usecases")
    assert all(m.training.output_names or m.training.task in ("auto", "anomaly") for m in manifests)
    assert load_manifest(Path(__file__).resolve().parents[1] / "usecases" / "greenhouse-climate").training.output_names


# --------------------------------------------------------------------------- serving: shapes, aliases, scenarios

from fastapi.testclient import TestClient  # noqa: E402

SCENARIO = {
    "levers": {"x1": {"min": -2, "max": 2, "step": 0.5, "effects": {"y_linear": "increasing"}},
               "x2": {"min": -2, "max": 2, "step": 0.5}},
    "outputs": {"y_linear": {"label": "Linear", "value_eur": 2.0}, "y_positive": {"label": "Positive", "value_eur": -3.0}},
}
ALIASES = {"multi-linear": "y_linear", "multi-flag": "y_flag"}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("served")
    table().to_csv(root / "multi.csv", index=False)
    manifest = manifest_for(root, scenario=SCENARIO, aliases=ALIASES)
    train_use_case(write(root, manifest), bootstrap.build_registry(root / "models"))
    client = TestClient(bootstrap.build_app(root / "usecases", root / "models"))
    return {"root": root, "client": client, "manifest": manifest}


FEATURES = {"x1": 0.5, "x2": -0.2, "x3": 1.0, "kind": "b"}


def predict_item(client, use_case="multi", features=None):
    response = client.post(f"/models/{use_case}/predict", json={"records": [{"features": features or FEATURES}]})
    assert response.status_code == 200, response.text
    return response.json()["predictions"][0], response.json()


def test_the_use_case_list_shows_outputs_and_the_aliases(served):
    listed = {u["use_case_id"]: u for u in served["client"].get("/use-cases").json()}
    assert listed["multi"]["outputs"] == ["y_linear", "y_positive", "y_flag"] and listed["multi"]["alias_of"] is None
    assert listed["multi-linear"]["alias_of"] == {"use_case_id": "multi", "output": "y_linear"}
    assert listed["multi-linear"]["outputs"] == ["y_linear"] and listed["multi-flag"]["task_type"] == "classification"


def test_predict_with_several_outputs_has_an_outputs_map_and_no_flat_fields(served):
    item, body = predict_item(served["client"])
    assert set(item) == {"record_id", "outputs", "warnings"} and set(item["outputs"]) == set(OUTPUTS)
    assert isinstance(item["outputs"]["y_linear"]["prediction"], float) and item["outputs"]["y_positive"]["prediction"] >= 0
    assert set(item["outputs"]["y_flag"]["probabilities"]) == {"0", "1"}
    assert body["outputs"]["y_positive"] == {"task_type": "regression", "algorithm": body["outputs"]["y_positive"]["algorithm"],
                                             "target_name": "y_positive", "target_unit": "m3"}
    assert body["target_name"] == "y_linear"  # the top-level keys describe the first output


def test_an_alias_answers_with_the_flat_single_output_shape_and_the_same_numbers(served):
    client = served["client"]
    multi, _ = predict_item(client)
    linear, body = predict_item(client, "multi-linear")
    assert set(linear) == {"record_id", "prediction", "warnings"} and body["use_case_id"] == "multi-linear"
    assert linear["prediction"] == multi["outputs"]["y_linear"]["prediction"]
    flag, _ = predict_item(client, "multi-flag")
    assert flag["prediction"] == multi["outputs"]["y_flag"]["prediction"] and "probabilities" in flag and "outputs" not in flag
    assert body["target_unit"] == "units" and body["outputs"] == {"y_linear": body["outputs"]["y_linear"]}


def test_batch_prediction_returns_all_outputs_per_line(served):
    line = json.dumps({"record_id": "r1", "features": FEATURES})
    response = served["client"].post("/models/multi/predict/batch", content=line + "\n" + line,
                                     headers={"content-type": "application/x-ndjson"})
    rows = [json.loads(x) for x in response.text.splitlines()]
    assert response.status_code == 200 and len(rows) == 2 and set(rows[0]["outputs"]) == set(OUTPUTS)


def test_describe_and_evaluation_have_a_block_per_output_and_keep_the_first_one_flat(served):
    client = served["client"]
    described = client.get("/models/multi").json()
    assert described["target_name"] == "y_linear" and described["task_type"] == "regression"
    assert {n: o["task_type"] for n, o in described["outputs"].items()} == {
        "y_linear": "regression", "y_positive": "regression", "y_flag": "classification"}
    assert described["outputs"]["y_flag"]["classes"] == ["0", "1"] and described["outputs"]["y_linear"]["target_unit"] == "units"
    evaluation = client.get("/models/multi/evaluation").json()
    assert list(evaluation["outputs"]["y_linear"]["candidates"]) == ["linear_regression"]
    assert set(evaluation["outputs"]["y_positive"]["candidates"]) == {"lightgbm", "blend"}
    assert "confusion_matrix" in evaluation["outputs"]["y_flag"]["sample"] and "regression" in evaluation["outputs"]["y_positive"]["sample"]
    assert evaluation["metrics"] == evaluation["outputs"]["y_linear"]["metrics"]
    alias = client.get("/models/multi-flag/evaluation").json()
    assert alias["task_type"] == "classification" and list(alias["outputs"]) == ["y_flag"]


def test_a_scenario_gives_every_output_and_one_net_result(served):
    body = served["client"].post("/models/multi/scenario", json={
        "base": {"features": FEATURES}, "scenarios": [{"name": "more x1", "changes": {"x1": {"delta": 1}}}]}).json()
    scenario = body["scenarios"][0]
    assert set(body["base"]["outputs"]) == set(scenario["outputs"]) == set(scenario["difference"]) == set(OUTPUTS)
    d = scenario["difference"]
    expected = round(d["y_linear"]["eur"] + d["y_positive"]["eur"], 2)
    assert scenario["net_eur"] == pytest.approx(expected, abs=0.01) and d["y_flag"].get("eur") is None  # unpriced
    assert d["y_linear"]["change"] > 0  # declared: more x1 never lowers y_linear
    assert "Net worth" in scenario["summary"] and "units" in scenario["summary"] and "m3" in scenario["summary"]


def test_the_margin_of_error_and_unit_are_those_of_each_output(served):
    scenario = served["client"].get("/models/multi").json()["scenario"]
    outputs = {o["name"]: o for o in scenario["outputs"]}
    assert set(outputs) == set(OUTPUTS) and (outputs["y_linear"]["unit"], outputs["y_positive"]["unit"]) == ("units", "m3")
    assert outputs["y_linear"]["margin"] != outputs["y_positive"]["margin"] and outputs["y_flag"]["margin"] is None
    assert outputs["y_linear"]["constraints"]["effects"] == {"x1": "increasing"} and outputs["y_positive"]["constraints"] is None


def test_scenarios_and_sweeps_of_an_alias_only_show_its_output(served):
    client = served["client"]
    request = {"base": {"features": FEATURES}, "scenarios": [{"name": "more x1", "changes": {"x1": {"delta": 1}}}]}
    multi = client.post("/models/multi/scenario", json=request).json()["scenarios"][0]
    alias = client.post("/models/multi-linear/scenario", json=request).json()["scenarios"][0]
    assert set(alias["outputs"]) == set(alias["difference"]) == {"y_linear"}
    assert alias["difference"]["y_linear"]["change"] == multi["difference"]["y_linear"]["change"]
    assert alias["net_eur"] == alias["difference"]["y_linear"]["eur"]  # only the output this id is about
    sweep = client.post("/models/multi/scenario/sweep", json={"base": {"features": FEATURES}, "lever": "x1"}).json()
    assert len(sweep["points"]) > 3 and set(sweep["points"][0]["outputs"]) == set(OUTPUTS)
    assert client.post("/models/multi-linear/scenario/sweep", json={"base": {"features": FEATURES}, "lever": "x1"}).status_code == 200


def test_a_manifest_that_declares_other_outputs_than_the_trained_version_stops_the_service(served, tmp_path):
    other = json.loads(json.dumps(served["manifest"]))
    other["training"]["outputs"] = {"y_linear": {}, "y_flag": {"task": "classification"}}
    other.pop("scenario"), other.pop("aliases")
    write(tmp_path, other)
    with pytest.raises(ManifestError, match=r"trained version has outputs \['y_flag', 'y_linear', 'y_positive'\] but the manifest declares"):
        bootstrap.build_service(tmp_path / "usecases", served["root"] / "models")


def test_the_alias_of_a_use_case_without_a_trained_model_is_listed_as_not_trained(served, tmp_path):
    fresh = json.loads(json.dumps(served["manifest"]))
    write(tmp_path, fresh)
    service = bootstrap.build_service(tmp_path / "usecases", tmp_path / "empty-models")
    listed = {u["use_case_id"]: u for u in service.list_use_cases()}
    assert listed["multi"]["status"] == "not_trained" == listed["multi-linear"]["status"] and listed["multi"]["outputs"] is None


def test_train_and_list_on_the_command_line_know_outputs_and_aliases(served, tmp_path, capsys, monkeypatch):
    from predictor.cli import main

    table().to_csv(tmp_path / "multi.csv", index=False)
    write(tmp_path, manifest_for(tmp_path, aliases=ALIASES))
    monkeypatch.setenv("PREDICTOR_USECASES_DIR", str(tmp_path / "usecases"))
    monkeypatch.setenv("PREDICTOR_MODELS_DIR", str(tmp_path / "models"))
    assert main(["train", "multi-linear"]) == 0  # an alias trains its use case, once
    out = capsys.readouterr().out
    assert out.count("multi [y_linear]") == out.count("multi [y_positive]") == out.count("multi [y_flag]") == 1
    assert (tmp_path / "models" / "multi" / "1" / "y_flag" / "model.onnx").is_file()
    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "[y_linear, y_positive, y_flag]" in listed and "multi-linear" in listed and "alias of multi / y_linear" in listed


def test_every_output_has_its_own_comparable_cases_map_built_from_its_own_rows(served):
    version = served["root"] / "models" / "multi" / "1"
    assert not (version / "support.json").exists()  # nothing shared at the top
    weights = {}
    for output in OUTPUTS:
        support = json.loads((version / output / "support.json").read_text())
        certainty = json.loads((version / output / "descriptor.json").read_text())["certainty"]
        assert certainty == {"available": True, "training_rows": support["rows"], "radius": round(support["radius"], 6)}
        weights[output] = {f["name"]: round(f["weight"], 3) for f in support["features"]}
    assert weights["y_linear"] != weights["y_positive"]  # each output weights the inputs it really uses


def test_a_scenario_shows_the_weakest_certainty_and_the_numbers_per_output(served):
    body = served["client"].post("/models/multi/scenario", json={
        "base": {"features": FEATURES}, "scenarios": [{"name": "more x1", "changes": {"x1": {"delta": 1}}}]}).json()
    certainty = body["scenarios"][0]["certainty"]
    by_output = certainty["by_output"]
    assert set(by_output) == set(OUTPUTS) and all("message" not in v for v in by_output.values())
    assert certainty["comparable_cases"] == min(v["comparable_cases"] for v in by_output.values())  # never promises more
    assert body["base"]["certainty"]["by_output"].keys() == by_output.keys()
    alias = served["client"].post("/models/multi-linear/scenario", json={
        "base": {"features": FEATURES}, "scenarios": [{"name": "more x1", "changes": {"x1": {"delta": 1}}}]}).json()
    assert alias["scenarios"][0]["certainty"]["comparable_cases"] == by_output["y_linear"]["comparable_cases"]
    assert "by_output" not in alias["scenarios"][0]["certainty"]  # one output: the shape of before


def test_the_typical_error_sentence_is_said_once_when_several_outputs_are_within_it():
    from predictor.core.manifest import ScenarioConfig
    from predictor.core.scenario import compare, summarise

    cfg = ScenarioConfig.model_validate({"levers": {"x": {"min": 0, "max": 1, "step": 1}},
                                         "outputs": {"gas": {"label": "Gas use"}, "plants": {"label": "Plants"}, "weeks": {"label": "Weeks"}}})
    descriptors = {n: {"task_type": "regression", "target_unit": u, "metrics": {"mae": m}}
                   for n, u, m in (("gas", "m3", 100.0), ("plants", "plants", 4000.0), ("weeks", "weeks", 0.9))}
    before = {n: {"value": v} for n, v in (("gas", 1000.0), ("plants", 100000.0), ("weeks", 10.0))}
    now = {n: {"value": v} for n, v in (("gas", 1050.0), ("plants", 101000.0), ("weeks", 8.0))}  # weeks changes by more than its error
    text = summarise(cfg, descriptors, "warmer", compare(cfg, descriptors, {}, before, {}, now))
    assert text.count("typical error") == 1 and "weeks 20.0% lower" in text
    assert text.endswith("Within the model's typical error, so no clear effect: gas use (±100 m3), plants (±4,000 plants).")
    assert "this is within" not in text  # nothing repeated inside the sentences
    one = {"gas": now["gas"], "plants": before["plants"], "weeks": before["weeks"]}  # only one output within its error
    single = summarise(cfg, descriptors, "warmer", compare(cfg, descriptors, {}, before, {}, one))
    assert "; this is within the model's typical error of ±100 m3, so treat it as no clear effect" in single
    assert "Within the model's typical error, so no clear effect" not in single  # one output keeps the inline wording
