"""Trainer features added for ticket 1: backtests, baseline columns, refit, scale-aware parity."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from predictor import bootstrap
from predictor.core.manifest import ManifestError, load_manifest
from predictor.trainer.export import check_parity, to_onnx
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import knmi  # noqa: E402
import make_nursery  # noqa: E402


def write(tmp_path, manifest):
    directory = tmp_path / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def yearly_data(path, years=range(2000, 2026)):
    rng = np.random.default_rng(0)
    rows = []
    for year in years:
        for i in range(12):
            rain = rng.uniform(200, 600)
            normal = 10 + 0.1 * (year - 2000)
            rows.append({"season": f"{year}-01-01", "normal": round(normal, 2), "rain": round(rain, 1),
                         "yield": round(normal - 0.004 * (rain - 400) + rng.normal(0, 0.3), 2)})
    pd.DataFrame(rows).to_csv(path, index=False)


def base_manifest(data, **training):
    return {"manifest_version": "1.0", "use_case_id": "yields",
            "data_source": {"adapter": "csv-file", "path": str(data)},
            "training": {"target": "yield", "task": "regression", "time_column": "season", **training}}


def test_test_from_makes_an_exact_backtest(tmp_path):
    data = tmp_path / "yields.csv"
    yearly_data(data)
    path = write(tmp_path, base_manifest(data, test_from="2021-01-01"))
    result = train_use_case(path, bootstrap.build_registry(tmp_path / "models"))
    assert result["split"]["test_from"] == "2021-01-01"
    assert result["split"]["test"] == 5 * 12  # 2021-2025, nothing else
    assert "backtest: test from 2021-01-01" in Path(result["report"]).read_text()


def test_baseline_column_is_reported_with_its_label(tmp_path):
    data = tmp_path / "yields.csv"
    yearly_data(data)
    manifest = base_manifest(data, test_from="2021-01-01",
                             baseline={"column": "normal", "label": "same as normal"})
    result = train_use_case(write(tmp_path, manifest), bootstrap.build_registry(tmp_path / "models"))
    ref = result["reference"]
    assert ref["label"] == "same as normal"
    assert result["metrics"]["mae"] < ref["persistence_mae"]  # rain really matters in this data
    assert "normal" in result["features"]  # the guess may also be used as a feature


@pytest.mark.parametrize("training, message", [
    ({"test_from": "2021-01-01", "time_column": None}, "test_from"),
    ({"baseline": {"column": "normal", "label": "x"}, "horizon_steps": 2}, "baseline"),
])
def test_invalid_backtest_settings_fail_fast(tmp_path, training, message):
    manifest = base_manifest("data.csv", **training)
    with pytest.raises(ManifestError, match=message):
        load_manifest(write(tmp_path, manifest))


def test_refit_uses_train_and_validation(tmp_path):
    """The final model is refitted on train + validation, so the most recent data is not wasted."""
    data = tmp_path / "yields.csv"
    yearly_data(data)
    result = train_use_case(write(tmp_path, base_manifest(data, test_from="2021-01-01")),
                            bootstrap.build_registry(tmp_path / "models"))
    assert "retrained on train + validation" in Path(result["report"]).read_text()


def test_parity_tolerance_scales_with_large_outputs():
    """Gas in m3 reaches 100,000: float32 rounding there is not a broken export."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 200), "b": rng.uniform(0, 1, 200)})
    y = 90_000 * X["a"] + 5_000 * X["b"] + 0.123456
    model = Pipeline([("model", LinearRegression())]).fit(X.to_numpy(np.float32), y)
    artifact = to_onnx(model, 2, "regression")
    parity = check_parity(model, artifact, X, {"task_type": "regression"})
    assert parity["passed"], parity


def test_trained_model_learns_true_scenario_directions(tmp_path):
    """On simulated nursery data the model must recover the clear effects of the answer key."""
    daily = knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=2)
    df, sim, batches = make_nursery.generate(daily, compartments=10, seed=4)
    data = tmp_path / "nursery.csv"
    df.to_csv(data, index=False)
    manifest = {"manifest_version": "1.0", "use_case_id": "weeks",
                "data_source": {"adapter": "csv-file", "path": str(data)},
                "training": {"target": "weeks_to_harvest", "task": "regression", "time_column": "start_date",
                             "drop_columns": ["batch_id", "compartment", "plants_sold", "gas_m3", "electricity_kwh"]}}
    registry = bootstrap.build_registry(tmp_path / "models")
    train_use_case(write(tmp_path, manifest), registry)
    model = registry.resolve("weeks")

    from predictor.core.validation import FeatureValidator
    validator = FeatureValidator(model.feature_schema, reject_unknown_fields=False)

    def predict(frame):
        rows = [{k: r[k] for k in validator.feature_order} for r in frame.to_dict("records")]
        return np.asarray(model.runner.predict(validator.to_matrix(rows)).values)

    for lever, delta in [("temp_setpoint_c", 1.0), ("co2_ppm", 200.0)]:
        changed = df.assign(**{lever: df[lever] + delta})
        predicted = (predict(changed) - predict(df)).mean()
        truth = make_nursery.true_effects(sim, batches, {lever: delta})["weeks_to_harvest"]["mean_change"]
        assert truth < 0 and predicted < 0, (lever, truth, predicted)
        assert 0.3 * abs(truth) < abs(predicted) < 2 * abs(truth), (lever, truth, predicted)
