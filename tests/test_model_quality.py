"""Model choice: categories for linear models, the yes/no cutoff, the blend's mix and a finer plan search."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import write_manifest
from predictor import bootstrap
from predictor.core import plan as planning
from predictor.core.manifest import ManifestError, ScenarioConfig, load_manifest
from predictor.trainer import models
from predictor.trainer.run import train_use_case


def train(root, manifest):
    path = write_manifest(root / "usecases", {"manifest_version": "1.0", **manifest})
    return train_use_case(path, bootstrap.build_registry(root / "models"))


def client(root):
    return TestClient(bootstrap.build_app(root / "usecases", root / "models"))


def csv(root, name, df):
    df.to_csv(root / name, index=False)
    return {"adapter": "csv-file", "path": str(root / name)}


# --------------------------------------------------------------------------- categories in linear models

def test_categories_become_0_1_columns_after_the_numbers():
    step = models.CategoryIndicators([0, 3, 0]).fit(np.zeros((1, 3)))
    X = np.array([[1.5, 0, 7], [2.5, 1, 8], [3.5, 2, 9], [4.5, 1.5, 0]], dtype=np.float32)   # 1.5: an imputed code
    assert step.transform(X).tolist() == [[1.5, 7, 0, 0], [2.5, 8, 1, 0], [3.5, 9, 0, 1], [4.5, 0, 0, 0]]
    assert step.sources() == [0, 2, 1, 1]


def test_a_linear_model_learns_a_category_that_is_not_a_scale(tmp_path):
    """Quality H=0, L=1, M=2 as codes, but the effect is L 0, M +10, H +5."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"quality": rng.choice(["L", "M", "H"], 900), "load": rng.uniform(0, 10, 900).round(2)})
    effect = df["quality"].map({"L": 0.0, "M": 10.0, "H": 5.0})
    df["output"] = (effect + 2 * df["load"] + rng.normal(0, 0.3, 900)).round(3)
    result = train(tmp_path, {"use_case_id": "by-quality", "data_source": csv(tmp_path, "quality.csv", df),
                              "training": {"target": "output", "task": "regression"},
                              "model": {"algorithms": ["linear_regression"]}})
    assert result["metrics"]["mae"] < 0.5                    # with the codes as a number it is off by about 3
    assert result["parity"]["passed"] and set(result["feature_importance"]) == {"quality", "load"}
    records = [{"features": {"quality": q, "load": 0.0}} for q in "LMH"]
    answers = client(tmp_path).post("/models/by-quality/predict", json={"records": records}).json()["predictions"]
    assert [round(a["prediction"]) for a in answers] == [0, 10, 5]


def test_declared_effects_still_hold_next_to_categories(tmp_path):
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"kind": rng.choice(["a", "b", "c"], 800), "heat": rng.uniform(0, 10, 800).round(2)})
    df["gas"] = (3 * df["heat"] + df["kind"].map({"a": 0, "b": 4, "c": 1}) + rng.normal(0, 1, 800)).round(2)
    result = train(tmp_path, {"use_case_id": "gas", "data_source": csv(tmp_path, "gas.csv", df),
                              "training": {"target": "gas", "task": "regression"},
                              "model": {"algorithms": ["linear_regression"]},
                              "scenario": {"levers": {"heat": {"min": 0, "max": 10, "step": 1,
                                                               "effects": {"gas": "increasing"}}}}})
    assert result["parity"]["passed"] and result["metrics"]["mae"] < 1.2
    assert result["scenario_constraints"]["held_at_zero"] == []


# --------------------------------------------------------------------------- the yes/no cutoff

def test_the_cutoff_moves_only_when_that_clearly_helps():
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(2000) < 0.05).astype(int))
    sharp = np.clip(y * 0.9 + rng.normal(0, 0.05, 2000), 0, 1)        # well separated: 0.5 is fine
    unsure = np.clip(0.03 + y * 0.25 + rng.normal(0, 0.05, 2000), 0, 1)   # never above 0.5: it must come down
    assert models.choose_threshold(y, sharp) == 0.5
    assert models.choose_threshold(y, unsure) < 0.3
    assert models.choose_threshold(y, unsure, {"missed": 20, "false_alarm": 1}) <= models.choose_threshold(y, unsure)
    assert models.choose_threshold(pd.Series([0] * 100 + [1] * 3), np.zeros(103)) == 0.5   # too few "yes" rows


@pytest.mark.parametrize("training, message", [
    ({"target": "y", "task": "regression", "decision_threshold": 0.3}, "only apply to a yes/no"),
    ({"target": "y", "decision_threshold": 1.5}, "between 0 and 1"),
    ({"target": "y", "decision_threshold": 0.3, "error_costs": {"missed": 5, "false_alarm": 1}}, "not both"),
    ({"outputs": {"y": {}}, "decision_threshold": 0.3}, "per output"),
])
def test_cutoff_settings_are_checked(tmp_path, training, message):
    path = write_manifest(tmp_path, {"manifest_version": "1.0", "use_case_id": "x",
                                     "data_source": {"adapter": "csv-file"}, "training": training})
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


def test_the_service_says_yes_from_the_cutoff_on(scratch):
    train(scratch, {"use_case_id": "pumps-cautious",
                    "data_source": {"adapter": "csv-file", "path": str(scratch / "data" / "pumps.csv")},
                    "training": {"target": "failure", "drop_columns": ["pump_id"], "decision_threshold": 0.3}})
    version = scratch / "models" / "pumps-cautious" / "1"
    descriptor = json.loads((version / "failure" / "descriptor.json").read_text(encoding="utf-8"))
    assert descriptor["decision_threshold"] == 0.3 and descriptor["parity"]["label_agreement"] == 1.0
    assert "Yes/no cutoff" in (version / "report.md").read_text(encoding="utf-8")
    rng = np.random.default_rng(1)
    records = [{"features": {"pump_type": "ABC"[i % 3], "vibration_mm_s": float(rng.normal(3.5, 1)),
                             "pressure_bar": float(rng.normal(6.2, 0.8)), "filter_ok": i % 2}} for i in range(200)]
    answers = client(scratch).post("/models/pumps-cautious/predict", json={"records": records}).json()["predictions"]
    assert all(a["prediction"] == ("1" if a["probabilities"]["1"] >= 0.3 else "0") for a in answers)
    assert any(0.3 <= a["probabilities"]["1"] < 0.5 for a in answers)   # the cutoff mattered for some of them


# --------------------------------------------------------------------------- the blend's mix

def test_the_blend_leans_on_the_part_that_fits(tmp_path):
    """A straight line plus noise: the linear part is right, the trees only add steps."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.uniform(0, 100, 600).round(2), "z": rng.normal(0, 1, 600).round(3)})
    df["y"] = (0.7 * df["x"] + rng.normal(0, 1, 600)).round(3)
    result = train(tmp_path, {"use_case_id": "line", "data_source": csv(tmp_path, "line.csv", df),
                              "training": {"target": "y", "task": "regression"}, "model": {"algorithms": ["blend"]}})
    assert result["blend_weight"] > 0.5 and result["parity"]["passed"]
    assert "linear model" in (tmp_path / "models" / "line" / "1" / "report.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- the plan search

def test_only_levers_the_grid_thinned_are_refined():
    two = ScenarioConfig.model_validate({"levers": {"a": {"min": 0, "max": 10, "step": 1},
                                                    "b": {"min": 0, "max": 5, "step": 1}}})
    assert planning.coarse_levers(two, ["a", "b"]) == {}
    five = ScenarioConfig.model_validate({"levers": {n: {"min": 0, "max": 100, "step": 1} for n in "abcde"}})
    gaps = planning.coarse_levers(five, list("abcde"))
    assert set(gaps) == set("abcde") and gaps["a"] == pytest.approx(34)   # 4 values: 0, 33, 67, 100
    around = planning.neighbours(five, {n: 33 for n in "abcde"}, "a", 5)
    assert [p["a"] for p in around] == [28, 29, 30, 31, 32, 34, 35, 36, 37, 38] and all(p["b"] == 33 for p in around)
    assert [p["a"] for p in planning.neighbours(five, {n: 0 for n in "abcde"}, "a", 2)] == [1, 2]   # within the lever


def test_with_five_levers_the_finder_looks_between_the_grid_values(tmp_path):
    """The grid tries 0, 33, 67 and 100 for every lever; the best value of `a` is 50."""
    rng = np.random.default_rng(8)
    levers = {name: {"min": 0, "max": 100, "step": 1} for name in "abcde"}
    df = pd.DataFrame({name: rng.uniform(0, 100, 3000).round(0) for name in levers})
    df["profit"] = (100 - 0.04 * (df["a"] - 50) ** 2 + 0.1 * df["b"] + rng.normal(0, 0.3, 3000)).round(3)
    train(tmp_path, {"use_case_id": "five", "data_source": csv(tmp_path, "five.csv", df),
                     "training": {"target": "profit", "task": "regression"}, "model": {"algorithms": ["lightgbm"]},
                     "scenario": {"levers": levers, "outputs": {"profit": {"value_eur": 1.0}}}})
    base = {"features": {"a": 10, "b": 10, "c": 50, "d": 50, "e": 50}}
    body = client(tmp_path).post("/models/five/scenario/best", json={"base": base}).json()
    assert body["searched"]["plans"] == 4 ** 5 + body["searched"]["refined"] and body["searched"]["refined"] > 0
    assert abs(body["best"]["plan"]["a"] - 50) <= 6          # the grid alone reaches 33 or 67 at best
