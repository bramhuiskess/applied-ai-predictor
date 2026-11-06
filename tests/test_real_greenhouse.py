"""Real greenhouse data: hold out a whole greenhouse, sign-constrained linear model, the dataset script."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from predictor import bootstrap
from predictor.core.manifest import ManifestError, load_manifest
from predictor.trainer import models
from predictor.trainer.export import check_parity, make_thresholds_float32_safe, to_onnx
from predictor.trainer.models import SignConstrainedLinearRegression
from predictor.trainer.profile import TrainingError
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_greenhouse_challenge as agc  # noqa: E402


def write(root, manifest):
    directory = root / "usecases" / manifest["use_case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


# --------------------------------------------------------------------------- sign-constrained linear model

def collinear(n=400, seed=0):
    """'pipe' and 'setpoint' move together; in this sample 'pipe' gets a negative coefficient."""
    rng = np.random.default_rng(seed)
    setpoint = rng.normal(20, 1.5, n)
    pipe = setpoint * 2 + rng.normal(0, 0.3, n)
    outside = rng.normal(8, 4, n)
    y = 0.5 * setpoint - 0.2 * outside - 0.15 * pipe + rng.normal(0, 0.2, n)
    return np.column_stack([setpoint, pipe, outside]), y


def test_free_fit_gets_the_wrong_sign_the_constrained_fit_holds_it_at_zero():
    X, y = collinear()
    assert LinearRegression().fit(X, y).coef_[1] < 0
    model = SignConstrainedLinearRegression([1, 1, 0]).fit(X, y)
    assert model.coef_[0] > 0 and model.coef_[1] == pytest.approx(0, abs=1e-9)
    assert model.coef_[2] < 0                       # an input without a declared effect stays free


def test_without_signs_it_is_least_squares_with_a_tiny_ridge():
    X, y = collinear()
    X = (X - X.mean(axis=0)) / X.std(axis=0)   # the trainer standardises first
    ridge = Ridge(alpha=SignConstrainedLinearRegression.RIDGE * len(X)).fit(X, y)
    np.testing.assert_allclose(SignConstrainedLinearRegression([0, 0, 0]).fit(X, y).coef_, ridge.coef_, atol=1e-6)
    # setpoint and pipe are near-duplicates (r = 0.99): the ridge moves them a few percent, nothing more
    np.testing.assert_allclose(ridge.coef_, LinearRegression().fit(X, y).coef_, rtol=0.05)


def test_redundant_inputs_still_export_exactly():
    """Lamp power and crop always go together: without the ridge the fit cancelled huge coefficients."""
    rng = np.random.default_rng(1)
    crop = rng.integers(0, 2, 400).astype(float)
    X = pd.DataFrame({"lamp_hours": rng.uniform(0, 18, 400), "crop": crop, "lamp_power": 117 + 25.5 * crop})
    y = X.lamp_hours * X.lamp_power / 1000 + rng.normal(0, 0.05, 400)
    model = models.candidates("regression", monotone=[1, 0, 0])["linear_regression"].fit(X.to_numpy("float32"), y)
    assert np.abs(model.named_steps["model"].coef_).max() < 10
    parity = check_parity(model, to_onnx(model, 3, "regression"), X, {"task_type": "regression", "features": list(X)})
    assert parity["passed"], parity


def test_the_blend_exports_and_obeys_declared_effects():
    X, y = collinear()
    X = pd.DataFrame(X, columns=["setpoint", "pipe", "outside"])
    model = models.candidates("regression", monotone=[1, 1, 0])["blend"].fit(X.to_numpy("float32"), y)
    make_thresholds_float32_safe(model)
    assert check_parity(model, to_onnx(model, 3, "regression"), X, {"task_type": "regression", "features": list(X)})["passed"]
    grid = np.repeat(X.to_numpy("float32")[:50], 5, axis=0)
    grid[:, 0] = np.tile(np.linspace(16, 24, 5), 50)
    assert (np.diff(model.predict(grid).reshape(50, 5), axis=1) >= -1e-9).all()


@pytest.mark.parametrize("scores, best, expected", [
    ({"linear_regression": (2.99, 0.6), "blend": (2.63, 0.45), "lightgbm": (2.60, 0.46)}, "lightgbm", "linear_regression"),
    ({"linear_regression": (3.20, 0.1), "blend": (2.75, 0.45), "lightgbm": (2.60, 0.10)}, "lightgbm", "lightgbm"),
    ({"linear_regression": (3.20, 0.1), "blend": (2.65, 0.10), "lightgbm": (2.60, 0.10)}, "lightgbm", "blend"),
])
def test_the_simplest_model_within_one_standard_error_wins(scores, best, expected):
    validation = {n: {"mae": m, "mae_se": se} for n, (m, se) in scores.items()}
    assert models.one_standard_error_choice("regression", validation, best) == expected


# --------------------------------------------------------------------------- hold out a whole greenhouse

def greenhouses(path, n_per=120):
    rng = np.random.default_rng(3)
    rows = []
    for g, insulation in zip(["A", "B", "C", "D"], [1.0, 1.2, 0.9, 1.1]):
        for day in range(n_per):
            outside, setpoint = rng.normal(8, 4), rng.normal(19, 1)
            rows.append({"date": str(pd.Timestamp("2020-01-01") + pd.Timedelta(days=day))[:10], "greenhouse": g,
                         "setpoint": round(setpoint, 2), "outside": round(outside, 2),
                         "heat": round(max(0.0, 0.2 * insulation * (setpoint - outside) + rng.normal(0, 0.1)), 3)})
    pd.DataFrame(rows).to_csv(path, index=False)


def base_manifest(path, **training):
    return {"manifest_version": "1.0", "use_case_id": "heat",
            "data_source": {"adapter": "csv-file", "path": str(path)},
            "training": {"target": "heat", "task": "regression", "time_column": "date", "target_min": 0, **training},
            "scenario": {"levers": {"setpoint": {"min": 14, "max": 24, "step": 0.5,
                                                 "effects": {"heat": "increasing"}}}}}


def test_the_test_set_is_the_unseen_greenhouse(tmp_path):
    greenhouses(tmp_path / "days.csv")
    m = base_manifest(tmp_path / "days.csv", test_group={"column": "greenhouse", "values": ["B"]})
    result = train_use_case(write(tmp_path, m), bootstrap.build_registry(tmp_path / "models"))
    assert result["split"]["test"] == 120 and result["split"]["train"] + result["split"]["validation"] == 360
    assert result["split"]["test_group"] == {"column": "greenhouse", "values": ["B"]}
    assert "greenhouse" not in result["features"]   # used to split, never as an input
    assert "(test on unseen greenhouse: B)" in Path(result["report"]).read_text()
    # the model is chosen on the same question: each training greenhouse left out in turn
    assert result["split"]["validation_method"] == "leave one greenhouse out (3 folds)"
    assert all("mae_se" in c["validation"] for c in result["candidates"].values())
    assert result["metrics"]["r2"] > 0.6   # B is better insulated than any greenhouse it saw


def test_test_group_and_test_from_cannot_be_combined(tmp_path):
    m = base_manifest(tmp_path / "days.csv", test_group={"column": "greenhouse", "values": ["B"]},
                      test_from="2020-03-01")
    with pytest.raises(ManifestError, match="not both"):
        load_manifest(write(tmp_path, m) / "manifest.json")


@pytest.mark.parametrize("group, message", [
    ({"column": "farm", "values": ["B"]}, "not in the dataset"),
    ({"column": "greenhouse", "values": ["Z"]}, "to test on"),
])
def test_test_group_must_exist(tmp_path, group, message):
    greenhouses(tmp_path / "days.csv")
    with pytest.raises(TrainingError, match=message):
        train_use_case(write(tmp_path, base_manifest(tmp_path / "days.csv", test_group=group)),
                       bootstrap.build_registry(tmp_path / "models"))


# --------------------------------------------------------------------------- the dataset script (offline fixture)

def excel_day(ts):
    return (pd.Timestamp(ts) - pd.Timestamp("1899-12-30")) / pd.Timedelta(days=1)


def fake_archive(root, days=3):
    """The challenge layout: 5-minute logs with Excel day numbers, padded headers, 'NaN' strings."""
    times = pd.date_range("2019-12-16", periods=days * 288, freq="5min")
    t = [excel_day(x) for x in times]
    hour = times.hour + times.minute / 60
    (root / "Weather").mkdir(parents=True)
    pd.DataFrame({"%time": t, " Tout": 5.0, " Iglob": np.where((hour > 8) & (hour < 16), 100.0, 0.0),
                  " Windsp": 3.0, " RHout": 80.0}).to_csv(root / "Weather" / "Weather.csv", index=False)
    team = root / "TeamX"
    team.mkdir()
    climate = pd.DataFrame({"%time": t, "Tot_PAR_Lamps": np.where(hour < 6, 120.0, 0.0), "t_heat_vip": 19.0,
                            "t_ventlee_vip": 22.0, "EnScr": 50.0, "co2_vip": 800.0, "t_rail_min_vip": 40.0,
                            "t_grow_min_vip": 0.0})
    climate.loc[5, "t_heat_vip"] = np.nan
    climate.to_csv(team / "GreenhouseClimate.csv", index=False, na_rep="NaN")
    pd.DataFrame({"%Time": [excel_day(d) for d in pd.date_range("2019-12-16", periods=days)],
                  "Heat_cons": np.resize([2.0, 1.5, 1.0], days), "ElecHigh": 0.5, "ElecLow": 0.3}).to_csv(
        team / "Resources.csv", index=False)


def test_logs_are_aggregated_to_greenhouse_days(tmp_path):
    fake_archive(tmp_path)
    df, info = agc.build_second(tmp_path, teams=["TeamX"])
    df = agc.tidy(df)
    assert len(df) == 3 and list(df["date"]) == ["2019-12-16", "2019-12-17", "2019-12-18"]
    first = df.iloc[0]
    assert first["greenhouse"] == "TeamX 2020" and first["crop"] == "tomato"
    assert first["lamp_hours"] == pytest.approx(6.0)
    assert first["heat_setpoint_c"] == pytest.approx(19.0) and first["rail_min_c"] == pytest.approx(40.0)
    assert first["radiation_j_cm2"] == pytest.approx(100 * 300 * 95 / 1e4)   # 95 samples of 100 W/m2
    assert first["heat_mj_m2"] == 2.0 and first["electricity_kwh_m2"] == pytest.approx(0.8)
    assert info["peak_share"] == pytest.approx(0.625) and info["eur_per_kwh"] == pytest.approx(0.065)


def test_days_with_too_few_samples_are_dropped(tmp_path):
    fake_archive(tmp_path)
    weather = pd.read_csv(tmp_path / "Weather" / "Weather.csv")
    weather.drop(index=range(288, 288 + 50)).to_csv(tmp_path / "Weather" / "Weather.csv", index=False)  # day 2: 238
    df, _ = agc.build_second(tmp_path, teams=["TeamX"])
    assert list(agc.tidy(df)["date"]) == ["2019-12-16", "2019-12-18"]


def fake_archive_2018(root, days=3):
    """Edition 1: weather rows line up with the climate log, setpoints in vip.csv, one team with day periods."""
    times = pd.date_range("2018-09-01", periods=days * 288, freq="5min")
    t = [excel_day(x) for x in times]
    hour = times.hour + times.minute / 60
    pd.DataFrame({"time": times.strftime("%-m/%-d/%Y"), "Tout": 12.0, "Iglob": 50.0, "Windsp": 2.0,
                  "Rhout": 70.0}).to_csv(root / "meteo.csv", index=False)
    for folder, periods in [("Reference(Growers)", False), ("AiCU", True)]:
        (root / folder).mkdir(parents=True)
        pd.DataFrame({"GHtime": t, "AssimLight": np.where(hour < 10, 100.0, 0.0), "EnScr": 20.0}).to_csv(
            root / folder / "Greenhouse_climate.csv", index=False)
        vip = {"time": t, "CO2_Vip": 600.0, "MinPipeGrow_Vip": 0.0, "MinPipeLow_Vip": 35.0}
        if periods:
            vip.update({f"HeatTemp_Vip_{i}": 18.0 + i for i in range(1, 7)})      # mean 21.5
            vip.update({f"VentLeew_Vip_{i}": 22.0 for i in range(1, 7)})
        else:
            vip.update({"HeatTemp_Vip": 20.0, "VentLeew_Vip": 23.0})
        pd.DataFrame(vip).to_csv(root / folder / "vip.csv", index=False)
        pd.DataFrame({"time": [excel_day(d) for d in pd.date_range("2018-09-01", periods=days)],
                      "Heating_Energy": 0.7, "Electricity_Lamp": 1.17}).to_csv(root / folder / "ResourceCalculations.csv",
                                                                                index=False)


def test_edition_1_is_read_with_its_own_layout(tmp_path):
    fake_archive_2018(tmp_path)
    df = agc.tidy(agc.build_first(tmp_path, {"Reference(Growers)": "Reference", "AiCU": "AiCU"}))
    assert set(df["greenhouse"]) == {"Reference 2018", "AiCU 2018"} and set(df["crop"]) == {"cucumber"}
    ref, aicu = df[df.greenhouse == "Reference 2018"].iloc[0], df[df.greenhouse == "AiCU 2018"].iloc[0]
    assert ref["heat_setpoint_c"] == 20.0 and aicu["heat_setpoint_c"] == pytest.approx(21.5)   # mean of six periods
    assert ref["lamp_hours"] == pytest.approx(10.0) and ref["outside_temp_c"] == 12.0
    assert ref["heat_mj_m2"] == 0.7   # already MJ/m2, see the module docstring
    assert ref["lamp_power_w_m2"] == pytest.approx(116.875)


def test_harvest_timing_uses_a_fixed_window_after_flowering(tmp_path):
    fake_archive(tmp_path, days=60)
    climate = pd.read_csv(tmp_path / "TeamX" / "GreenhouseClimate.csv")
    day = (pd.to_datetime(climate["%time"], unit="D", origin="1899-12-30").dt.floor("D")
           - pd.Timestamp("2019-12-16")).dt.days
    climate["Tair"] = np.where(day < 39, 20.0, 30.0)      # warm only after the 39-day window
    climate["Tot_PAR"] = 200.0
    climate["CO2air"], climate["Rhair"] = 800.0, 85.0
    climate.to_csv(tmp_path / "TeamX" / "GreenhouseClimate.csv", index=False, na_rep="NaN")
    pd.DataFrame({"%time": [excel_day("2020-02-04")], "ProdA": [0.5], "ProdB": [0.0],
                  "Truss development time ": [50]}).to_csv(tmp_path / "TeamX" / "Production.csv", index=False)
    h = agc.harvest_timing(tmp_path, teams=["TeamX"])
    assert len(h) == 1
    row = h.iloc[0]
    assert row["flowering_date"] == "2019-12-16" and row["days_to_harvest"] == 50
    assert row["air_temp_c"] == pytest.approx(20.0)          # days 40-50 are not in the window
    assert row["light_mol_m2_day"] == pytest.approx(200 * 300 * 288 / 1e6)
