"""Real greenhouse data: hold out a whole greenhouse, sign-constrained linear model, the dataset script."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from predictor import bootstrap
from predictor.core.manifest import ManifestError, load_manifest
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


def test_without_signs_it_is_ordinary_least_squares():
    X, y = collinear()
    np.testing.assert_allclose(SignConstrainedLinearRegression([0, 0, 0]).fit(X, y).coef_,
                               LinearRegression().fit(X, y).coef_, atol=1e-6)


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
                  "Heat_cons": [2.0, 1.5, 1.0][:days], "ElecHigh": 0.5, "ElecLow": 0.3}).to_csv(
        team / "Resources.csv", index=False)


def test_logs_are_aggregated_to_greenhouse_days(tmp_path):
    fake_archive(tmp_path)
    df, info = agc.build(tmp_path, teams=["TeamX"])
    assert len(df) == 3 and list(df["date"]) == ["2019-12-16", "2019-12-17", "2019-12-18"]
    first = df.iloc[0]
    assert first["greenhouse"] == "TeamX" and first["season_day"] == 0
    assert first["lamp_hours"] == pytest.approx(6.0)
    assert first["heat_setpoint_c"] == pytest.approx(19.0) and first["rail_min_c"] == pytest.approx(40.0)
    assert first["radiation_j_cm2"] == pytest.approx(100 * 300 * 95 / 1e4)   # 95 samples of 100 W/m2
    assert first["heat_mj_m2"] == 2.0 and first["electricity_kwh_m2"] == pytest.approx(0.8)
    assert info["peak_share"] == pytest.approx(0.625) and info["eur_per_kwh"] == pytest.approx(0.065)


def test_days_with_too_few_samples_are_dropped(tmp_path):
    fake_archive(tmp_path)
    weather = pd.read_csv(tmp_path / "Weather" / "Weather.csv")
    weather.drop(index=range(288, 288 + 50)).to_csv(tmp_path / "Weather" / "Weather.csv", index=False)  # day 2: 238
    df, _ = agc.build(tmp_path, teams=["TeamX"])
    assert list(df["date"]) == ["2019-12-16", "2019-12-18"]
