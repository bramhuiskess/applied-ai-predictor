"""The final test set: the 4th greenhouse challenge is read into the same columns as the training data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import final_test  # noqa: E402


def fake_edition(folder, days=10):
    """Two teams, 5-minute logs with local clock times, dimmable lamps, zero written as 1e-10."""
    folder.mkdir(parents=True)
    times = pd.date_range("2024-10-20", periods=days * 288, freq="5min")
    stamp = times.strftime("%Y-%m-%d %H:%M:%S") + "+02:00"
    pd.DataFrame({"time": stamp, "weather/air_temperature.outside": 10.0, "weather/radiation_global": 100.0,
                  "weather/wind_speed": 3.0, "weather/relative_humidity.outside": 80.0}).to_csv(folder / "weather.csv", index=False)
    night = times.hour < 6
    for file in final_test.TEAMS:
        level = np.where(night, 50.0, 0.0)                                   # lamps at half power for 6 hours
        pd.DataFrame({
            "time": stamp, "compartment/lamps_activation_percentage": level,
            "energy/electricity_use.lighting": np.where(night, 0.5 * 0.060 / 12, 1e-10),   # 60 W/m2 at full power
            "energy/energy_use.heating": np.where(night, 0.002, 1e-10),
            "compartment/heating_temperature_vip": 18.0, "compartment/ventilation_temperature_lee_side_vip": 21.0,
            "compartment/screen_energy/screen_position": 40.0, "compartment/co2_concentration_vip": 600.0,
            "compartment/minimum_pipe_temperature_vip": 30.0}).to_csv(folder / f"{file}.csv", index=False)


def test_the_final_test_has_the_columns_of_the_training_data(tmp_path):
    fake_edition(tmp_path / "ts")
    df = final_test.build(tmp_path / "ts")
    assert len(df) == 6 * 10 and set(df["greenhouse"]) == {f"{t} 2024" for t in final_test.TEAMS.values()}
    row = df.iloc[0]
    assert row["crop"] == "tomato" and row["grow_min_c"] == 0 and row["rail_min_c"] == 30
    assert row["lamp_hours"] == pytest.approx(3.0)                 # 6 hours at half power = 3 hours at full power
    assert row["lamp_power_w_m2"] == pytest.approx(60.0)           # taken from the electricity log
    assert row["electricity_kwh_m2"] == pytest.approx(0.18)        # 3 h x 60 W/m2
    assert row["lamp_kwh_m2"] == pytest.approx(0.18)               # what the electricity model gets
    assert row["heat_mj_m2"] == pytest.approx(72 * 0.002)          # the 1e-10 placeholders count as zero
    assert row["radiation_j_cm2"] == pytest.approx(100 * 300 * 288 / 1e4)
    for column in ("heat_setpoint_c", "vent_setpoint_c", "energy_screen_pct", "co2_setpoint_ppm", "outside_temp_c",
                   "outside_temp_min_c", "wind_ms", "outside_rh_pct"):
        assert column in df.columns and df[column].notna().all()


def test_the_week_ago_yardstick_is_the_same_greenhouse_seven_days_earlier(tmp_path):
    fake_edition(tmp_path / "ts")
    df = final_test.build(tmp_path / "ts")
    one = df[df["greenhouse"] == "Reference 2024"].sort_values("date")
    assert one["heat_week_ago_mj_m2"].isna().sum() == 7 and one["heat_week_ago_mj_m2"].iloc[7] == one["heat_mj_m2"].iloc[0]


def test_the_summary_compares_the_model_with_plain_guesses():
    dates = pd.date_range("2024-10-07", periods=28).strftime("%Y-%m-%d")
    out = pd.DataFrame({"greenhouse": "A", "date": dates, "actual": np.linspace(1, 2, 28)})
    out["predicted"], out["week_ago"] = out["actual"] * 1.1, out["actual"].shift(7)
    table = final_test.summary(out, training_mean=5.0)
    assert table.loc["model", "season total % (median)"] == pytest.approx(10.0)
    assert table.loc["model", "weekly error % (median)"] == pytest.approx(10.0)
    assert table.loc["plain guess: the average day of the training data", "daily error"] > table.loc["model", "daily error"]
