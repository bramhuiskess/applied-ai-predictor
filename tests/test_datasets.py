"""Ticket 1: the test datasets (KNMI parsing, arable builder, nursery simulator, faults)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import knmi  # noqa: E402
import make_arable  # noqa: E402
import make_nursery  # noqa: E402

KNMI_SAMPLE = """\
BRON / SOURCE: KONINKLIJK NEDERLANDS METEOROLOGISCH INSTITUUT (KNMI)
# STN,YYYYMMDD,   TG,   TN,   TX,   SQ,    Q,   RH,   UG, EV24
  260,20240601,  153,  101,  211,   85, 1890,   -1,   72,   31
  260,20240602,  141,   98,  176,   -1,  920,   55,   88,   12
  260,20240603,     ,   90,  160,   12, 1001,    0,   80,   15
"""


# --------------------------------------------------------------------------- KNMI

def test_knmi_parse_converts_units_and_special_values():
    df = knmi.parse(KNMI_SAMPLE)
    assert list(df["date"].dt.day) == [1, 2, 3]
    first = df.iloc[0]
    assert first["temp_mean_c"] == pytest.approx(15.3)
    assert first["rain_mm"] == 0            # -1 means "less than 0.05 mm"
    assert df.iloc[1]["sun_hours"] == 0     # same for sunshine
    assert df.iloc[1]["rain_mm"] == pytest.approx(5.5)
    assert first["evaporation_mm"] == pytest.approx(3.1)
    assert np.isnan(df.iloc[2]["temp_mean_c"])  # missing stays missing


def test_synthetic_weather_has_seasons():
    df = knmi.synthetic_daily("2020-01-01", "2020-12-31")
    by_month = df.groupby(df["date"].dt.month)["temp_mean_c"].mean()
    assert by_month[7] - by_month[1] > 10
    assert df["radiation_j_cm2"].min() > 0


# --------------------------------------------------------------------------- arable

def test_rain_deficit_accumulates_and_never_goes_below_zero():
    days = pd.date_range("2020-04-01", "2020-04-30")
    daily = pd.DataFrame({"date": days, "temp_mean_c": 12.0, "temp_min_c": 5.0, "temp_max_c": 18.0,
                          "rain_mm": 0.0, "radiation_j_cm2": 1500.0, "evaporation_mm": 2.0})
    daily.loc[10, "rain_mm"] = 100.0  # a downpour resets the deficit to zero, not below
    summary = make_arable.season_weather(daily, 2020, 4, 4)
    assert summary["max_rain_deficit_mm"] == pytest.approx(2.0 * 19)  # 19 dry days after the reset
    assert summary["season_rain_mm"] == 100.0
    assert summary["wet_days"] == 1


def test_arable_history_never_contains_the_year_itself():
    years = range(1994, 2006)
    yields = pd.DataFrame({"year": list(years), "province": "Flevoland", "crop": "winter_wheat",
                           "harvested_ha": 1000.0, "yield_t_ha": [float(y - 1990) for y in years]})
    weather = {"Flevoland": knmi.synthetic_daily("1994-01-01", "2005-12-31")}
    df = make_arable.build(yields, weather)
    row = df[df["year"] == 2001].iloc[0]
    assert row["yield_5y_avg"] == pytest.approx(np.mean([y - 1990 for y in range(1996, 2001)]))
    assert row["yield_t_ha"] == 11.0
    assert df["year"].min() == make_arable.FIRST_YEAR


# --------------------------------------------------------------------------- nursery simulator

@pytest.fixture(scope="module")
def nursery():
    daily = knmi.synthetic_daily("2015-06-01", "2025-12-31", seed=3)
    df, sim, batches = make_nursery.generate(daily, compartments=6, seed=5)
    return df, sim, batches


def test_nursery_is_reproducible():
    daily = knmi.synthetic_daily("2015-06-01", "2019-12-31")
    a, _, _ = make_nursery.generate(daily, compartments=3, seed=1, end="2019-12-31")
    b, _, _ = make_nursery.generate(daily, compartments=3, seed=1, end="2019-12-31")
    pd.testing.assert_frame_equal(a, b)


def test_nursery_values_are_plausible(nursery):
    df, _, _ = nursery
    assert len(df) > 200
    assert df["weeks_to_harvest"].between(4, make_nursery.MAX_WEEKS).all()
    assert (df["gas_m3"] >= 0).all() and (df["plants_sold"] > 0).all()
    assert (df.loc[df["lamp_type"] == "none", ["light_hours", "electricity_kwh"]] == 0).all().all()
    winter = df[df["start_week"].between(1, 8)]["gas_m3"].mean()
    summer = df[df["start_week"].between(24, 30)]["gas_m3"].mean()
    assert winter > 3 * summer


def test_gas_price_follows_the_2022_crisis():
    prices = make_nursery.gas_price(pd.to_datetime(["2019-06-01", "2022-08-26", "2024-06-01"]))
    assert prices[1] > 2 * prices[2] > prices[0]  # this caught a pandas 3 time-unit bug


def test_growers_react_to_the_gas_price(nursery):
    """The confounding a scenario model must see through."""
    df, _, _ = nursery
    assert df[["gas_price_eur_m3", "temp_setpoint_c"]].corr().iloc[0, 1] < -0.2


def test_counterfactual_without_changes_is_the_same_batch(nursery):
    _, sim, batches = nursery
    b = batches[10]
    assert sim.counterfactual(b) == sim.simulate(b)


def test_answer_key_effects_make_horticultural_sense(nursery):
    _, sim, batches = nursery
    warmer = make_nursery.true_effects(sim, batches, {"temp_setpoint_c": 1.0})
    light = make_nursery.true_effects(sim, batches, {"light_hours": 2.0})
    denser = make_nursery.true_effects(sim, batches, {"plant_density_m2": 5.0})
    assert warmer["weeks_to_harvest"]["mean_change"] < 0       # faster growth
    assert light["electricity_kwh"]["mean_change"] > 0         # lamps cost power...
    assert light["gas_m3"]["mean_change"] < 0                  # ...but their heat saves gas
    assert denser["plants_sold"]["mean_change"] > 0


def test_weather_features_do_not_depend_on_decisions(nursery):
    """Exogenous weather: a scenario changes levers, never the weather the model sees."""
    df, sim, _ = nursery
    row = df.iloc[20]
    assert sim.exogenous_weather(row["start_date"])["outside_temp_avg_c"] == row["outside_temp_avg_c"]


def test_faulty_copy_logs_exactly_what_was_broken(nursery):
    df, _, _ = nursery
    bad, log = make_nursery.inject_faults(df)
    faults = {f["fault"]: f for f in log}
    assert set(faults) == {"stuck_sensor", "spike", "missing", "unit_error", "duplicate_rows"}
    assert (bad["co2_ppm"] == 5000).sum() >= len(faults["spike"]["rows"])
    assert bad["radiation_avg_j_cm2"].isna().sum() == len(faults["missing"]["rows"])
    assert len(bad) == len(df) + len(faults["duplicate_rows"]["rows"])
    json.dumps(log)  # the log must be serialisable for later "bad data" tests
