"""Harvest months ahead: the crop model and the seasons made with it."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.trainer.run import train_use_case

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import greenhouse_heat as heating  # noqa: E402
import knmi  # noqa: E402
import make_tomato_season as season  # noqa: E402
import tomato_crop as crop  # noqa: E402


# --------------------------------------------------------------------------- the crop model

def harvest(temp=22.0, light=20.0, days=200):
    return crop.simulate(np.full(days, temp), np.full(days, light))


def test_a_truss_ripens_after_its_temperature_sum():
    first = int(np.argmax(harvest(temp=22.4) > 0))
    # flowering starts on day 15; 705 degree-days at 22.4 - 7.4 = 15 per day is 47 days
    assert first == crop.FIRST_FLOWERING_DAY + 47


def test_warmer_is_sooner_and_colder_is_later():
    first = {t: int(np.argmax(harvest(temp=t) > 0)) for t in (20.0, 22.0, 24.0)}
    assert first[24.0] < first[22.0] < first[20.0]
    assert first[22.0] - first[24.0] == pytest.approx(3 * 2, abs=3)      # a few days per degree


def test_light_is_what_becomes_fruit():
    assert harvest(light=0.0).sum() == 0
    low, high = harvest(light=10.0).sum(), harvest(light=12.0).sum()
    assert high == pytest.approx(low * 1.2, rel=0.02)                    # 20% more light, 20% more fruit
    # but a plant cannot use endless light: the trusses limit it
    assert harvest(light=200.0).sum() < harvest(light=100.0).sum() * 1.05


def test_nothing_is_harvested_below_the_base_temperature():
    assert harvest(temp=crop.BASE_TEMP_C - 1).sum() == 0


def test_calibration_finds_the_parameters_back():
    truth = crop.CropParameters(6.5, 0.02)
    rng = np.random.default_rng(0)
    real = {}
    for name in "ABCD":
        temp, light = rng.normal(22, 1, 170), np.linspace(12, 30, 170) + rng.normal(0, 2, 170)
        real[name] = pd.DataFrame({"temp": temp, "light": light, "kg": crop.simulate(temp, light, truth, True)},
                                  index=pd.date_range("2020-01-01", periods=170))
    found = crop.calibrate(real)
    assert found.fruit_g_per_mol == pytest.approx(6.5, rel=0.03)
    assert found.sink_g_per_degree_day == pytest.approx(0.02, rel=0.1)
    assert crop.validate(real)["season_total_pct"].max() < 2


# --------------------------------------------------------------------------- the seasons

@pytest.fixture(scope="module")
def daily():
    return knmi.synthetic_daily("2015-08-01", "2025-12-31", seed=3)


@pytest.fixture(scope="module")
def table(daily):
    return season.generate(daily, seasons=900, seed=5)


def test_lamps_follow_the_darkness():
    hours = season.lamp_hours(np.array([300.0] * 10 + [2500.0] * 10), 16.0)
    assert hours[0] == 16.0 and hours[-1] == 0.0


def test_one_row_is_one_decision(table):
    assert table.notna().all().all() and table["season_id"].is_unique
    assert (table["harvest_1_month_kg_m2"] <= table["harvest_3_months_kg_m2"]).all()
    assert (table["harvest_3_months_kg_m2"] <= table["harvest_5_months_kg_m2"]).all()
    start = table[table["days_since_planting"] == 0]
    assert (start["harvest_1_month_kg_m2"] == 0).all()                    # no fruit in the first month after planting
    assert (start["temp_so_far_c"] == start["temp_plan_c"]).all()
    changed = table[table["temp_so_far_c"] != table["temp_plan_c"]]
    assert 0.2 < len(changed) / len(table) < 0.6                           # some growers change course


def test_the_answer_key_respects_the_crop(table, daily):
    daily = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    for _, row in table[table["days_since_planting"].between(28, 84)].head(25).iterrows():
        base = season.counterfactual(row, daily)
        more_light = season.counterfactual(row, daily, lamp_hours_plan=4.0)
        warmer = season.counterfactual(row, daily, temp_plan_c=1.0)
        assert all(more_light[k] >= base[k] - 1e-9 for k in season.HORIZONS)   # more light is never less fruit
        assert more_light["electricity_5_months_kwh_m2"] > base["electricity_5_months_kwh_m2"]
        assert more_light["heating_5_months_mj_m2"] <= base["heating_5_months_mj_m2"] + 1e-9   # lamps give some heat
        assert warmer["heating_5_months_mj_m2"] > base["heating_5_months_mj_m2"]
        assert warmer["electricity_5_months_kwh_m2"] == base["electricity_5_months_kwh_m2"]
        assert warmer["heat_last_4_weeks_mj_m2"] == base["heat_last_4_weeks_mj_m2"]   # the past does not change
        assert warmer["harvest_1_month_kg_m2"] >= base["harvest_1_month_kg_m2"] - 1e-9   # warmer ripens sooner
        assert season.counterfactual(row, daily) == base                   # no randomness in the answer key


def test_the_trained_model_follows_the_crop_model(table, daily, tmp_path):
    """End to end: train the real manifest on these seasons and compare its scenarios with the answer key."""
    table.to_csv(tmp_path / "tomato_season.csv", index=False)
    manifest = json.loads((ROOT / "usecases" / "tomato-season" / "manifest.json").read_text())
    manifest["data_source"]["path"] = str(tmp_path / "tomato_season.csv")
    directory = tmp_path / "usecases" / "tomato-season"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    result = train_use_case(directory, bootstrap.build_registry(tmp_path / "models"))
    five = result["outputs"]["harvest_5_months_kg_m2"]
    assert five["metrics"]["mae"] < five["reference"]["persistence_mae"]    # better than "a normal plan in an average year"

    client = TestClient(bootstrap.build_app(tmp_path / "usecases", tmp_path / "models"))
    order = client.get("/models/tomato-season").json()["feature_schema"]["x-feature-order"]
    clean = daily.dropna(subset=["temp_mean_c", "radiation_j_cm2"]).reset_index(drop=True)
    true, predicted = [], []
    rows = table[(table["decision_date"] >= "2023-07-01") & table["days_since_planting"].between(28, 112)
                 & (table["lamp_hours_plan"] <= 14)].head(40)
    for _, row in rows.iterrows():
        features = {k: float(row[k]) for k in order}
        body = client.post("/models/tomato-season/scenario", json={
            "base": {"features": features},
            "scenarios": [{"name": "more light", "changes": {"lamp_hours_plan": features["lamp_hours_plan"] + 4}}]}).json()
        change = body["scenarios"][0]["difference"]["harvest_5_months_kg_m2"]["change"]
        assert change >= 0                                                 # the declared effect holds for every case
        predicted.append(change)
        true.append(season.counterfactual(row, clean, lamp_hours_plan=4.0)["harvest_5_months_kg_m2"]
                    - season.counterfactual(row, clean)["harvest_5_months_kg_m2"])
    assert np.mean(predicted) == pytest.approx(np.mean(true), rel=0.5)       # the right size on average
    # the net result is the three money items of the last case, at that grower's prices
    d = body["scenarios"][0]["difference"]
    expected = (d["harvest_5_months_kg_m2"]["change"] * features["tomato_price_eur_kg"]
                - d["heating_5_months_mj_m2"]["change"] * features["heat_price_eur_mj"]
                - d["electricity_5_months_kwh_m2"]["change"] * features["electricity_price_eur_kwh"])
    assert body["scenarios"][0]["net_eur"] == pytest.approx(expected, abs=0.02)
    assert d["harvest_5_months_kg_m2"]["eur"] > 0 > d["electricity_5_months_kwh_m2"]["eur"]   # income up, cost up
    assert d["harvest_1_month_kg_m2"]["eur"] is None                       # shorter horizons are not counted twice


# --------------------------------------------------------------------------- what the plan costs

def test_heat_balance_follows_the_physics():
    base = heating.heat_balance(np.array([22.0]), np.array([5.0]), np.array([300.0]), np.array([1.5]))[0]
    assert heating.heat_balance(np.array([23.0]), np.array([5.0]), np.array([300.0]), np.array([1.5]))[0] > base      # warmer inside
    assert heating.heat_balance(np.array([22.0]), np.array([10.0]), np.array([300.0]), np.array([1.5]))[0] < base     # milder outside
    assert heating.heat_balance(np.array([22.0]), np.array([5.0]), np.array([1500.0]), np.array([1.5]))[0] < base     # sunnier
    assert heating.heat_balance(np.array([22.0]), np.array([5.0]), np.array([300.0]), np.array([3.0]))[0] < base      # more lamps
    assert heating.heat_balance(np.array([18.0]), np.array([25.0]), np.array([2500.0]), np.array([0.0]))[0] == 0       # never negative


def test_a_greenhouse_gets_its_own_factor_from_its_own_use():
    balance = np.array([2.0, 3.0, 1.0])
    assert heating.own_factor(balance * 1.6, balance) == pytest.approx(1.6)
    assert heating.own_factor(np.zeros(3), np.zeros(3)) == 1.0


def test_heat_calibration_finds_the_parameters_back():
    truth = heating.HeatParameters(0.25, 0.08, 0.2, 0.4)
    rng = np.random.default_rng(1)
    real = {}
    for name in "ABCD":
        days = pd.DataFrame({"temp": rng.normal(21, 1, 160), "outside": rng.normal(7, 4, 160), "radiation": rng.uniform(100, 1500, 160),
                             "lamp_kwh": rng.uniform(0, 2.2, 160)}, index=pd.date_range("2020-01-01", periods=160))
        days["heat"] = heating.heat_balance(days.temp, days.outside, days.radiation, days.lamp_kwh, truth)
        real[name] = days
    found = heating.calibrate(real)
    assert found.loss_mj_per_degree == pytest.approx(0.25, rel=0.03) and found.lamp_share == pytest.approx(0.2, rel=0.1)


def test_the_own_factor_is_a_level_that_drifts_slowly():
    path = season.own_factor_path(7, 328)
    assert len(path) == 328 and (path > 0).all()
    assert len(np.unique(path)) <= 12                                       # one value per month
    assert np.array_equal(path, season.own_factor_path(7, 328))             # the same greenhouse every time
    levels = [season.own_factor_path(s, 328)[0] for s in range(300)]
    assert 0.8 < np.median(levels) < 1.25 and np.percentile(levels, 95) / np.percentile(levels, 5) > 2.5   # growers differ a lot


def test_the_last_four_weeks_tell_how_a_greenhouse_heats(table):
    later = table[table["days_since_planting"] >= 28]
    assert later["heat_last_4_weeks_mj_m2"].corr(later["heating_5_months_mj_m2"]) > 0.5
    assert (table["heat_last_4_weeks_mj_m2"] > 0).all()                     # also at planting: the greenhouse was in use


def test_prices_are_the_growers_own_and_do_not_mark_the_year(table):
    assert table["tomato_price_eur_kg"].between(1.0, 5.5).all()
    assert table["heat_price_eur_mj"].between(0.005, 0.04).all() and table["electricity_price_eur_kwh"].between(0.04, 0.30).all()
    by_year = table.groupby(table["decision_date"].str[:4])["heat_price_eur_mj"].mean()
    assert by_year.max() / by_year.min() < 1.25
    january, june = table[table["decision_date"].str[5:7] == "01"], table[table["decision_date"].str[5:7] == "06"]
    assert january["tomato_price_eur_kg"].mean() > june["tomato_price_eur_kg"].mean()      # winter tomatoes pay more


def test_the_net_result_is_income_minus_energy(table):
    row = table.iloc[10]
    outcome = {name: float(row[name]) for name in season.OUTPUTS}
    expected = (row["harvest_5_months_kg_m2"] * row["tomato_price_eur_kg"] - row["heating_5_months_mj_m2"] * row["heat_price_eur_mj"]
                - row["electricity_5_months_kwh_m2"] * row["electricity_price_eur_kwh"])
    assert season.net_eur(row, outcome) == pytest.approx(expected)
